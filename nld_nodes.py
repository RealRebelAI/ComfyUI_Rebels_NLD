# ComfyUI_Rebels_NLD - NVIDIA NL-Diffusion-Image (masked discrete diffusion LM + IBQ VQ decoder)
# RealRebelAI.
#
# Model selection is DROPDOWN-ONLY (scans models/unet + models/diffusion_models for .gguf,
# models/vae for the vqvae split). Configs/tokenizer/modeling code live in ./model_assets/.
#
# CRITICAL dequant note: city96's dequant functions need the quant TYPE and ORIGINAL SHAPE
# passed explicitly. Handing them a bare tensor of quantized bytes silently returns the
# compressed bytes as if they were weights (no error, garbage math downstream).

import os
# Fix fragmentation OOM (6.6GB allocated, 0 free, 1GB request fails): let CUDA
# satisfy large allocs from non-contiguous free space. Must be set before torch inits.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
import inspect
import textwrap
import re

import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F

import folder_paths
import comfy.model_management as mm
import gguf
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights

HERE = os.path.dirname(os.path.realpath(__file__))
MODEL_ASSETS_DIR = os.path.join(HERE, "model_assets")

# ---------------------------------------------------------------------------
# dropdowns
# ---------------------------------------------------------------------------
GGUF_SCAN_DIRS = [
    os.path.join(folder_paths.models_dir, "unet"),
    os.path.join(folder_paths.models_dir, "diffusion_models"),
]


def get_gguf_files():
    files = []
    for d in GGUF_SCAN_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".gguf") and f not in files:
                    files.append(f)
    return files if files else ["<put dLM .gguf in models/diffusion_models>"]


def resolve_gguf(name):
    for d in GGUF_SCAN_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise RuntimeError("dLM GGUF '{}' not found in models/unet or models/diffusion_models".format(name))


# ---------------------------------------------------------------------------
# city96 dequant import
# ---------------------------------------------------------------------------
comfy_gguf_path = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-GGUF")
if comfy_gguf_path not in sys.path:
    sys.path.append(comfy_gguf_path)

try:
    import dequant
except ImportError:
    dequant = None
    print("[NLD] Warning: ComfyUI-GGUF dequant module not found. Quantized models will not run.")


def dequant_bytes(qbytes, qtype, oshape, dtype):
    """Dequantize raw quant bytes. qtype and oshape MUST be passed explicitly."""
    if qtype == gguf.GGMLQuantizationType.F32:
        return qbytes.view(torch.float32).reshape(oshape).to(dtype)
    if qtype == gguf.GGMLQuantizationType.F16:
        return qbytes.view(torch.float16).reshape(oshape).to(dtype)
    if dequant is None:
        raise RuntimeError("[NLD] ComfyUI-GGUF dequant module not found.")
    return dequant.dequantize(qbytes, qtype, oshape, dtype=dtype)


# ---------------------------------------------------------------------------
# quantized module wrappers
# ---------------------------------------------------------------------------
class GGUFLinearWrapper(torch.nn.Module):
    def __init__(self, qbytes, qtype, oshape, bias=None):
        super().__init__()
        self.register_buffer("qdata", qbytes, persistent=False)
        self.qtype = qtype
        self.oshape = tuple(oshape)
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None
        self._w_cache = None

    _reported = False

    # output features above this -> chunk the projection to avoid a full-vocab spike
    BIG_OUT = 40000

    def forward(self, x):
        if not GGUFLinearWrapper._reported:
            GGUFLinearWrapper._reported = True
            print("[NLD] FIRST LINEAR: x.device={} x.dtype={} qdata.device={}".format(
                x.device, x.dtype, self.qdata.device))
        # cached bf16 weight (per-generate) short-circuits the dequant entirely
        cached = getattr(self, "_w_cache", None)
        if cached is not None and cached.device == x.device:
            b = self.bias.to(x.device, x.dtype) if self.bias is not None else None
            return F.linear(x, cached.to(x.dtype), b)

        q = self.qdata if self.qdata.device == x.device else self.qdata.to(x.device, non_blocking=True)
        out_features = int(self.oshape[0])

        if out_features >= GGUFLinearWrapper.BIG_OUT:
            # dequant full weight (Q4 bytes are small); chunk the matmul over output rows
            w = dequant_bytes(q, self.qtype, self.oshape, x.dtype)
            b = self.bias.to(x.device, x.dtype) if self.bias is not None else None
            out = torch.empty(*x.shape[:-1], out_features, dtype=x.dtype, device=x.device)
            step = 16384
            for i in range(0, out_features, step):
                j = min(i + step, out_features)
                bi = b[i:j] if b is not None else None
                out[..., i:j] = F.linear(x, w[i:j], bi)
            del w
            if q is not self.qdata:
                del q
            return out

        w = dequant_bytes(q, self.qtype, self.oshape, x.dtype)
        b = self.bias.to(x.device, x.dtype) if self.bias is not None else None
        out = F.linear(x, w, b)

        if _DequantCache.enabled and self._w_cache is None:
            nb = w.numel() * w.element_size()
            if _DequantCache.can_fit(nb):
                self._w_cache = w.detach()
                _DequantCache.note(nb)
                w = None
        if w is not None:
            del w
        if q is not self.qdata:
            del q
        return out


class GGUFEmbeddingWrapper(torch.nn.Module):
    """Row-gather embedding: dequantize ONLY the rows being looked up."""

    def __init__(self, qbytes, qtype, oshape):
        super().__init__()
        n_rows = int(oshape[0])
        self.register_buffer("qdata", qbytes.reshape(n_rows, -1), persistent=False)
        self.qtype = qtype
        self.oshape = tuple(oshape)

    def forward(self, indices):
        flat = indices.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        rows = self.qdata[uniq.to(self.qdata.device)].contiguous()
        rows = rows.to(indices.device, non_blocking=True)
        w = dequant_bytes(rows, self.qtype, (rows.shape[0], self.oshape[1]), torch.bfloat16)
        out = w[inv.to(indices.device)].reshape(*indices.shape, self.oshape[1])
        del w, rows
        return out


def _nld_lean_x0p(logits, x0, chunk=256):
    """Exactly gather(softmax(logits,-1), x0) without building the full-vocab softmax.

    p(x0) = exp(logits[x0] - logsumexp(logits, -1)). Computed in fp32 over sequence
    chunks so peak memory is ~chunk x vocab instead of batch x seq x vocab (2GB).
    This value feeds confidence_policy='mmada' (log(x0_p) + gumbel -> topk), which
    decides which tokens unmask each step. It must be correct or the unmask order is
    random and the image comes out locally shredded.
    """
    b, l, _ = logits.shape
    out = torch.empty((b, l), dtype=torch.float32, device=logits.device)
    idx = x0.long().unsqueeze(-1)
    for i in range(0, l, chunk):
        j = min(i + chunk, l)
        lg = logits[:, i:j, :].float()
        lse = torch.logsumexp(lg, dim=-1)
        chosen = torch.gather(lg, -1, idx[:, i:j, :]).squeeze(-1)
        out[:, i:j] = torch.exp(chosen - lse)
        del lg, lse, chosen
    return out.to(logits.dtype)


# ---------------------------------------------------------------------------
# Dynamic VRAM Patcher for In-Place Math
# ---------------------------------------------------------------------------
def patch_text_to_image_for_vram(model):
    """
    Dynamically rewrites the model's text_to_image source code in memory 
    to force CFG mathematical operations to happen in-place, eliminating 
    the 1GB+ VRAM spike during intermediate tensor allocation.
    """
    model_cls = model.__class__
    if getattr(model_cls, "_vram_math_patched", False):
        return
        
    try:
        src = inspect.getsource(model_cls.text_to_image)
        src = textwrap.dedent(src)
        
        # (a) in-place CFG math (kills a ~1GB spike)
        target = r"logits\s*=\s*\(1\.0\s*\+\s*guidance_scale\)\s*\*\s*logits\s*-\s*guidance_scale\s*\*\s*logits_un"
        replacement = (
            "logits.mul_(1.0 + guidance_scale)\n"
            "            logits_un.mul_(guidance_scale)\n"
            "            logits.sub_(logits_un)\n"
            "            del logits_un"
        )
        new_src = re.sub(target, replacement, src)

        # (b) CRITICAL: never materialize the full-vocab softmax (2GB), but keep x0_p
        #     mathematically identical to gather(softmax(logits), x0). The confidence
        #     ranking at confidence_policy=='mmada' depends on x0_p; feeding it garbage
        #     makes the model unmask tokens in random order -> shredded ears/eyes/text.
        #     log p = logits[x0] - logsumexp(logits, -1); p = exp(log p).
        new_src = new_src.replace(
            "            probs = logits.softmax(dim=-1)",
            "            probs = None  # replaced by _nld_lean_x0p (identical values, chunked)"
        )
        new_src = new_src.replace(
            "                x0 = dists.Categorical(logits=logits / temperature).sample()\n"
            "                x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)",
            "                x0 = dists.Categorical(logits=logits / temperature).sample()\n"
            "                x0_p = _nld_lean_x0p(logits, x0)"
        )
        new_src = new_src.replace(
            "                x0 = logits.argmax(-1)\n"
            "                x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)",
            "                x0 = logits.argmax(-1)\n"
            "                x0_p = _nld_lean_x0p(logits, x0)"
        )
        
        if new_src != src:
            namespace = dict(sys.modules[model_cls.__module__].__dict__)
            namespace["_nld_lean_x0p"] = _nld_lean_x0p
            locs = {}
            exec(new_src, namespace, locs)
            setattr(model_cls, "text_to_image", locs["text_to_image"])
            setattr(model_cls, "_vram_math_patched", True)
            print("[NLD] Patched: in-place CFG + lean exact x0_p (no 2GB softmax, correct confidence).")
        else:
            print("[NLD] CFG math line not found for patching.")
            
    except Exception as e:
        print(f"[NLD] Dynamic VRAM patch failed: {e}")


# ---------------------------------------------------------------------------
# Gumbel-Max Sampler for VRAM Optimization
# ---------------------------------------------------------------------------
def _move_transformer_to(model, dev, exclude_vqvae=True):
    """Move the LM transformer (encoder.layers + embeddings/heads) to `dev`, leaving the
    vqvae where it is. Used to evacuate the 5.5GB transformer to CPU during vqvae decode
    so the conv decoder gets the whole GPU (fixes decode-time pagefile thrash)."""
    import torch as _t
    enc = getattr(model, "encoder", model)
    moved = 0
    for name, mod in enc.named_modules():
        if exclude_vqvae and name.startswith("vqvae"):
            continue
        for pn, p in list(mod.named_parameters(recurse=False)):
            if p.device.type != dev.split(":")[0] and p.device.type != "meta":
                try:
                    setattr(mod, pn, _t.nn.Parameter(p.to(dev), requires_grad=False))
                    moved += 1
                except Exception:
                    pass
        for bn, b in list(mod.named_buffers(recurse=False)):
            if b is not None and b.device.type != dev.split(":")[0] and b.device.type != "meta":
                try:
                    mod._buffers[bn] = b.to(dev)
                except Exception:
                    pass
    return moved


# ---------------------------------------------------------------------------
# Dequant cache: within one generate() call the same Q4 bytes are dequantized on
# every forward (steps x cfg passes x layers = thousands of redundant dequants).
# Caching the bf16 weight per module removes nearly all of that compute. Full-model
# bf16 is ~18GB so we cache under a VRAM budget, smallest-first (attention projections
# are hottest and cheapest), and drop everything at the end of the run.
# ---------------------------------------------------------------------------
class _DequantCache:
    enabled = False
    budget_bytes = 0
    used_bytes = 0

    @classmethod
    def start(cls, budget_gb):
        cls.enabled = budget_gb > 0
        cls.budget_bytes = int(budget_gb * (1024 ** 3))
        cls.used_bytes = 0

    @classmethod
    def can_fit(cls, nbytes):
        return cls.enabled and (cls.used_bytes + nbytes) <= cls.budget_bytes

    @classmethod
    def note(cls, nbytes):
        cls.used_bytes += nbytes

    @classmethod
    def stop(cls, modules):
        n = 0
        for m in modules:
            if getattr(m, "_w_cache", None) is not None:
                m._w_cache = None
                n += 1
        cls.enabled = False
        cls.used_bytes = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return n


class GumbelCategorical:
    """Drop-in replacement for torch.distributions.Categorical."""
    def __init__(self, probs=None, logits=None, validate_args=None, **kwargs):
        self.logits = logits if logits is not None else torch.log(probs.clamp(min=1e-8))

    def sample(self, sample_shape=torch.Size()):
        if self.logits.ndim < 2:
            u = torch.rand_like(self.logits, dtype=torch.float32)
            gumbel = -torch.log(-torch.log(u.clamp(min=1e-6, max=1.0 - 1e-6)))
            return torch.argmax(self.logits.float() + gumbel, dim=-1)

        seq_len = self.logits.shape[-2]
        out = torch.empty(self.logits.shape[:-1], dtype=torch.long, device=self.logits.device)
        
        chunk_size = 256
        for i in range(0, seq_len, chunk_size):
            chunk_logits = self.logits[..., i:i+chunk_size, :]
            u = torch.rand_like(chunk_logits, dtype=torch.float32)
            gumbel = -torch.log(-torch.log(u.clamp(min=1e-6, max=1.0 - 1e-6)))
            out[..., i:i+chunk_size] = torch.argmax(chunk_logits.float() + gumbel, dim=-1)
        
        return out


# ---------------------------------------------------------------------------
# loader node
# ---------------------------------------------------------------------------
class NLDLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gguf_name": (get_gguf_files(),),
                "vqvae_name": (folder_paths.get_filename_list("vae"),),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "weights_location": (["cpu_stream (low VRAM)", "gpu"], {"default": "cpu_stream (low VRAM)"}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("NLD_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Rebels_NLD"

    def load_model(self, gguf_name, vqvae_name, device, weights_location, attention="auto"):
        try:
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception:
            pass
        qdev = "cpu" if weights_location.startswith("cpu_stream") else device
        print("[NLD] device={} weights_location={} (qdata -> {})".format(device, weights_location, qdev))
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("[NLD] device=cuda selected but CUDA is not available")
        gguf_path = resolve_gguf(gguf_name)
        vqvae_path = folder_paths.get_full_path("vae", vqvae_name)

        print("[NLD] Loading config and tokenizer from {}".format(MODEL_ASSETS_DIR))
        config = AutoConfig.from_pretrained(MODEL_ASSETS_DIR, trust_remote_code=True)

        # Attention backend. NVIDIA's code requests flash_attn at each forward and silently
        # falls back if unavailable. "auto" uses flash when it imports cleanly, else sdpa.
        # Explicit choices let users force one (e.g. a broken flash-attn build -> pick sdpa).
        def _flash_ok():
            try:
                import flash_attn  # noqa: F401
                return True
            except Exception:
                return False

        attn_choice = attention
        if attn_choice == "auto":
            attn_choice = "flash_attention_2" if _flash_ok() else "sdpa"
        elif attn_choice == "flash_attention_2" and not _flash_ok():
            print("[NLD] flash_attention_2 requested but flash_attn not importable -> falling back to sdpa")
            attn_choice = "sdpa"
        try:
            config._attn_implementation = attn_choice
            config._attn_implementation_internal = attn_choice
        except Exception:
            pass
        # If the user explicitly wants NON-flash, neutralize their hardcoded
        # overwrite_attn_impl='flash_attn' by shadowing the flash import inside their module
        # so it degrades to the config choice instead of using flash anyway.
        self._force_no_flash = (attn_choice != "flash_attention_2")
        print("[NLD] attention backend: {}".format(attn_choice))
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ASSETS_DIR, trust_remote_code=True, fix_mistral_regex=True
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ASSETS_DIR, trust_remote_code=True)

        print("[NLD] Initializing empty weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model = model.to(torch.bfloat16)

        print("[NLD] Loading GGUF from {}".format(gguf_path))
        reader = gguf.GGUFReader(gguf_path)

        gg = {}
        for t in reader.tensors:
            name = str(t.name)
            arr = np.ascontiguousarray(t.data)
            qbytes = torch.from_numpy(arr.view(np.uint8).reshape(-1)).clone()
            oshape = [int(d) for d in reversed(t.shape)]
            key = "comfy.gguf.orig_shape.{}".format(name)
            if key in reader.fields:
                fld = reader.fields[key]
                try:
                    oshape = [int(fld.parts[i][0]) for i in fld.data]
                except Exception:
                    pass
            gg[name] = (qbytes, t.tensor_type, oshape)

        modules_dict = dict(model.named_modules())
        state_keys = set(model.state_dict().keys())

        def resolve_key(gname):
            if gname in state_keys:
                return gname
            for pref in ("model.", "language_model.", "model.language_model."):
                if pref + gname in state_keys:
                    return pref + gname
            hits = [k for k in state_keys if k.endswith("." + gname) or k.endswith(gname)]
            return hits[0] if len(hits) == 1 else None

        attached = 0
        unmatched = []
        consumed_biases = set()

        for gname, (qbytes, qtype, oshape) in gg.items():
            if gname in consumed_biases:
                continue
            target_key = resolve_key(gname)
            if target_key is None:
                unmatched.append(gname)
                continue

            parent_name, param_name = target_key.rsplit(".", 1)
            parent_module = modules_dict.get(parent_name)
            if parent_module is None:
                unmatched.append(gname)
                continue

            is_quant = qtype not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32)

            if is_quant and isinstance(parent_module, torch.nn.Embedding) and param_name == "weight":
                wrapper = GGUFEmbeddingWrapper(qbytes.to(qdev), qtype, oshape)
                gp_name, child = (parent_name.rsplit(".", 1) + [""])[:2] if "." in parent_name else ("", parent_name)
                gp = modules_dict.get(gp_name, model) if gp_name else model
                setattr(gp, child if gp_name else parent_name, wrapper)
                attached += 1

            elif is_quant and isinstance(parent_module, torch.nn.Linear) and param_name == "weight":
                bias = None
                bias_gname = gname.rsplit(".", 1)[0] + ".bias"
                if bias_gname in gg:
                    bb, bqt, bsh = gg[bias_gname]
                    bias = dequant_bytes(bb, bqt, bsh, torch.bfloat16).to(device)
                    consumed_biases.add(bias_gname)
                wrapper = GGUFLinearWrapper(qbytes.to(qdev), qtype, oshape, bias=bias)
                gp_name, child = (parent_name.rsplit(".", 1) + [""])[:2] if "." in parent_name else ("", parent_name)
                gp = modules_dict.get(gp_name, model) if gp_name else model
                setattr(gp, child if gp_name else parent_name, wrapper)
                attached += 1

            else:
                data = dequant_bytes(qbytes, qtype, oshape, torch.bfloat16).to(device)
                setattr(parent_module, param_name, torch.nn.Parameter(data, requires_grad=False))
                attached += 1

        n_gg = len(gg)
        del reader, gg
        import gc as _gc
        _gc.collect()
        print("[NLD] {}/{} gguf tensors attached".format(attached, n_gg))
        if unmatched:
            print("[NLD] Warning: unmatched tensors ({}): {}".format(len(unmatched), unmatched[:5]))

        print("[NLD] Loading VQVAE split from {}".format(vqvae_path))
        vqvae_sd = load_file(vqvae_path)
        vqvae_sd = {k: v.to(torch.bfloat16).to(device) for k, v in vqvae_sd.items()}
        missing, unexpected = model.load_state_dict(vqvae_sd, strict=False, assign=True)
        print("[NLD] vqvae loaded ({} tensors), unexpected={}".format(len(vqvae_sd), len(unexpected)))

        for name, module in model.named_modules():
            if "vision_tower" in name:
                continue
            for pn, p in list(module.named_parameters(recurse=False)):
                if p.device.type not in ("meta", device):
                    setattr(module, pn, torch.nn.Parameter(p.to(device), requires_grad=False))
            if isinstance(module, (GGUFLinearWrapper, GGUFEmbeddingWrapper)):
                continue  
            for bn, b in list(module.named_buffers(recurse=False)):
                if b is not None and b.device.type not in ("meta", device):
                    module._buffers[bn] = b.to(device)

        meta_params = [n for n, p in model.named_parameters()
                       if p.device.type == "meta" and "vision_tower" not in n]
        if meta_params:
            print("[NLD] Warning: still-meta params outside vision_tower: {}".format(meta_params[:5]))

        model.eval()
        return ({"model": model, "tokenizer": tokenizer, "device": device,
                 "force_no_flash": getattr(self, "_force_no_flash", False)},)


# ---------------------------------------------------------------------------
# generate node
# ---------------------------------------------------------------------------
class NLDTextToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nld_model": ("NLD_MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "decode_device": (["gpu", "cpu (slow, no OOM)"], {"default": "gpu"}),
                "dequant_cache_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 12.0, "step": 0.5}),
                "kv_cache": (["off (flat step time)", "on"], {"default": "off (flat step time)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Rebels_NLD"

    def generate(self, nld_model, prompt, width, height, steps, guidance, temperature, seed, decode_device="gpu", dequant_cache_gb=0.0, kv_cache="off (flat step time)"):
        model = nld_model["model"]
        tokenizer = nld_model["tokenizer"]

        torch.manual_seed(seed & 0x7FFFFFFF)

        if not hasattr(model, "text_to_image"):
            available = [name for name, _ in inspect.getmembers(model, predicate=inspect.ismethod)]
            raise RuntimeError("text_to_image not found. Available methods:\n{}".format(available))

        # Dynamically patch the model to avoid allocating 1GB tensors during math
        patch_text_to_image_for_vram(model)

        micro_cond = (
            "ORIGINAL WIDTH : {}; ORIGINAL HEIGHT : {}; TOP : 0; LEFT : 0; SCORE : 6.5"
            .format(width, height)
        )

        # Clean allocator state BEFORE the forward so every run starts from the same
        # baseline as the first (the ~5.5GB of weights stay resident intentionally via
        # ComfyUI's cached loader output; what we free here is the prior run's transients
        # and fragmented cache, which is what makes run 2+ OOM otherwise).
        import gc as _gc
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        use_kv_cache = kv_cache.startswith("on")
        if not use_kv_cache:
            print("[NLD] KV cache OFF (prevents growing seq -> rising s/it and terminal OOM)")

        _cache_mods = [m for m in model.modules() if isinstance(m, GGUFLinearWrapper)]
        for _m in _cache_mods:
            _m._w_cache = None
        _DequantCache.start(dequant_cache_gb)
        if dequant_cache_gb > 0:
            print("[NLD] dequant cache: up to {:.1f} GB (skips redundant dequants across steps)".format(
                dequant_cache_gb))

        print("[NLD] text_to_image: 1024x1024, {} steps, cfg {}, temp {}".format(
            steps, guidance, temperature))

        orig_global_cat = torch.distributions.Categorical
        torch.distributions.Categorical = GumbelCategorical
        
        model_module = sys.modules[model.__class__.__module__]
        orig_local_cat = getattr(model_module, 'Categorical', None)
        orig_dists_cat = getattr(getattr(model_module, 'dists', None), 'Categorical', None)

        if orig_local_cat is not None:
            model_module.Categorical = GumbelCategorical
        if orig_dists_cat is not None:
            model_module.dists.Categorical = GumbelCategorical

        force_no_flash = nld_model.get("force_no_flash", False)
        _flash_saved = None
        if force_no_flash:
            # Their attention layer checks attn_implementation / a flash flag; the most
            # robust neutralization is to make 'flash_attn' unimportable within their module
            # scope for the duration of the call so it falls to sdpa.
            for _fname in ("_flash_supports_window_size", "flash_attn_func", "flash_attn_varlen_func"):
                if hasattr(model_module, _fname):
                    _flash_saved = _flash_saved or {}
                    _flash_saved[_fname] = getattr(model_module, _fname)
                    setattr(model_module, _fname, None)

        orig_get_logits = getattr(model_module, "_t2i_get_logits", None)
        
        if orig_get_logits is not None:
            def sequential_cfg_get_logits(model_inner, all_input_embeddings, new_token_mask, **kwargs):
                b_size = all_input_embeddings.shape[0]
                if b_size > 1 and b_size % 2 == 0:
                    half = b_size // 2
                    
                    kwargs_un = {}
                    kwargs_cond = {}
                    for k, v in kwargs.items():
                        if isinstance(v, torch.Tensor) and getattr(v, "ndim", 0) > 0 and v.shape[0] == b_size:
                            kwargs_un[k] = v[:half]
                            kwargs_cond[k] = v[half:]
                        else:
                            kwargs_un[k] = v
                            kwargs_cond[k] = v

                    out_un = orig_get_logits(model_inner, all_input_embeddings[:half], new_token_mask[:half], **kwargs_un)
                    
                    # Pre-allocate to avoid a 1GB memory spike from torch.cat
                    out_combined = torch.empty((b_size, *out_un.shape[1:]), dtype=out_un.dtype, device=out_un.device)
                    out_combined[:half] = out_un
                    del out_un
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    out_combined[half:] = orig_get_logits(model_inner, all_input_embeddings[half:], new_token_mask[half:], **kwargs_cond)
                    return out_combined
                    
                return orig_get_logits(model_inner, all_input_embeddings, new_token_mask, **kwargs)

            model_module._t2i_get_logits = sequential_cfg_get_logits

        # Offload transformer to CPU during vqvae decode so the conv decoder gets the full
        # card (their decode_image_gen runs single-shot for one image and OOM/thrashes
        # otherwise). Restore after decode. Only meaningful when weights are GPU-resident;
        # in cpu_stream mode the qdata already lives on CPU so this is a cheap no-op there.
        _orig_decode = model.decode_image_gen
        _dev = nld_model.get("device", "cuda")

        _decode_on_cpu = decode_device.startswith("cpu")

        def _decode_with_offload(images_to_decode, height, width):
            try:
                mm.soft_empty_cache()
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            vq = model.encoder.vqvae
            if _decode_on_cpu:
                vq.to("cpu")
                images_to_decode = images_to_decode.to("cpu")
                try:
                    return _orig_decode(images_to_decode, height, width)
                finally:
                    _n = _DequantCache.stop(_cache_mods)
                    if _n:
                        print("[NLD] dequant cache freed ({} tensors)".format(_n))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            try:
                images_to_decode = images_to_decode.to(next(vq.parameters()).device)
            except Exception:
                pass
            return _orig_decode(images_to_decode, height, width)

        model.decode_image_gen = _decode_with_offload

        try:
            with torch.inference_mode():
                out = model.text_to_image(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    image_resolution=1024,
                    n_tokens=4096,
                    n_steps=steps,
                    guidance_scale=guidance,
                    temperature=temperature,
                    micro_cond=micro_cond,
                    use_cache=use_kv_cache,
                    cache_prompt=use_kv_cache,
                )
        finally:
            torch.distributions.Categorical = orig_global_cat
            if orig_local_cat is not None:
                model_module.Categorical = orig_local_cat
            if orig_dists_cat is not None:
                model_module.dists.Categorical = orig_dists_cat
            if orig_get_logits is not None:
                model_module._t2i_get_logits = orig_get_logits
            if _flash_saved:
                for _fname, _fval in _flash_saved.items():
                    setattr(model_module, _fname, _fval)
            model.decode_image_gen = _orig_decode

        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "images"):
            out = out.images[0]

        if hasattr(out, "convert"):  
            image_tensor = torch.from_numpy(np.array(out.convert("RGB"))).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
        elif isinstance(out, torch.Tensor):
            image_tensor = out.detach().float().cpu()
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            if image_tensor.shape[1] in (1, 3): 
                image_tensor = image_tensor.permute(0, 2, 3, 1)
            if image_tensor.min() < 0:
                image_tensor = (image_tensor + 1.0) / 2.0
            elif image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            image_tensor = image_tensor.clamp(0, 1)
        else:
            raise ValueError("Unknown output type: {}".format(type(out)))

        # release the raw model output + any dangling refs before returning so the next
        # run starts clean (image_tensor is already on CPU)
        del out
        import gc as _gc2
        _gc2.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (image_tensor,)


NODE_CLASS_MAPPINGS = {
    "NLDLoaderGGUF": NLDLoaderGGUF,
    "NLDTextToImage": NLDTextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NLDLoaderGGUF": "NVIDIA NLD GGUF Loader",
    "NLDTextToImage": "NVIDIA NLD Text to Image",
}