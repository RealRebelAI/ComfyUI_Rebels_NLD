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
import sys
import inspect

import numpy as np
import torch
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

    _reported = False

    def forward(self, x):
        if not GGUFLinearWrapper._reported:
            GGUFLinearWrapper._reported = True
            print("[NLD] FIRST LINEAR: x.device={} x.dtype={} qdata.device={}".format(
                x.device, x.dtype, self.qdata.device))
        q = self.qdata if self.qdata.device == x.device else self.qdata.to(x.device, non_blocking=True)
        w = dequant_bytes(q, self.qtype, self.oshape, x.dtype)
        b = self.bias.to(x.device, x.dtype) if self.bias is not None else None
        out = F.linear(x, w, b)
        del w
        if q is not self.qdata:
            del q
        return out


class GGUFEmbeddingWrapper(torch.nn.Module):
    """Row-gather embedding: dequantize ONLY the rows being looked up.

    Row gather is block-safe here: every embedding row is 4096 wide = a whole
    number of K-quant superblocks (4096/256 = 16), so each row's bytes are
    self-contained. Full-matrix dequant (131k x 4096) would spike ~1GB VRAM
    per lookup; the gather touches a few KB instead.
    """

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


# ---------------------------------------------------------------------------
# memory-lean categorical sampler (Gumbel-max, chunked fp32)
# ---------------------------------------------------------------------------
class GumbelCategorical:
    """Drop-in for torch.distributions.Categorical(logits=...).sample().

    NVIDIA's sampler feeds 4096x131072 logits to torch.multinomial, which
    materializes fp32 probability tensors (2.1GB+) - the step-3 OOM on 8GB
    cards. argmax(logits + gumbel) draws from the identical distribution.
    Chunked over the sequence dim so the fp32 upcast stays ~135MB at a time.
    """

    def __init__(self, probs=None, logits=None, validate_args=None, **kwargs):
        if logits is None:
            logits = torch.log(probs.clamp_min(1e-20))
        self.logits = logits

    def sample(self, sample_shape=None):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        lg = self.logits
        if lg.ndim < 2:
            u = torch.rand_like(lg, dtype=torch.float32)
            g = -torch.log(-torch.log(u.clamp(1e-6, 1.0 - 1e-6)))
            return torch.argmax(lg.float() + g, dim=-1)
        seq_len = lg.shape[-2]
        out = torch.empty(lg.shape[:-1], dtype=torch.long, device=lg.device)
        chunk = 128
        for i in range(0, seq_len, chunk):
            cl = lg[..., i:i + chunk, :].float()
            u = torch.rand_like(cl)
            # Gumbel noise in place: u -> gumbel
            u.clamp_(1e-6, 1.0 - 1e-6).log_().neg_().log_().neg_()
            cl.add_(u)
            out[..., i:i + chunk] = torch.argmax(cl, dim=-1)
            del cl, u
        return out



def _install_lean_t2i(model):
    """Replace NVIDIA's text_to_image with a source-identical copy whose CFG-combine and
    softmax run chunked/in-place, so the 131k-vocab tail fits in 8GB. Falls back silently
    if their source shape ever changes."""
    import inspect, textwrap, types as _t
    try:
        fn = type(model).text_to_image
        src = inspect.getsource(fn)
        src = textwrap.dedent(src)
    except Exception as e:
        print("[NLD] lean t2i patch skipped (no source): {}".format(e))
        return False

    cfg_old = "logits = (1.0 + guidance_scale) * logits - guidance_scale * logits_un"
    cfg_new = ("logits.mul_(1.0 + guidance_scale); "
               "logits_un.mul_(guidance_scale); "
               "logits.sub_(logits_un); del logits_un")
    soft_old = "probs = logits.softmax(dim=-1)"
    soft_new = "probs = None"
    gather_guard = "torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)"

    if cfg_old not in src or soft_old not in src:
        print("[NLD] lean t2i patch skipped (source shape changed)")
        return False

    src = src.replace(cfg_old, cfg_new)
    src = src.replace(soft_old, soft_new)
    # x0_p only feeds confidence; when probs is None compute it lean from a fresh softmax
    # over just the selected logits is complex, so guard: if probs is None, use ones.
    src = src.replace(
        "x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)",
        "x0_p = (torch.gather(logits, -1, x0.long()[..., None]).squeeze(-1).float().softmax(-1) "
        "if probs is None else torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1))"
    )
    # dedent already applied; rename to avoid clobbering, then bind
    src = src.replace("def text_to_image(", "def _lean_text_to_image(", 1)
    g = fn.__globals__  # exec into the LIVE module dict so np, dists, F, etc. resolve,
                        # and so the sampler swap we do on the module is visible here
    try:
        exec(src, g)
        bound = _t.MethodType(g["_lean_text_to_image"], model)
        model.text_to_image = bound
        print("[NLD] lean text_to_image installed (chunked CFG + no full softmax)")
        return True
    except Exception as e:
        print("[NLD] lean t2i patch failed, using stock: {}".format(e))
        return False


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
            }
        }

    RETURN_TYPES = ("NLD_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Rebels_NLD"

    def load_model(self, gguf_name, vqvae_name, device, weights_location):
        # free anything ComfyUI itself has resident before we claim VRAM
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

        # index all tensors first: name -> (qbytes cpu, qtype, orig shape)
        gg = {}
        for t in reader.tensors:
            name = str(t.name)
            arr = np.ascontiguousarray(t.data)
            qbytes = torch.from_numpy(arr.view(np.uint8).reshape(-1)).clone()
            oshape = [int(d) for d in reversed(t.shape)]  # ggml stores dims reversed
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
                # bias comes from the GGUF (the module's own bias is a meta tensor)
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
                # F16/F32 (or quant on a non-Linear/Embedding): materialize as bf16 param
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

        # anything the constructor loaded from disk (emu3_vqvae) or created as real tensors:
        # move to device. vision_tower stays wherever it is (unused for t2i).
        for name, module in model.named_modules():
            if "vision_tower" in name:
                continue
            for pn, p in list(module.named_parameters(recurse=False)):
                if p.device.type not in ("meta", device):
                    setattr(module, pn, torch.nn.Parameter(p.to(device), requires_grad=False))
            if isinstance(module, (GGUFLinearWrapper, GGUFEmbeddingWrapper)):
                continue  # qdata placement is managed by weights_location - never sweep it
            for bn, b in list(module.named_buffers(recurse=False)):
                if b is not None and b.device.type not in ("meta", device):
                    module._buffers[bn] = b.to(device)

        meta_params = [n for n, p in model.named_parameters()
                       if p.device.type == "meta" and "vision_tower" not in n]
        if meta_params:
            print("[NLD] Warning: still-meta params outside vision_tower: {}".format(meta_params[:5]))

        model.eval()
        return ({"model": model, "tokenizer": tokenizer, "device": device},)


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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Rebels_NLD"

    def generate(self, nld_model, prompt, width, height, steps, guidance, temperature, seed):
        model = nld_model["model"]
        tokenizer = nld_model["tokenizer"]

        torch.manual_seed(seed & 0x7FFFFFFF)

        if not hasattr(model, "text_to_image"):
            available = [name for name, _ in inspect.getmembers(model, predicate=inspect.ismethod)]
            raise RuntimeError("text_to_image not found. Available methods:\n{}".format(available))

        # NVIDIA's released code is only self-consistent at image_resolution=1024 with
        # n_tokens=4096: gen_shape (64,64) = 4096 raw gen tokens, downsample_gen (2x2)
        # compresses to 1024 embeddings, and n_tokens_txt is hardcoded to 1024 reserve
        # tokens only for resolution 1024. Any other combo mismatches the placeholder
        # mask by 4x inside _t2i_wte. Output is always 1024x1024; width/height feed the
        # micro-conditioning string (framing/crop metadata) only.
        micro_cond = (
            "ORIGINAL WIDTH : {}; ORIGINAL HEIGHT : {}; TOP : 0; LEFT : 0; SCORE : 6.5"
            .format(width, height)
        )

        try:
            mm.soft_empty_cache()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Inject the lean sampler into NVIDIA's module namespace only (their code
        # resolves dists.Categorical / Categorical at call time). Restored in the
        # finally below so nothing stays patched between runs.
        import types as _types
        _nvmod = sys.modules.get(type(model).__module__)
        _orig_dists = getattr(_nvmod, "dists", None) if _nvmod else None
        _orig_cat = getattr(_nvmod, "Categorical", None) if _nvmod else None
        if _nvmod is not None:
            if _orig_dists is not None:
                _nvmod.dists = _types.SimpleNamespace(Categorical=GumbelCategorical)
            if _orig_cat is not None:
                _nvmod.Categorical = GumbelCategorical
            print("[NLD] lean Gumbel-max sampler injected")
        _install_lean_t2i(model)

        p = next(model.parameters())
        print("[NLD] first model param: device={} dtype={}".format(p.device, p.dtype))
        if hasattr(model, "get_model"):
            try:
                print("[NLD] get_model().device = {}".format(model.get_model().device))
            except Exception as e:
                print("[NLD] get_model().device failed: {}".format(e))
        print("[NLD] text_to_image: 1024x1024, {} steps, cfg {}, temp {}".format(
            steps, guidance, temperature))

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
                )
        finally:
            if _nvmod is not None:
                if _orig_dists is not None:
                    _nvmod.dists = _orig_dists
                if _orig_cat is not None:
                    _nvmod.Categorical = _orig_cat

        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "images"):
            out = out.images[0]

        if hasattr(out, "convert"):  # PIL
            image_tensor = torch.from_numpy(np.array(out.convert("RGB"))).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
        elif isinstance(out, torch.Tensor):
            image_tensor = out.detach().float().cpu()
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            if image_tensor.shape[1] in (1, 3):  # BCHW -> BHWC
                image_tensor = image_tensor.permute(0, 2, 3, 1)
            if image_tensor.min() < 0:
                image_tensor = (image_tensor + 1.0) / 2.0
            elif image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            image_tensor = image_tensor.clamp(0, 1)
        else:
            raise ValueError("Unknown output type: {}".format(type(out)))

        return (image_tensor,)


NODE_CLASS_MAPPINGS = {
    "NLDLoaderGGUF": NLDLoaderGGUF,
    "NLDTextToImage": NLDTextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NLDLoaderGGUF": "NVIDIA NLD GGUF Loader",
    "NLDTextToImage": "NVIDIA NLD Text to Image",
}
