# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import copy
import math
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.resnet import Downsample2D, Upsample2D
from einops import rearrange
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.generation.utils import GenerateOutput

from .configuration_nemotron_labs_diffusion_image import NemotronLabsDiffusionImageConfig
from .modeling_ministral import Ministral3Model
from .modeling_ministral_dlm import MinistralDiffEncoderModel
# The imports below are not used directly but MUST stay here so that HF's
# dynamic-module cache scanner (regex: r"from\.X import") copies every
# transitive dependency into the hash directory.
from .chat_utils import generate_with_prefix_cache_block_diff as _gcbd  # noqa: F401
from .nemotron_diffusion_image_utils import maybe_truncate_last_dim as _mtld  # noqa: F401
from .configuration_ministral_dlm import MinistralDLMConfig as _MinistralDLMConfig  # noqa: F401


def _resolve_local_path(path_value: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_vqvae_from_local(vqvae_path: Path):
    """Load Emu3p5VisionVQModel directly from local files.

    Bypasses AutoModel.from_pretrained because newer huggingface_hub versions
    validate the path argument as a HF repo ID, rejecting absolute local paths.
    """
    import importlib.util
    import json
    import sys
    import types

    from safetensors.torch import load_file

    pkg = f"_emu3_vqvae_{vqvae_path.name}"

    # Create a package namespace so relative imports inside the vqvae files work
    pkg_mod = types.ModuleType(pkg)
    pkg_mod.__path__ = [str(vqvae_path)]
    pkg_mod.__package__ = pkg
    sys.modules[pkg] = pkg_mod

    def _load_mod(mod_name, filename):
        spec = importlib.util.spec_from_file_location(
            f"{pkg}.{mod_name}",
            vqvae_path / filename,
            submodule_search_locations=[str(vqvae_path)],
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg
        sys.modules[f"{pkg}.{mod_name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    cfg_mod = _load_mod("configuration_emu3p5visionvq", "configuration_emu3p5visionvq.py")
    mdl_mod = _load_mod("modeling_emu3p5visionvq", "modeling_emu3p5visionvq.py")

    with open(vqvae_path / "config.json") as f:
        cfg_data = json.load(f)

    # PretrainedConfig accepts and stores arbitrary kwargs, so pass everything
    vqvae_config = cfg_mod.Emu3p5VisionVQConfig(**cfg_data)
    model = mdl_mod.Emu3p5VisionVQModel(vqvae_config)

    sf_path = vqvae_path / "model.safetensors"
    state_dict = load_file(str(sf_path))
    model.load_state_dict(state_dict)

    return model


def _preprocess_emu3_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.asarray(image, dtype=np.float32)
    image = image / 127.5 - 1.0
    return torch.from_numpy(image).permute(2, 0, 1).float()


class Emu3ImageProcessor:
    def preprocess(self, image):
        return _preprocess_emu3_image(image).unsqueeze(0)


# ---------------------------------------------------------------------------
# T2I helpers (inlined — no llava imports required)
# ---------------------------------------------------------------------------

class _NC:
    """Token constants for the Ministral diffusion model."""
    reserve_id           = 18
    reserve_id_token     = '<SPECIAL_18>'
    reserve_id_enc       = 19
    reserve_id_token_enc = '<SPECIAL_19>'
    mask_id              = 100
    eos_id               = 11
    gen_im_start_token   = '<SPECIAL_21>'
    gen_im_end_token     = '<SPECIAL_22>'


def _pad_along_last_dim(tensor: torch.Tensor, size: int) -> torch.Tensor:
    pad_size = size - tensor.shape[-1]
    if pad_size <= 0:
        return tensor
    padding = torch.zeros(*tensor.shape[:-1], pad_size,
                          dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=-1)


def _maybe_truncate_last_dim(tensor: torch.Tensor, size: int) -> torch.Tensor:
    if size >= tensor.shape[-1]:
        return tensor
    return tensor[..., :size]


_INT_MAX = 1_000_000


def _t2i_wte(model, x, gen_shape=None, x_gen=None,
             inputs_embeds_curr=None, new_token_mask=None):
    """Embed text tokens and splice in gen-token embeddings."""
    assert x_gen is not None
    if new_token_mask is None or not torch.any(new_token_mask):
        if inputs_embeds_curr is None:
            return model.embed_tokens(x), new_token_mask
        return inputs_embeds_curr, new_token_mask
    gen_latents_comp_embeds = model.call_gen_embedding(x_gen, gen_shape)
    if inputs_embeds_curr is None:
        x_txt_only = x.clone()
        x_txt_only[new_token_mask] = 0
        inputs_embeds_curr = model.embed_tokens(x_txt_only)
    inputs_embeds_curr[new_token_mask] = (
        _pad_along_last_dim(gen_latents_comp_embeds, inputs_embeds_curr.shape[-1])
        .view(-1, inputs_embeds_curr.shape[-1])
    )
    return inputs_embeds_curr, new_token_mask


def _t2i_get_logits(model, input_embeddings, modality_indices,
                    past_key_values=None, gen_shape=None, timesteps=None,
                    input_modality_indices=None):
    """Forward pass returning generation logits only."""
    if input_modality_indices is None:
        input_modality_indices = modality_indices
    output = model(
        None,
        input_embeddings=input_embeddings,
        modality_indices=input_modality_indices,
        past_key_values=past_key_values,
        is_training=False,
        overwrite_attn_impl='flash_attn',
    )
    hidden_states = output.last_hidden_state
    gen_hidden_states = hidden_states[modality_indices]
    gen_hidden_states = _maybe_truncate_last_dim(gen_hidden_states, model.config.d_model_gen)
    gen_logits = model.call_gen_predictor(gen_hidden_states, gen_shape, timesteps=timesteps)
    seq_len_per_img = int(np.prod(gen_shape))
    if len(gen_logits.shape) == 2:
        gen_logits = gen_logits.view(-1, seq_len_per_img, gen_logits.shape[-1])
    else:
        gen_logits = gen_logits.view(-1, seq_len_per_img, *gen_logits.shape[-2:])
    return gen_logits


def _cosine_schedule_2(x):
    x = 1.0 - np.clip(x, 0.0, 1.0)
    return np.cos(np.pi * x / 2.0)


def _exp_schedule(x):
    z = (1.0 - np.exp(-5.0 * x)) / (1.0 - np.exp(-5.0))
    return np.clip(z, 0.0001, 1.0)


def _logit_normal_schedule(shift, sigmas):
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int,
                              schedule: str = 'shift',
                              shift: int = 3) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    steps = int(min(steps, mask_num[0]))
    t = torch.linspace(0, 1, steps + 1)
    sigmas = _logit_normal_schedule(shift, t)
    sigmas = sigmas.to(mask_num.device)
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps,
                                      device=mask_index.device, dtype=torch.int64)
    for i in range(mask_num.size(0)):
        sigmas_sample = (sigmas * mask_num[i]).to(torch.int64)
        sigmas_sample = sigmas_sample[1:] - sigmas_sample[:-1]
        sigmas_sample = torch.clamp(sigmas_sample, 1, None)
        delta = sigmas_sample.sum() - mask_num[i]
        assert delta >= 0
        j = 0
        while delta > 0:
            j = j % len(sigmas_sample)
            if sigmas_sample[j] == 1:
                j += 1
                continue
            delta -= 1
            sigmas_sample[j] -= 1
            j += 1
        assert sigmas_sample.sum() == mask_num[i]
        num_transfer_tokens[i] = sigmas_sample
    return num_transfer_tokens.flip(-1)


class _MinistralConv:
    """Minimal CHATML conversation template for the Ministral model."""
    _SYSTEM = (
        "<|im_start|>system\n"
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    )
    _SEP = "<|im_end|>"
    _ROLES = ("<|im_start|>user", "<|im_start|>assistant")

    def __init__(self):
        self.messages: List[Tuple[str, Optional[str]]] = []

    def append_message(self, role: str, message: Optional[str]) -> None:
        self.messages.append((role, message))

    def get_prompt(self) -> str:
        ret = self._SYSTEM + self._SEP + "\n"
        for role, message in self.messages:
            if message is not None:
                ret += role + "\n" + message + self._SEP + "\n"
            else:
                ret += role + "\n"
        return ret

    @property
    def roles(self):
        return self._ROLES


_IMAGE_TOKEN_INDEX = -200


def _tokenizer_image_token(prompt: str, tokenizer,
                            return_tensors: str = "pt") -> torch.Tensor:
    """Tokenise a prompt that may contain <image> placeholder tokens."""
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def _insert_sep(X, sep):
        return [e for pair in zip(X, [sep] * len(X)) for e in pair][:-1]

    input_ids: List[int] = []
    offset = 0
    if (prompt_chunks and prompt_chunks[0]
            and prompt_chunks[0][0] == tokenizer.bos_token_id):
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for x in _insert_sep(prompt_chunks, [_IMAGE_TOKEN_INDEX] * (offset + 1)):
        input_ids.extend(x[offset:])
    ids = torch.tensor(input_ids, dtype=torch.long)
    if return_tensors == "pt":
        return ids
    return ids.tolist()


def _stratified_random(n: int = 64, seed: Optional[int] = None,
                        shuffle_blocks: bool = True) -> List[int]:
    """Progressive Multi-Jittered ordering over an n×n integer grid."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive power of two")
    rng = random.Random(seed)
    occupied = [[False] * n for _ in range(n)]
    seq: List[int] = []
    blocks: List[Tuple[int, int, int]] = [(0, 0, n)]

    def _has(x0, y0, size):
        for yy in range(y0, y0 + size):
            for xx in range(x0, x0 + size):
                if occupied[yy][xx]:
                    return True
        return False

    def _place(x0, y0, size):
        x, y, attempts = rng.randrange(x0, x0 + size), rng.randrange(y0, y0 + size), 0
        while occupied[y][x]:
            x, y = rng.randrange(x0, x0 + size), rng.randrange(y0, y0 + size)
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("placement failed")
        occupied[y][x] = True
        seq.append(y * n + x)

    size = n
    while size > 1:
        half = size // 2
        children = [(x0 + dx, y0 + dy, half)
                    for (x0, y0, _) in blocks
                    for dx, dy in [(0, 0), (half, 0), (0, half), (half, half)]]
        if shuffle_blocks:
            rng.shuffle(children)
        for (x0, y0, s) in children:
            if not _has(x0, y0, s):
                _place(x0, y0, s)
        blocks = children
        size = half

    remaining = [y * n + x for y in range(n) for x in range(n) if not occupied[y][x]]
    rng.shuffle(remaining)
    seq.extend(remaining)
    return seq


def _gumbel_noise(t: torch.Tensor) -> torch.Tensor:
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -torch.log(-torch.log(noise))


class SimpleUVitBlock(nn.Module):
    def __init__(self, channels, downsample: bool, upsample: bool):
        super().__init__()
        self.downsample = None
        self.upsample = None
        if downsample:
            self.downsample = Downsample2D(
                channels,
                use_conv=True,
                padding=0,
                name="Conv2d_0",
                kernel_size=2,
                norm_type="rms_norm",
                eps=1e-6,
                elementwise_affine=True,
                bias=False,
                out_channels=channels,
            )
        if upsample:
            self.upsample = Upsample2D(
                channels,
                use_conv_transpose=True,
                kernel_size=2,
                padding=0,
                name="conv",
                norm_type="rms_norm",
                eps=1e-6,
                elementwise_affine=True,
                bias=False,
                interpolate=False,
                out_channels=channels,
            )

    def forward(self, hidden_states, size):
        hidden_states = rearrange(hidden_states, "b (h w) d -> b d h w", h=size[0], w=size[1])
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)
        return rearrange(hidden_states, "b d h w -> b (h w) d")


class NemotronLabsDiffusionImageModel(Ministral3Model):
    config_class = NemotronLabsDiffusionImageConfig

    def __init__(self, config):
        super().__init__(config)
        self.build_vqvae(config)
        self.build_gen_embedding(config)
        self.image_newline = nn.Parameter(torch.empty(config.hidden_size))

    def build_vqvae(self, config):
        mm_vqvae = getattr(config, "mm_vqvae", "emu3_vqvae")
        # Prefer model_dir/_name_or_path so this works both from the release dir
        # and when loaded via trust_remote_code (where __file__ is the HF cache).
        model_dir = Path(getattr(config, "_name_or_path", ""))
        if model_dir.is_dir():
            vqvae_path = (model_dir / mm_vqvae).resolve()
        else:
            vqvae_path = _resolve_local_path(mm_vqvae)
        # When loading from HF hub, the vqvae subdirectory is not copied to
        # the dynamic-module cache hash dir.  Fall back to snapshot_download.
        if not vqvae_path.is_dir():
            repo_id = getattr(config, "_name_or_path", "")
            if repo_id and not Path(repo_id).is_dir():
                from huggingface_hub import snapshot_download
                local_dir = snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[f"{mm_vqvae}/*"],
                )
                vqvae_path = Path(local_dir) / mm_vqvae
        self.vqvae = _load_vqvae_from_local(vqvae_path)
        self.vqvae.eval()
        self.vqvae.requires_grad_(False)
        self.image_processor_gen = Emu3ImageProcessor()

    def build_gen_embedding(self, config):
        self.downsample_gen = SimpleUVitBlock(config.d_model_gen, downsample=True, upsample=False) if config.downsample else None
        self.upsample_gen = SimpleUVitBlock(config.d_model_gen, downsample=False, upsample=True) if config.downsample else None
        self.gen_embedding = nn.Embedding(self.vqvae.config.codebook_size + 256, config.d_model_gen)
        self.gen_predictor = nn.Linear(config.d_model_gen, self.vqvae.config.codebook_size, bias=config.include_bias)
        self.gen_embedding_2 = None
        self.gen_predictor_2 = None

    def call_gen_embedding(self, token_ids, gen_shape=None, enc=False):
        del enc
        hidden_states = self.gen_embedding(token_ids)
        if self.downsample_gen is not None:
            hidden_states = self.downsample_gen(hidden_states, gen_shape)
        return hidden_states

    def call_gen_predictor(self, gen_hidden_states, gen_shape=None, timesteps=None, labels=None):
        del timesteps, labels
        if self.upsample_gen is not None:
            seq_len_per_image = (gen_shape[0] // 2) * (gen_shape[1] // 2)
            gen_hidden_states = self.upsample_gen(
                gen_hidden_states.view(-1, seq_len_per_image, gen_hidden_states.shape[-1]),
                (gen_shape[0] // 2, gen_shape[1] // 2),
            )
            gen_hidden_states = gen_hidden_states.flatten(0, 1)
        return self.gen_predictor(gen_hidden_states)

    def encode_image_gen(self, images, enc=False):
        batch_size = images.shape[0]
        # Emu3p5VisionVQModel.encode does not accept mini_batch_size;
        # implement manual chunking for large images.
        if images.shape[2] > 256 and batch_size > 2:
            mini_bs = 2
            qs, idxs = [], []
            for i in range(0, batch_size, mini_bs):
                q, _, (_, _, idx) = self.vqvae.encode(images[i:i + mini_bs])
                qs.append(q)
                idxs.append(idx)
            quantized = torch.cat(qs, dim=0)
            indices = torch.cat(idxs, dim=0)
        else:
            quantized, _, (_, _, indices) = self.vqvae.encode(images)
        latent_height, latent_width = quantized.shape[-2], quantized.shape[-1]
        return indices.reshape(batch_size, -1), (latent_height, latent_width)

    @torch.no_grad()
    def decode_image_gen(self, images_to_decode, height, width):
        vae_scale_factor = 16
        indices = self.vqvae.quantize.get_codebook_entry(images_to_decode)
        indices = rearrange(
            indices,
            "b (h w) d -> b d h w",
            h=height // vae_scale_factor,
            w=width // vae_scale_factor,
        )
        # Emu3p5VisionVQModel.decode does not accept mini_batch_size;
        # implement manual chunking for large images.
        if height > 256 and len(indices) > 2:
            mini_bs = 2
            chunks = [self.vqvae.decode(indices[i:i + mini_bs])
                      for i in range(0, len(indices), mini_bs)]
            images = torch.cat(chunks, dim=0).float()
        else:
            images = self.vqvae.decode(indices).float()
        images = images.clamp(-1, 1)
        images = (images + 1) / 2
        images = (images * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return images


class NemotronLabsDiffusionImageForMaskedDiffusion(MinistralDiffEncoderModel):
    config_class = NemotronLabsDiffusionImageConfig
    supports_gradient_checkpointing = True
    base_model_prefix = ""

    def __init__(self, config: NemotronLabsDiffusionImageConfig, **kwargs):
        del kwargs
        config.d_model = config.hidden_size
        config.include_bias = config.mlp_bias
        if not hasattr(config, "d_model_gen") or config.d_model_gen < 0:
            config.d_model_gen = config.d_model
        if not hasattr(config, "mlp_hidden_size_gen") or config.mlp_hidden_size_gen < 0:
            config.mlp_hidden_size_gen = config.intermediate_size
        if not hasattr(config, "downsample"):
            config.downsample = False
        super().__init__(config)
        self.encoder = NemotronLabsDiffusionImageModel(self.config)
        self.post_init()

    @property
    def model(self):
        return self.encoder

    def get_model(self):
        return self.encoder

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = None,
        return_nfe: bool = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        del image_sizes, modalities
        if images is not None:
            raise NotImplementedError("This public release only supports text-to-image generation without multimodal image inputs.")
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("inputs_embeds is not supported")
        if self.config.dlm_paradigm == "bidirectional":
            kwargs.setdefault("causal_context", False)
        inputs_embeds = self.get_model().embed_tokens(inputs)
        output, nfe = MinistralDiffEncoderModel.generate_diffusion(
            self,
            prompt_ids=None,
            prompt_embeds=inputs_embeds,
            **kwargs,
        )
        if return_nfe:
            return output, nfe
        return output

    def encode_image_gen(self, images, enc=False):
        return self.encoder.encode_image_gen(images, enc=enc)

    def decode_image_gen(self, images_to_decode, height, width):
        return self.encoder.decode_image_gen(images_to_decode, height, width)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        return super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    @torch.no_grad()
    def text_to_image(
        self,
        prompt: str,
        tokenizer,
        sample_policy: str = 'multinomial',
        confidence_policy: str = 'mmada',
        guidance_scale: float = 5.0,
        n_steps: int = 20,
        batch_size: int = 1,
        image_resolution: int = 512,
        n_tokens: int = 1024,
        shift: int = 3,
        alg_temp: float = 1.0,
        min_temperature: float = 0.01,
        dynamic_temperature: bool = False,
        micro_cond: str = 'ORIGINAL WIDTH : 1024; ORIGINAL HEIGHT : 1024; TOP : 0; LEFT : 0; SCORE : 6.5',
        temperature: float = 1.0,
        schedule_temp: str = 'linear',
        shift_alg=None,
        top_p=None,
        top_k=None,
        unmask_order=None,
        cfg_interval=(0, 1),
        order_cutoff: float = 100,
        template: str = 'Generate an image with the caption:\n <prompt>',
        use_cache=None,
        cache_prompt=None,
        causal_context: bool = True,
        is_legacy: bool = False,
        edit_threshold: float = -1,
        disable_tqdm: bool = False,
        return_intermediate_steps: bool = False,
        **kwargs,
    ):
        """Generate an image from a text prompt using masked diffusion."""
        if shift_alg is None:
            shift_alg = shift

        NC = _NC
        device = self.get_model().device

        reserve_token  = NC.reserve_id_token
        reserve_id     = NC.reserve_id
        img_mask_id    = 131073   # Emu3 VQ mask token
        txt_mask_id    = NC.mask_id
        eot_id         = NC.eos_id
        img_begin      = NC.gen_im_start_token
        img_end        = NC.gen_im_end_token

        if use_cache is None:
            use_cache = True
        if cache_prompt is None:
            cache_prompt = True
        if self.config.dlm_paradigm == 'bidirectional':
            causal_context = False
            cache_prompt = False
            use_cache = False

        if is_legacy:
            img_begin = img_end = ''

        model_module = self.module if hasattr(self, "module") else self
        for layer in model_module.encoder.layers:
            layer.self_attn.mode = 'bidirectional'
        for layer in model_module.encoder.layers:
            if hasattr(layer.self_attn, 'diffusion_lm'):
                layer.self_attn.diffusion_lm = True

        gen_shape_map = {1024: (64, 64), 512: (32, 32), 256: (16, 16)}
        gen_shape = gen_shape_map[image_resolution]
        n_tokens_txt = 1024 if image_resolution == 1024 else n_tokens

        prompt_full = f"{prompt} {micro_cond}"
        question = template.replace('<prompt>', prompt_full)

        conv = _MinistralConv()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1],
                            f"Sure {img_begin}{reserve_token * n_tokens_txt}{img_end}")
        prompt_question = conv.get_prompt()
        print(prompt_question.replace(reserve_token, '*'))

        input_ids = _tokenizer_image_token(
            prompt_question, tokenizer, return_tensors="pt"
        ).unsqueeze(0).to(device)

        is_gen = input_ids == reserve_id
        is_gen_enc = input_ids == NC.reserve_id_enc
        is_eot = torch.where(input_ids == eot_id)[1]
        assert len(is_eot) == 3, f"Expected 3 EOT tokens, got {len(is_eot)}"
        prompt_cutoff = is_eot[1]
        is_prompt = torch.zeros_like(input_ids, dtype=torch.bool)
        is_prompt[:, :prompt_cutoff + 1] = True
        raw_input_ids = input_ids

        # Standard text embedding (no gen tokens yet)
        inputs_embeds = self.get_model().embed_tokens(raw_input_ids)

        inputs_embeds_uncond = inputs_embeds.clone()
        noise_embed = self.get_model().embed_tokens(
            torch.tensor([txt_mask_id], device=device)
        )
        inputs_embeds_uncond[is_prompt] = noise_embed

        xt = torch.full((batch_size, n_tokens), img_mask_id,
                        dtype=torch.long, device=device)

        mask_idx = xt == img_mask_id
        num_transfer_tokens = _get_num_transfer_tokens(
            mask_idx, n_steps, schedule='shift', shift=shift
        )
        print(num_transfer_tokens)

        sch_t = np.linspace(0, 1, n_steps)
        if schedule_temp == 'linear':
            sch_temperatures = (1.0 - sch_t) * (1.0 - min_temperature) + min_temperature
        elif schedule_temp == 'cosine2':
            sch_temperatures = _cosine_schedule_2(1.0 - sch_t) * (1.0 - min_temperature) + min_temperature
        elif schedule_temp == 'shift':
            sch_temperatures = _logit_normal_schedule(shift_alg, 1.0 - sch_t) * (1.0 - min_temperature) + min_temperature
        elif schedule_temp == 'exp':
            sch_temperatures = _exp_schedule(1.0 - sch_t) * (1.0 - min_temperature) + min_temperature
        else:
            raise NotImplementedError(f"Unknown schedule_temp: {schedule_temp}")
        sch_temperatures = torch.tensor(sch_temperatures, device=device, dtype=torch.float32)

        cfg_start = int(cfg_interval[0] * n_steps)
        cfg_end   = int(cfg_interval[1] * n_steps)

        if confidence_policy == 'stratified' and unmask_order is None:
            _dim = int(math.sqrt(n_tokens))
            unmask_order = _stratified_random(n=_dim, seed=42, shuffle_blocks=True)

        total_edited = 0
        intermediate_x0s = []
        temp_idx = 0
        past_key_values = None
        cache_len = 0

        for decode_step_idx, num_transfer in tqdm(
            enumerate(num_transfer_tokens[0]),
            total=num_transfer_tokens.shape[1],
            disable=disable_tqdm,
        ):
            local_temp = sch_temperatures[temp_idx]
            temp_idx += 1
            if temp_idx / n_steps > order_cutoff:
                confidence_policy = 'mmada'

            mask_idx = xt == img_mask_id
            n_mask = mask_idx.sum()
            timesteps = (n_mask / mask_idx.numel()).view(1)

            do_cfg = guidance_scale > 0 and cfg_start <= temp_idx <= cfg_end
            if do_cfg:
                input_embeddings_input = torch.cat([inputs_embeds_uncond, inputs_embeds]).clone()
                xt_input = torch.cat([xt, xt])
                new_token_mask = is_gen.repeat(2, 1)
                is_gen_enc_mask = is_gen_enc.repeat(2, 1)
                is_gen_enc_mask[0, :] = False
                timesteps_in = timesteps.repeat(2)
            else:
                input_embeddings_input = inputs_embeds.clone()
                new_token_mask = is_gen
                xt_input = xt
                is_gen_enc_mask = is_gen_enc
                timesteps_in = timesteps

            all_input_embeddings, new_token_mask = _t2i_wte(
                self.get_model(), None, gen_shape=gen_shape,
                x_gen=xt_input,
                inputs_embeds_curr=input_embeddings_input,
                new_token_mask=new_token_mask,
            )

            if use_cache and cache_prompt:
                if decode_step_idx == 0:
                    if causal_context:
                        for layer in model_module.encoder.layers:
                            if hasattr(layer.self_attn, 'diffusion_lm'):
                                layer.self_attn.diffusion_lm = False
                    output = self.get_model()(
                        None,
                        input_embeddings=all_input_embeddings[:, :prompt_cutoff],
                        modality_indices=new_token_mask[:, :prompt_cutoff],
                        output_hidden_states=True,
                        past_key_values=None,
                        is_training=False,
                        use_cache=True,
                        overwrite_attn_impl='flash_attn',
                    )
                    past_key_values = output.past_key_values
                    cache_len = past_key_values.get_seq_length()
                    if causal_context:
                        for layer in model_module.encoder.layers:
                            if hasattr(layer.self_attn, 'diffusion_lm'):
                                layer.self_attn.diffusion_lm = True
            else:
                past_key_values = None
                cache_len = 0

            logits = _t2i_get_logits(
                self.get_model(),
                all_input_embeddings[:, cache_len:],
                new_token_mask[:, cache_len:],
                past_key_values=past_key_values,
                gen_shape=gen_shape,
                input_modality_indices=new_token_mask[:, cache_len:],
                timesteps=timesteps_in,
            )

            if do_cfg:
                new_token_mask, _ = new_token_mask.chunk(2)
                logits_un, logits = logits.chunk(2)
                logits_is_ninf = logits == -np.inf
                logits = (1.0 + guidance_scale) * logits - guidance_scale * logits_un
                logits[logits_is_ninf] = -np.inf

            if top_p is not None or top_k is not None:
                _b, _l, _v = logits.shape
                logits_flat = logits.view(_b * _l, _v)
                if top_k and top_k > 0:
                    topk = min(top_k, logits_flat.size(-1))
                    idx_rm = logits_flat < torch.topk(logits_flat, topk)[0][..., -1, None]
                    logits_flat[idx_rm] = -np.inf
                if top_p and top_p < 1.0:
                    sl, si = torch.sort(logits_flat, descending=True)
                    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    si_rm = cp > top_p
                    si_rm[..., 1:] = si_rm[..., :-1].clone()
                    si_rm[..., 0] = 0
                    logits_flat[si_rm.scatter(1, si, si_rm)] = -np.inf
                logits = logits_flat.view(_b, _l, _v)

            probs = logits.softmax(dim=-1)
            if sample_policy == 'multinomial':
                x0 = dists.Categorical(logits=logits / temperature).sample()
                x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
            elif sample_policy == 'argmax':
                x0 = logits.argmax(-1)
                x0_p = torch.gather(probs, -1, x0.long()[..., None]).squeeze(-1)
            else:
                raise NotImplementedError(f"Unknown sample_policy: {sample_policy}")

            if edit_threshold <= 0:
                x0 = torch.where(mask_idx, x0, xt)

            if confidence_policy == 'mask_git':
                _alg_t = alg_temp * local_temp if dynamic_temperature else alg_temp
                confidence = torch.where(mask_idx, x0_p / _alg_t, torch.tensor(-np.inf, device=device))
                confidence = torch.softmax(confidence, dim=-1)
                select_index = torch.multinomial(confidence, num_samples=num_transfer)
            elif confidence_policy == 'mmada':
                _alg_t = alg_temp * local_temp if dynamic_temperature else alg_temp
                confidence = torch.log(x0_p.clamp(1e-20)) + _alg_t * _gumbel_noise(x0_p)
                confidence = torch.where(mask_idx, confidence, torch.tensor(-np.inf, device=device))
                _, select_index = torch.topk(confidence[0], k=num_transfer)
            elif confidence_policy == 'stratified':
                assert unmask_order is not None
                start = n_tokens - n_mask
                select_index = torch.tensor(
                    unmask_order[start: start + num_transfer],
                    device=x0.device, dtype=torch.long,
                )
            else:
                raise NotImplementedError(f"Unknown confidence_policy: {confidence_policy}")

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            transfer_index[0, select_index] = True
            xt[transfer_index] = x0[transfer_index]

            xt_is_mask = xt == img_mask_id
            if edit_threshold > 0:
                editable = (~xt_is_mask) & (~transfer_index)
                hi_conf = torch.where(editable, x0_p, torch.tensor(-torch.inf, device=device)) > edit_threshold
                changed = (x0 != xt) & hi_conf
                if changed.sum() > 0:
                    xt[changed] = x0[changed]
                    total_edited += changed.sum().item()

            if return_intermediate_steps:
                x0_inter = xt.clone()
                x0_inter[xt_is_mask] = x0[xt_is_mask]
                intermediate_x0s.append(x0_inter.cpu())

        xt = x0.clone()
        xt[xt == img_mask_id] = x0[xt == img_mask_id]
        x0_img = xt
        print(f"Total edited tokens: {total_edited}")

        if return_intermediate_steps:
            images_npy = self.decode_image_gen(
                torch.cat(intermediate_x0s).to(x0_img.device),
                image_resolution, image_resolution,
            )
            return [Image.fromarray(x) for x in images_npy]
        return Image.fromarray(
            self.decode_image_gen(x0_img, image_resolution, image_resolution)[0]
        )


AutoConfig.register("nemotron_labs_diffusion_image", NemotronLabsDiffusionImageConfig)
AutoModel.register(NemotronLabsDiffusionImageConfig, NemotronLabsDiffusionImageForMaskedDiffusion)
AutoModelForCausalLM.register(NemotronLabsDiffusionImageConfig, NemotronLabsDiffusionImageForMaskedDiffusion)
