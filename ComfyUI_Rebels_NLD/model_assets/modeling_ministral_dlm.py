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
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import random
import os
import sys
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutput
from transformers.utils import ModelOutput

from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask, or_masks

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.processing_utils import Unpack

from transformers.cache_utils import Cache, DynamicCache

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.generation import GenerationMixin

import math

from .chat_utils import generate_with_prefix_cache_block_diff
from .modeling_ministral import Ministral3Model, Ministral3PreTrainedModel, Ministral3Attention, apply_rotary_pos_emb, repeat_kv, _get_llama_4_attn_scale
from .configuration_ministral_dlm import MinistralDLMConfig

try:
    from flash_attn import flash_attn_func
except:
    print("flash attention not found, please install flash attention for better performance.")
__all__ = ["MinistralDiffEncoderModel", "MinistralFlexAttention"]

@dataclass
class MinistralDiffOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    causal_logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


# @torch.compile(dynamic=True, mode="reduce-overhead")
# @torch.compile(mode="default")
# @torch.compile(fullgraph=True, mode="reduce-overhead", dynamic=False)
@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs", dynamic=False)
def fused_flex_attention(q, k, v, block_mask=None):
    return flex_attention(q, k, v, block_mask=block_mask)


def _crop_dynamic_cache(past_key_values: DynamicCache, max_length: int):
    """Crop a DynamicCache to max_length, compatible with both old and new transformers."""
    if hasattr(past_key_values, 'crop'):
        past_key_values.crop(max_length)
    else:
        for layer_idx in range(len(past_key_values)):
            past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, :max_length]
            past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, :max_length]
        past_key_values._seen_tokens = max_length


def _extract_draft_kv_cache(past_key_values: DynamicCache, clean_len: int, block_length: int):
    """After quadratic decoding, extract only draft tokens (first of each block) from cache."""
    for layer_idx in range(len(past_key_values)):
        if hasattr(past_key_values, 'layers'):
            layer_cache = past_key_values.layers[layer_idx]
            k, v = layer_cache.keys, layer_cache.values
        else:
            k = past_key_values.key_cache[layer_idx]
            v = past_key_values.value_cache[layer_idx]

        clean_k, draft_k = k[:, :, :clean_len], k[:, :, clean_len::block_length + 1]
        clean_v, draft_v = v[:, :, :clean_len], v[:, :, clean_len::block_length + 1]
        new_k = torch.cat([clean_k, draft_k], dim=2)
        new_v = torch.cat([clean_v, draft_v], dim=2)

        if hasattr(past_key_values, 'layers'):
            layer_cache.keys = new_k
            layer_cache.values = new_v
        else:
            past_key_values.key_cache[layer_idx] = new_k
            past_key_values.value_cache[layer_idx] = new_v

    past_key_values._seen_tokens = clean_len + block_length

# with reference to https://github.com/pytorch-labs/attention-gym/blob/main/examples/flex_attn.ipynb
class MinistralFlexAttention(Ministral3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_seq_length = getattr(self.config, 'max_seq_length', 4096)
        self.block_size_orig = self.config.block_size
        self.bidirectional_mask = None
        if self.config.dlm_paradigm == 'bidirectional':
            self.bidirectional_mask = self.compute_block_mask(mode='bidirectional')
        elif self.config.dlm_paradigm == 'autoregressive':
            self.autoregressive_mask = self.compute_block_mask(mode='autoregressive')
        elif self.config.dlm_paradigm == 'block_diff':
            self.block_diff_mask = None
        elif self.config.dlm_paradigm == 'sbd_block_diff':
            self.sbd_block_diff_mask = None
        else:
            raise ValueError(f"Unknown attention mode: {self.config.dlm_paradigm}")

        self.block_size = self.block_size_orig
        self.mode = self.config.dlm_paradigm
        self._quadratic_block_mask = {}

        import torch._dynamo.config as dcfg
        dcfg.cache_size_limit = 512


    def _get_sbd_inference_quadratic_decoding_block_mask(self, block_length: int):
        if block_length not in self._quadratic_block_mask:
            draft_len = block_length * (block_length + 1)

            def quadratic(b, h, q_idx, kv_idx):
                first_clean = torch.logical_and(
                    kv_idx % (block_length + 1) == 0,
                    kv_idx < draft_len,
                )
                first_clean = torch.logical_and(first_clean, q_idx >= kv_idx)
                block_q = q_idx // (block_length + 1)
                block_kv = kv_idx // (block_length + 1)
                same_block = torch.logical_and(block_q == block_kv, q_idx < draft_len)
                same_block_except_first = torch.logical_and(
                    same_block,
                    q_idx % (block_length + 1) != 0,
                )
                draft_part = torch.logical_or(first_clean, same_block_except_first)
                clean_part = kv_idx >= draft_len
                return torch.logical_or(draft_part, clean_part)

            block_mask = create_block_mask(
                quadratic,
                B=None,
                H=None,
                Q_LEN=draft_len,
                KV_LEN=draft_len + self.config.max_position_embeddings,
                device="cuda",
            )

            self._quadratic_block_mask[block_length] = block_mask

        return self._quadratic_block_mask[block_length]


    def set_attention_mode(self, mode, block_size=None):
        self.mode = mode
        self.block_size = block_size

    def compute_block_mask(self, mode, q_len=None, block_size=None):

        def bidirectional_mask(b, h, q, kv): 
            return (q >= kv) | (q < kv)
        
        def autoregressive_mask(b, h, q, kv):
            return (q >= kv)

        def block_diff_mask(block_size, b, h, q_idx, kv_idx, n):
            x0_flag_q = (q_idx >= n)
            x0_flag_kv = (kv_idx >= n)

            # Compute block indices
            block_q = torch.where(x0_flag_q == 1,
                                    (q_idx - n) // block_size,
                                    q_idx // block_size)
            block_kv = torch.where(x0_flag_kv == 1,
                                    (kv_idx - n) // block_size,
                                    kv_idx // block_size)

            # **1. Block Diagonal Mask (M_BD) **
            block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

            # **2. Offset Block-Causal Mask (M_OBC) **
            offset_block_causal = (
                (block_q > block_kv)
                & (x0_flag_kv == 1)
                & (x0_flag_q == 0)
            )

            # **3. Block-Causal Mask (M_BC) **
            block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

            # **4. Combine Masks **
            return block_diagonal | offset_block_causal | block_causal
        

        def sbd_block_diff_mask(block_size, b, h, q_idx, kv_idx, n):
            x0_flag_q = (q_idx >= n)
            x0_flag_kv = (kv_idx >= n)

            # Compute block indices
            block_q = torch.where(x0_flag_q == 1,
                                    (q_idx - n) // block_size,
                                    q_idx // block_size)
            block_kv = torch.where(x0_flag_kv == 1,
                                    (kv_idx - n) // block_size,
                                    kv_idx // block_size)

            # **1. Block Diagonal Mask (M_BD) **
            block_diagonal = (block_q == block_kv) & (x0_flag_kv == 0) & (x0_flag_q == 0)

            # **2. Offset Block-Causal Mask (M_OBC) **
            offset_block_causal = (
                (block_q > block_kv)
                & (x0_flag_kv == 1)
                & (x0_flag_q == 0)
            )

            # **3. Fully Causal Mask (M_BC) **
            fully_causal = (q_idx >= kv_idx) & (x0_flag_kv == 1) & (x0_flag_q == 1)

            # **4. Combine Masks **
            return block_diagonal | offset_block_causal | fully_causal
        
        def modality_indices_based_mask(block_size, b, h, q_idx, kv_idx, image_doc_id):
            return  (image_doc_id[b, q_idx] > 0) & (image_doc_id[b, q_idx] == image_doc_id[b, kv_idx])

        if mode == 'bidirectional':
            attn_mask = bidirectional_mask
        elif mode == 'autoregressive':
            attn_mask = autoregressive_mask
        elif mode == 'block_diff':
            assert block_size is not None
            attn_mask = lambda b, h, q, kv: block_diff_mask(block_size, b, h, q, kv, self.max_seq_length)
        elif mode == 'sbd_block_diff':
            assert block_size is not None
            attn_mask = lambda b, h, q, kv: sbd_block_diff_mask(block_size, b, h, q, kv, self.max_seq_length)
        else:
            raise ValueError(f"Unknown attention mode: {mode}")

        if q_len is not None:
            Q_LEN = q_len
        else:
            if mode in ['block_diff', 'sbd_block_diff']:
                Q_LEN = self.max_seq_length * 2
            else:
                Q_LEN = self.max_seq_length

        block_mask = create_block_mask(
            attn_mask, B=None, H=None, Q_LEN=Q_LEN, KV_LEN=Q_LEN
        )

        return block_mask


    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        is_training: bool = True,
        overwrite_block_mask = None,
        overwrite_attn_impl = None,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if overwrite_attn_impl == 'base':
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                is_training=is_training,
                use_cache=use_cache,
                **kwargs,
            )
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        if self.mode in ['block_diff', 'sbd_block_diff'] and is_training:
            # Split query and key states in half along sequence length dimension
            q1, q2 = query_states.chunk(2, dim=2)
            k1, k2 = key_states.chunk(2, dim=2)
            
            # Apply RoPE independently to each half
            q1, k1 = apply_rotary_pos_emb(q1, k1, cos, sin)
            q2, k2 = apply_rotary_pos_emb(q2, k2, cos, sin)
            
            # Recombine the halves
            query_states = torch.cat([q1, q2], dim=2)
            key_states = torch.cat([k1, k2], dim=2)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states * _get_llama_4_attn_scale(
            cache_position,
            self.config.rope_parameters.get("llama_4_scaling_beta"),
            self.config.rope_parameters.get("original_max_position_embeddings"),
        ).to(query_states.dtype)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            if use_cache:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:  ## if use_cache == False, do not update cache
                old_k, old_v = past_key_values.layers[self.layer_idx].keys, past_key_values.layers[self.layer_idx].values
                key_states   = torch.cat([old_k, key_states], dim=-2)
                value_states = torch.cat([old_v, value_states], dim=-2)


        self_spec_inference_mode = getattr(self.config, "self_spec_inference_mode", None)
        if self_spec_inference_mode is not None:
            if self_spec_inference_mode == "quadratic":
                block_length = getattr(self.config, "block_length", None) or getattr(self.config, "block_size", None)
                if block_length is None:
                    raise ValueError("SBD quadratic decoding requires block_length in config.")
                if past_key_values is not None:
                    seq_len = key_states.shape[2]
                    draft_len = block_length * (block_length + 1)

                    clean_keys = key_states[:, :, :-draft_len]
                    draft_keys = key_states[:, :, -draft_len:]
                    clean_values = value_states[:, :, :-draft_len]
                    draft_values = value_states[:, :, -draft_len:]
                    key_states = torch.cat([draft_keys, clean_keys], dim=2)
                    value_states = torch.cat([draft_values, clean_values], dim=2)

                    block_mask: BlockMask = self._get_sbd_inference_quadratic_decoding_block_mask(
                        block_length=block_length
                    )
                    block_mask.seq_lengths = (draft_len, seq_len)
                else:
                    seq_len = query_states.shape[2]
                    draft_len = block_length * (block_length + 1)
                    clean_len = seq_len - draft_len

                    def _causal_mask(b, h, q_idx, kv_idx):
                        return torch.logical_and(q_idx >= kv_idx, q_idx < clean_len)

                    def _draft2clean_mask(b, h, q_idx, kv_idx):
                        full_clean = torch.logical_and(q_idx >= clean_len, kv_idx <= clean_len)
                        first_clean = torch.logical_and(
                            q_idx >= clean_len, (kv_idx - clean_len) % (block_length + 1) == 0
                        )
                        first_clean = torch.logical_and(first_clean, q_idx >= kv_idx)
                        return torch.logical_or(full_clean, first_clean)

                    def _draft_mask(b, h, q_idx, kv_idx):
                        block_q = (q_idx - clean_len) // (block_length + 1)
                        block_kv = (kv_idx - clean_len) // (block_length + 1)
                        quadrant = torch.logical_and(q_idx >= clean_len, kv_idx >= clean_len)
                        same_block = torch.logical_and(block_q == block_kv, quadrant)
                        same_block_except_first = torch.logical_and(
                            same_block,
                            (q_idx - clean_len) % (block_length + 1) != 0,
                        )
                        return torch.logical_and(block_q == block_kv, same_block_except_first)

                    mask = or_masks(_causal_mask, _draft2clean_mask)
                    mask = or_masks(mask, _draft_mask)

                    block_mask = create_block_mask(
                        mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
                    )

                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                attn_output = flex_attention(query_states, key_states, value_states, block_mask=block_mask)
                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, None

            elif self_spec_inference_mode == "default":
                block_length = getattr(self.config, "block_length", None) or getattr(self.config, "block_size", None)
                if block_length is None:
                    raise ValueError("SBD default decoding requires block_length in config.")
                seq_len = query_states.shape[2]
                prefix_len = seq_len - block_length

                def _clean_q_mask(b, h, q_idx, kv_idx):
                    return torch.logical_and(q_idx >= kv_idx, q_idx < prefix_len)

                def _noisy_q_mask(b, h, q_idx, kv_idx):
                    return q_idx >= prefix_len

                block_mask = create_block_mask(
                    or_masks(_clean_q_mask, _noisy_q_mask),
                    B=None,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                )

                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                attn_output = flex_attention(query_states, key_states, value_states, block_mask=block_mask)
                attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, None
        
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            if overwrite_block_mask is not None:
                block_mask = overwrite_block_mask
                if block_mask == 'full':
                    block_mask = None
            else:
                if self.mode == 'bidirectional':
                    block_mask = None
                    overwrite_attn_impl = 'flash_attn'
                    # if self.bidirectional_mask is None or q_len != self.bidirectional_mask.shape[-2]:
                    #     block_mask = self.compute_block_mask(mode='bidirectional', q_len=q_len)
                    # else:
                    #     block_mask = self.bidirectional_mask

                elif self.mode == 'autoregressive':
                    if self.autoregressive_mask is None or q_len != self.autoregressive_mask.shape[-2]:
                        block_mask = self.compute_block_mask(mode='autoregressive', q_len=q_len)
                    else:
                        block_mask = self.autoregressive_mask

                elif self.mode == 'block_diff':
                    if self.block_diff_mask is None or self.block_size != self.block_size_orig or q_len != self.block_diff_mask.shape[-2]:
                        block_mask = self.compute_block_mask(mode='block_diff', block_size=self.block_size, q_len=q_len)
                    else:
                        block_mask = self.block_diff_mask
                elif self.mode == 'sbd_block_diff':
                    if self.sbd_block_diff_mask is None or self.block_size != self.block_size_orig or q_len != self.sbd_block_diff_mask.shape[-2]:
                        block_mask = self.compute_block_mask(mode='sbd_block_diff', block_size=self.block_size, q_len=q_len)
                    else:
                        block_mask = self.sbd_block_diff_mask
                else:
                    raise ValueError(f"Unknown attention mode: {self.mode}")
            if overwrite_attn_impl == 'flash_attn':
                
    
                # FlashAttention expects (batch, seqlen, nheads, headdim)
                # Ensure your tensors are in this layout or permute them here
                #print(query_states.shape,key_states.shape,value_states.shape)
                if self.diffusion_lm:
                    causal = False
                else:
                    causal = True
                attn_output = flash_attn_func(
                    query_states.transpose(1,2), 
                    key_states.transpose(1,2), 
                    value_states.transpose(1,2), 
                    dropout_p=0.0,      # Set your dropout probability
                    softmax_scale=None, # Defaults to 1/sqrt(head_dim)
                    causal=causal         # Set to True if using a causal block_mask logic
                ).transpose(1,2)
                
            else:
                attn_output = fused_flex_attention(query_states, key_states, value_states, block_mask=block_mask)
            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()

            attn_output = self.o_proj(attn_output)

            return attn_output, None


def gumbel_topk(log_w: torch.Tensor, k: int) -> torch.Tensor:
    """Return a Bool mask of length len(log_w) with exactly k True."""
    g = -torch.log(-torch.log(torch.rand_like(log_w) + 1e-9) + 1e-9)
    topk = torch.topk(log_w + g, k).indices
    mask = torch.zeros_like(log_w, dtype=torch.bool)
    mask[topk] = True
    return mask


class MinistralDiffEncoderModel(Ministral3PreTrainedModel, GenerationMixin):
    """
    A single model with:
      - a bidirectional encoder + diffusion‐LM head over A
      - a causal decoder + LM head over B, conditioned on F_A
    """

    # Shared/tied tensors that can appear dynamically based on config.
    # Registering these patterns lets save_pretrained() deduplicate safely.
    # _dynamic_tied_weights_keys = [
    #     r"encoder\.embed_tokens\.weight",
    #     r"diffusion_head\.weight",
    #     r"encoder\.vision_tower(?:\.vision_tower)?\.visual_bridge_model\.quantizer\.quantize\.codebooks\.\d+\.(?:embed|embed_ema|cluster_size_ema)",
    # ]

    def __init__(self, config: MinistralDLMConfig):
        super().__init__(config)

        self.mask_token_id = config.mask_token_id

        diffusion_config = copy.deepcopy(config)
        diffusion_config.diffusion_lm = True

        use_flex = getattr(config, 'enable_self_spec', False)

        if config.dlm_paradigm in ['block_diff', 'sbd_block_diff']:
            diffusion_config.attn_class = MinistralFlexAttention
        elif config.dlm_paradigm in ['bidirectional', 'autoregressive']:
            diffusion_config.attn_class = MinistralFlexAttention if use_flex else Ministral3Attention
            if config.dlm_paradigm == 'autoregressive':
                diffusion_config.diffusion_lm = False
        else:
            raise ValueError(f"Unsupported DLM paradigm: {config.dlm_paradigm}")
        
        self.encoder = Ministral3Model(diffusion_config)
        self.diffusion_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size

        self.current_iter_ratio = None

        self.post_init()


    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.diffusion_head

    def set_output_embeddings(self, new_embeddings):
        self.diffusion_head = new_embeddings


    def forward_process(self, input_ids, eps=1e-3, block_size=None, loss_mask=None):
        b, l = input_ids.shape
        device = input_ids.device

        if self.config.dp_varying_mask_ratio:
            # Enable different random seeds for each DP rank during sampling
            import torch.distributed as dist
            dp_rank = 0
            if dist.is_initialized():
                try:
                    dp_rank = dist.get_rank()
                except Exception:
                    dp_rank = 0
            # Use a local generator to avoid affecting global RNG state
            generator = torch.Generator(device=device)
            generator.manual_seed(torch.seed() + dp_rank)
        else:
            generator = None
            
        if self.config.adaptive_mask_rate:
            assert block_size is not None

            # --- simple linear window mapping ---
            bs_min = getattr(self.config, "t_bs_min", 16)
            bs_max = getattr(self.config, "t_bs_max", 128)
            w = getattr(self.config, "t_window_width", 0.6)  # fixed width

            # fraction in [0,1] (unclamped first)
            frac = (float(block_size) - float(bs_min)) / max(1.0, float(bs_max - bs_min))
            # upper bound decreases linearly from 1.0 -> 0.5
            u_max = 1.0 - w * frac
            # clamp to [0.6, 1.0] to handle bs outside [bs_min, bs_max]
            u_max = max(0.6, min(1.0, u_max))
            u_min = u_max - w  # ensures width = w

            # sample t ~ Uniform(u_min, u_max)
            t = u_min + (u_max - u_min) * torch.rand(b, device=device, generator=generator)
        else:
            t = torch.rand(b, device=device, generator=generator)
        
        p_mask = (1 - eps) * t + eps  # shape: (b,)
        p_mask = p_mask[:, None].expand(-1, l)  # shape: (b, l)

        masked_indices = torch.rand((b, l), device=device) < p_mask

        if loss_mask is not None:
            masked_indices[loss_mask == 0] = 0

        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)        

        return noisy_batch, masked_indices, p_mask


    def forward_process_exp(
        self,
        input_ids: torch.Tensor,
        eps: float = 1e-3,
        block_size: int | None = None,
        half_life_ratio: float = 0.25, # λ = ln 2 / (half_life_ratio·L)
        loss_mask: Optional[torch.Tensor] = None,
    ):
        """
        Two-stage corruption with optional per-block sampling.
        • Stage 1:  m ~ U(eps, 1)   →   k = round(m · len)  (exact budget).
        • Stage 2:  sample exactly k positions with weights
                    w_i(m) = exp[ λ · (1−m) · i ]   (late-heavy when m→0,
                                                     uniform when m→1).
          If `block_size` is given, the procedure is run *independently*
          inside each contiguous block of that length (last block may be shorter).
          When block_size is provided, m is sampled per-block and p_mask is per-block.
        Args
        ----
        input_ids : (B, L)  LongTensor
        eps       : minimum corruption ratio
        block_size: if not None, operate block-wise with per-block m sampling
        half_life_ratio : controls steepness when m→0
        """
        B, L = input_ids.shape
        device = input_ids.device
        dtype  = torch.float32

        masked_indices = torch.zeros((B, L), dtype=torch.bool, device=device)
        p_mask = torch.zeros((B, L), dtype=dtype, device=device)

        # ---------- Stage 1 & 2: whole-sentence or block-wise -------------------
        for b in range(B):
            if block_size is None:
                # ---------- Per-batch sampling (original behavior) ----------
                m = eps + (1.0 - eps) * torch.rand(1, device=device).item()   # scalar
                k_tot = int(round(m * L))
                k_tot = max(1, min(k_tot, L))  # clamp to [1, L]
                
                # Fill p_mask for this batch
                p_mask[b, :] = m
                
                slope = 1.0 - m          # ∈ [0,1]; 0 ⇒ uniform, 1 ⇒ late-heavy
                
                # ------- single pool over the whole sentence -------------
                lam_base = math.log(2.0) / (half_life_ratio * L) # base decay rate (λ when slope=1)

                pos   = torch.arange(L, device=device, dtype=dtype)
                log_w = (lam_base * slope * pos).clone()

                masked_indices[b] = gumbel_topk(log_w, k_tot)

            else:
                # ---------- Per-block sampling ----------
                num_blocks = math.ceil(L / block_size)
                lam_base = math.log(2.0) / (half_life_ratio * block_size) # base decay rate (λ when slope=1)

                for blk in range(num_blocks):
                    start = blk * block_size
                    end   = min((blk + 1) * block_size, L)
                    blk_len = end - start

                    # Sample m per block
                    m_blk = eps + (1.0 - eps) * torch.rand(1, device=device).item()
                    
                    # Fill p_mask for this block
                    p_mask[b, start:end] = m_blk
                    
                    # per-block budget
                    k_blk = int(round(m_blk * blk_len))
                    k_blk = max(0, min(k_blk, blk_len))
                    if k_blk == 0:
                        continue

                    slope = 1.0 - m_blk          # ∈ [0,1]; 0 ⇒ uniform, 1 ⇒ late-heavy

                    pos   = torch.arange(blk_len, device=device, dtype=dtype)
                    log_w = lam_base * slope * pos

                    blk_mask = gumbel_topk(log_w, k_blk)
                    masked_indices[b, start:end] = blk_mask

        if loss_mask is not None:
            masked_indices[loss_mask == 0] = 0

        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)
        return noisy_batch, masked_indices, p_mask
    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor]   = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor]       = None,
        split_len: Optional[int]                 = None,
        past_key_values: Optional[Cache]         = None,
        block_size: Optional[int]                = None,
        block_diff_ppl: bool                     = False,
        eps: float                               = 1e-3,
        is_teacher: bool                        = False,
        masked_indices: Optional[torch.Tensor]   = None,
        p_mask: Optional[torch.Tensor]           = None,
        teacher_logits: Optional[torch.Tensor]   = None,
        masked_indices_teacher: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        ce_loss_weight: float = 1.0,
        output_last_hidden_states_only: bool = False,
        skip_loss: bool = False,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        if input_ids is None:
            if inputs_embeds is None:
                raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")
            batch_size, seq_len = inputs_embeds.shape[:2]
            if labels is not None:
                raise ValueError("`labels` training path requires `input_ids`.")
        else:
            batch_size, seq_len = input_ids.shape


        if self.config.dlm_paradigm == 'bidirectional' or self.config.dlm_paradigm == 'autoregressive':
            if labels is not None and torch.rand(1) < self.config.random_length_prob:
                raise NotImplementedError("Random length training not yet implemented for bidirectional/autoregressive paradigms.")
                random_length = torch.randint(2, input_ids.shape[1] + 1, (1,))
                input_ids = input_ids[:, :random_length]
                labels = labels[:, :random_length]
                
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :random_length]
                if position_ids is not None:
                    position_ids = position_ids[:, :random_length]
                if loss_mask is not None:
                    loss_mask = loss_mask[:, :random_length]

        elif self.config.dlm_paradigm in ['block_diff', 'sbd_block_diff']:
            if labels is not None and block_size is None:
                if torch.rand(1) < self.config.random_length_prob:
                    block_size = torch.randint(1, 8, (1,)).item() * 4  ## [4, 32] divisible by 4
                else:
                    block_size = self.config.block_size

        else:
            raise ValueError(f"Unknown dLM paradigm: {self.config.dlm_paradigm}")

        if labels is not None and self.config.dlm_paradigm != 'autoregressive':
            if masked_indices is not None:
                # assert p_mask is not None

                if loss_mask is not None:
                    masked_indices[loss_mask == 0] = 0

                noisy_inputs = torch.where(masked_indices, self.mask_token_id, input_ids)

            else:
                if self.config.tok_mask_half_life_ratio is not None:
                    noisy_inputs, masked_indices, p_mask = self.forward_process_exp(input_ids, eps=eps, block_size=block_size, half_life_ratio=self.config.tok_mask_half_life_ratio, loss_mask=loss_mask)
                else:
                    noisy_inputs, masked_indices, p_mask = self.forward_process(input_ids, eps=eps, block_size=block_size, loss_mask=loss_mask)

        else:
            noisy_inputs = input_ids
            masked_indices = None
            p_mask = None

        if self.config.dlm_paradigm in ['block_diff', 'sbd_block_diff']:
            for layer in self.encoder.layers:
                if hasattr(layer.self_attn, 'set_attention_mode'):
                    layer.self_attn.set_attention_mode(self.config.dlm_paradigm, block_size=block_size)

        input_ids_len = noisy_inputs.shape[1] if noisy_inputs is not None else seq_len
        if labels is not None and self.config.dlm_paradigm in ['block_diff', 'sbd_block_diff']:
            if position_ids is None:
                position_ids = torch.arange(input_ids_len, device=noisy_inputs.device).unsqueeze(0)
            noisy_inputs = torch.cat([noisy_inputs, input_ids], dim=1)

        if block_diff_ppl:
            if position_ids is None:
                position_ids = torch.arange(input_ids_len // 2, device=noisy_inputs.device).unsqueeze(0)

        enc_out  = self.encoder(
            past_key_values=past_key_values,
            input_ids=noisy_inputs,
            inputs_embeds=inputs_embeds if noisy_inputs is None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_training=(labels is not None) or (block_diff_ppl),
            **kwargs,
        )

        if output_last_hidden_states_only:
            return BaseModelOutput(last_hidden_state=enc_out.last_hidden_state)

        logits = self.diffusion_head(enc_out.last_hidden_state)  # (batch, len_B, vocab)
        causal_logits = None

        if labels is not None and self.config.dlm_paradigm in ['block_diff', 'sbd_block_diff']:
            if self.config.dlm_paradigm == 'sbd_block_diff':
                causal_logits = logits[:, input_ids_len:]
            else:
                causal_logits = None

            logits = logits[:, :input_ids_len]

        loss = None
        if labels is not None and not skip_loss:
            if self.config.dlm_paradigm == 'autoregressive':
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                if loss_mask is None:
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    loss = loss_fct(shift_logits, shift_labels)

                else:
                    loss_mask = loss_mask[..., 1:].contiguous()

                    loss_fct = CrossEntropyLoss(reduction='none')
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    
                    token_losses = loss_fct(shift_logits, shift_labels)
                                    
                    flat_loss_mask = loss_mask.reshape(-1)
                    loss = token_losses[flat_loss_mask == 1].sum() / flat_loss_mask.sum()

            else:
                # Handle DREAM vs LLADA style losses
                if hasattr(self.config, 'dlm_type') and self.config.dlm_type == 'dream':
                    logits = logits[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()
                    masked_indices = masked_indices[:, 1:]
                    p_mask = p_mask[:, 1:]

                if self.config.ada_perm_ratio_per_block is not None:
                    # Only compute loss for the top ada_perm_ratio_per_block tokens by confidence within each block
                    block_size = self.config.block_size
                    batch_size, seq_len = masked_indices.shape
                    num_blocks = seq_len // block_size
                    
                    # Get the max logit (confidence) for each position
                    confidence = logits.max(dim=-1).values.detach()  # (batch_size, seq_len)
                    
                    # Create a mask for tokens to include in loss
                    selected_mask = torch.zeros_like(masked_indices, dtype=torch.bool)
                    
                    for blk in range(num_blocks):
                        start = blk * block_size
                        end = min((blk + 1) * block_size, seq_len)
                        
                        # Get masked indices within this block
                        block_masked = masked_indices[:, start:end]  # (batch_size, block_len)
                        block_confidence = confidence[:, start:end]  # (batch_size, block_len)
                        
                        for b in range(batch_size):
                            # Get positions that are masked in this block for this batch
                            masked_positions = torch.where(block_masked[b])[0]
                            num_masked = len(masked_positions)
                            
                            if num_masked > 0:
                                # Number of tokens to keep (top by confidence)
                                k = min(max(1, int(block_size * self.config.ada_perm_ratio_per_block)), num_masked)
                                
                                # Get confidence values for masked positions
                                masked_confidence = block_confidence[b, masked_positions]
                                
                                # Get indices of top-k confident tokens
                                _, topk_indices = torch.topk(masked_confidence, k)
                                selected_positions = masked_positions[topk_indices]
                                
                                # Mark these positions in the selected mask
                                selected_mask[b, start + selected_positions] = True
                    
                    # Calculate loss only for selected positions
                    token_loss = torch.nn.functional.cross_entropy(
                        logits[selected_mask],
                        labels[selected_mask],
                        reduction='none'
                    ) / p_mask[selected_mask]

                    num_mask_tokens = selected_mask.sum()

                else:
                    # Calculate token-wise cross entropy loss for masked positions in B
                    token_loss = torch.nn.functional.cross_entropy(
                        logits[masked_indices], 
                        labels[masked_indices], 
                        reduction='none'
                    ) / p_mask[masked_indices]

                    num_mask_tokens = masked_indices.sum()

                if self.config.global_loss_avg:
                    loss = token_loss.sum()
                else:
                    loss = token_loss.sum() / num_mask_tokens
                
                if self.config.ada_dlm_loss_ratio is not None:
                    assert self.current_iter_ratio is not None
                    assert self.config.dlm_loss_weight is not None

                    dlm_loss_weight = min(self.config.dlm_loss_weight, self.current_iter_ratio / self.config.ada_dlm_loss_ratio * self.config.dlm_loss_weight)
                    loss = dlm_loss_weight * loss

                elif self.config.dlm_loss_weight is not None:
                    loss = self.config.dlm_loss_weight * loss

                if self.config.dlm_paradigm == 'sbd_block_diff':
                    causal_logits = causal_logits[..., :-1, :].contiguous()
                    causal_logits = causal_logits.view(-1, causal_logits.size(-1))

                    if hasattr(self.config, 'dlm_type') and self.config.dlm_type == 'dream':
                        causal_labels = labels.view(-1)
                    else: 
                        causal_labels = labels[..., 1:].contiguous().view(-1)
                    
                    if self.config.global_loss_avg:
                        loss_fct = CrossEntropyLoss(reduction='sum')
                        ar_loss = loss_fct(causal_logits, causal_labels)

                        self.loss_diffusion = loss.detach().item() / num_mask_tokens
                        self.loss_ar = ar_loss.detach().item() / seq_len

                        loss = loss + self.config.ar_loss_weight * ar_loss
                    else:
                        loss_fct = CrossEntropyLoss()
                        ar_loss = loss_fct(causal_logits, causal_labels)

                        self.loss_diffusion = loss.detach().item()
                        self.loss_ar = ar_loss.detach().item()

                        loss = loss + self.config.ar_loss_weight * ar_loss
                
                if self.config.global_loss_avg:
                    if self.config.dlm_paradigm == 'sbd_block_diff':
                        loss = (loss, num_mask_tokens + int(self.config.ar_loss_weight * seq_len))
                    else: 
                        loss = (loss, num_mask_tokens)

        return MinistralDiffOutputWithPast(
            loss=loss if not is_teacher else logits,
            logits=logits,
            causal_logits=causal_logits,
            past_key_values=enc_out.past_key_values,
            hidden_states=None,
            attentions=None,
        )


    def generate_diffusion(self, prompt_ids, max_new_tokens=512, steps=512, block_length=32, shift_logits=False, threshold=0.9, causal_context=True, temperature=0, eos_token_id=None, max_thinking_tokens=None, end_think_token_id=None, step_ratio=None,prompt_embeds=None,**kwargs):
        if prompt_embeds is None and prompt_ids is not None and torch.is_floating_point(prompt_ids):
            prompt_embeds = prompt_ids
            prompt_ids = None

        if (prompt_ids is None) == (prompt_embeds is None):
            raise ValueError("Exactly one of `prompt_ids` or `prompt_embeds` must be provided.")

        if eos_token_id is None:
            eos_token_id = getattr(self.config, 'eos_token_id', None)
        if step_ratio is not None:
            steps_per_block = int(block_length * step_ratio)
            num_blocks = max_new_tokens // block_length
            steps = steps_per_block * num_blocks
        out_ids, nfe = generate_with_prefix_cache_block_diff(
                        model=self,
                        prompt=prompt_ids,
                        prompt_embeds=prompt_embeds,
                        gen_length=max_new_tokens,
                        steps=steps,
                        block_length=block_length,
                        remasking="low_confidence",
                        temperature=temperature,
                        mask_id=self.mask_token_id,
                        threshold=threshold,
                        shift_logits=shift_logits,
                        neg_entropy=False,
                        causal_context=causal_context,
                        eos_token_id=eos_token_id,
                        max_thinking_tokens=max_thinking_tokens,
                        end_think_token_id=end_think_token_id,
                    )

        return out_ids, nfe


    @torch.no_grad()
    def sbd_inference_diffusion_quadratic(
        self,
        clean_input_ids: Optional[torch.Tensor],
        draft_input_ids: torch.Tensor,
        block_length: int,
        draft_only: bool = False,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ):
        enc_config = self.encoder.config
        enc_config.use_sbd_objective = True
        enc_config.block_length = block_length

        if draft_only:
            assert clean_input_ids is not None

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            enc_config.self_spec_inference_mode = "default"
            input_ids = torch.cat([clean_input_ids, draft_input_ids], dim=-1)
            outputs = self.encoder(
                input_ids=input_ids,
                position_ids=None,
                past_key_values=past_key_values,
                use_cache=use_cache,
                is_training=False,
            )

            hidden_states = outputs.last_hidden_state
            logits = self.diffusion_head(hidden_states)

            past_key_values = getattr(outputs, "past_key_values", None)
            if use_cache and past_key_values is not None:
                _crop_dynamic_cache(past_key_values, clean_input_ids.shape[1])

            return logits, past_key_values
        else:
            enc_config.self_spec_inference_mode = "quadratic"

            draft_len = block_length * (block_length + 1)
            draft_input_ids = torch.cat(
                [
                    draft_input_ids.view(-1, block_length, 1),
                    torch.full(
                        (draft_input_ids.shape[0], block_length, block_length),
                        fill_value=self.config.mask_token_id,
                        device=draft_input_ids.device,
                    ),
                ],
                dim=-1,
            ).view(-1, draft_len)

            if use_cache:
                assert past_key_values is not None, (
                    "Past key values should be provided when using cache, e.g. run draft_only=True first."
                )
                assert clean_input_ids is None, (
                    "Clean input ids should already be in cache, thus none should be provided."
                )
                clean_len = past_key_values.get_seq_length()
                input_ids = draft_input_ids
            else:
                clean_len = clean_input_ids.shape[1]
                input_ids = torch.cat([clean_input_ids, draft_input_ids], dim=-1)

            per_block_position_ids = torch.arange(
                clean_len, clean_len + block_length + 1, device=draft_input_ids.device
            )[None,].repeat(block_length, 1)
            per_block_position_ids += torch.arange(block_length, device=draft_input_ids.device).view(-1, 1)

            if use_cache:
                position_ids = per_block_position_ids.view(-1)[None,]
            else:
                clean_position_ids = torch.arange(clean_len, device=draft_input_ids.device)
                position_ids = torch.cat([clean_position_ids, per_block_position_ids.view(-1)], dim=-1)[None,]

            outputs = self.encoder(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                is_training=False,
            )

            hidden_states = outputs.last_hidden_state
            logits = self.diffusion_head(hidden_states)
            past_key_values = getattr(outputs, "past_key_values", None)

            if use_cache and past_key_values is not None:
                _extract_draft_kv_cache(past_key_values, clean_len, block_length)

            return logits, past_key_values


    @torch.no_grad()
    def ar_generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
        max_thinking_tokens: Optional[int] = None,
        end_think_token_id: Optional[int] = None,
    ) -> tuple:
        """Autoregressive generation calling the encoder directly (injected by build_hf_tidar_repo).

        Bypasses MinistralDiffEncoderModel.forward() to avoid diffusion-specific
        code paths. Calls self.encoder (Ministral3Model) with explicit cache_position,
        position_ids, and use_cache so the KV cache and causal masking behave
        identically to MistralForCausalLM / vLLM.

        Returns:
            (output_ids, nfe) where output_ids includes the prompt.
        """
        for layer in self.encoder.layers:
            if hasattr(layer.self_attn, 'diffusion_lm'):
                layer.self_attn.diffusion_lm = False

        if eos_token_id is None:
            eos_token_id = getattr(self.config, 'eos_token_id', None)

        device = prompt_ids.device
        batch_size, prompt_len = prompt_ids.shape

        past_key_values = DynamicCache()
        cache_position = torch.arange(prompt_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        enc_out = self.encoder(
            input_ids=prompt_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        past_key_values = enc_out.past_key_values
        next_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)

        generated_tokens = []
        nfe = 0

        for step in range(max_new_tokens):
            nfe += 1

            if temperature > 0:
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logit, dim=-1, keepdim=True)

            # ---- thinking budget enforcement ----
            if end_think_token_id is not None and max_thinking_tokens is not None:
                if step >= max_thinking_tokens:
                    if generated_tokens:
                        gen_tensor = torch.cat(generated_tokens, dim=1)
                        has_end_think = (gen_tensor == end_think_token_id).any(dim=1)
                    else:
                        has_end_think = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    for b in range(batch_size):
                        if not has_end_think[b]:
                            next_token[b] = end_think_token_id

            generated_tokens.append(next_token)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            if step < max_new_tokens - 1:
                cur_pos = prompt_len + step
                step_cache_pos = torch.tensor([cur_pos], device=device)
                step_pos_ids = step_cache_pos.unsqueeze(0).expand(batch_size, -1)

                enc_out = self.encoder(
                    input_ids=next_token,
                    position_ids=step_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=step_cache_pos,
                )
                past_key_values = enc_out.past_key_values
                next_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)

        all_generated = torch.cat(generated_tokens, dim=1)
        output_ids = torch.cat([prompt_ids, all_generated], dim=1)
        return output_ids, nfe


    @torch.no_grad()
    def self_spec_generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        steps: int = 128,
        block_length: int = 16,
        ar_mix_weight: Optional[float] = None,
        temperature: float = 0.0,
        mask_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        max_thinking_tokens: Optional[int] = None,
        end_think_token_id: Optional[int] = None,
    ):
        self.config.use_sbd_objective = True
        self.config.dlm_paradigm = "sbd"

        if prompt_ids.shape[0] != 1:
            raise ValueError("Self speculation quadratic decoding currently requires batch_size == 1")

        token_mask_id = mask_token_id if mask_token_id is not None else self.config.mask_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)

        x = torch.full(
            (1, prompt_ids.shape[1] + max_new_tokens + block_length * 2),
            token_mask_id,
            dtype=torch.long,
            device=prompt_ids.device,
        )
        x[:, : prompt_ids.shape[1]] = prompt_ids.clone()

        if max_new_tokens % block_length != 0:
            raise ValueError("max_new_tokens must be divisible by block_length")
        num_blocks = max_new_tokens // block_length
        if steps % num_blocks != 0:
            raise ValueError("steps must be divisible by (max_new_tokens // block_length)")

        prompt_len = prompt_ids.shape[1]
        nfe = 0
        nfe += 1
        logits, past_key_values = self.sbd_inference_diffusion_quadratic(
            clean_input_ids=x[:, :prompt_len],
            draft_input_ids=x[:, prompt_len : prompt_len + block_length],
            block_length=block_length,
            draft_only=True,
            use_cache=True,
        )

        logits_proposal = logits[:, prompt_len - 1 : prompt_len + block_length]
        logits_proposal[:, 1] = logits_proposal[:, 0]
        logits_proposal = logits_proposal[:, 1:]
        x0_proposal = torch.argmax(logits_proposal, dim=-1)
        x[:, prompt_len : prompt_len + block_length] = x0_proposal

        total_accept_token = 0
        while True:
            nfe += 1
            block_start = prompt_len + total_accept_token
            block_end = block_start + block_length
            draft_input_ids = x[:, block_start:block_end]

            logits, past_key_values = self.sbd_inference_diffusion_quadratic(
                clean_input_ids=None,
                draft_input_ids=draft_input_ids,
                block_length=block_length,
                draft_only=False,
                past_key_values=past_key_values,
                use_cache=True,
            )

            useful_token_logits = logits.view(1, block_length, block_length + 1, -1)
            if ar_mix_weight is None:
                useful_token_logits[:, :, 1] = useful_token_logits[:, :, 0]
            else:
                if not (0.0 <= ar_mix_weight <= 1.0):
                    raise ValueError("ar_mix_weight must be between 0 and 1")
                mix_logits = useful_token_logits[:, :, 0] * ar_mix_weight + useful_token_logits[:, :, 1] * (1 - ar_mix_weight)
                useful_token_logits[:, :, 0] = mix_logits
                useful_token_logits[:, :, 1] = mix_logits

            if temperature > 0:
                useful_token_logits = useful_token_logits / temperature

            useful_token_pred = torch.argmax(useful_token_logits, dim=-1)
            new_draft_input_ids = useful_token_pred[:, 0, 1:]
            accept_cnt = 1

            while accept_cnt < block_length:
                if useful_token_pred[:, accept_cnt - 1, 0].item() != draft_input_ids[:, accept_cnt].item():
                    break
                new_draft_input_ids = useful_token_pred[:, accept_cnt, 1:]
                accept_cnt += 1

            x[:, block_start : block_start + accept_cnt] = draft_input_ids[:, :accept_cnt]

            # EoS early stopping: all accepted tokens are finalized left-to-right,
            # so if any is EoS we can truncate and return immediately.
            if eos_token_id is not None:
                accepted = x[0, block_start : block_start + accept_cnt]
                eos_positions = (accepted == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    first_eos_rel = eos_positions[0].item()
                    total_accept_token += first_eos_rel + 1
                    output_end = prompt_len + total_accept_token
                    return x[:, :output_end], nfe

            x[:, block_start + accept_cnt : block_start + accept_cnt + block_length] = new_draft_input_ids
            past_key_values.crop(block_start + accept_cnt)

            # ---- thinking budget enforcement ----
            # Insert end_think as the first token of the next draft block,
            # shifting all subsequent tokens right by 1 (discarding the last).
            # The first draft token is always accepted unconditionally, so
            # end_think is guaranteed to be finalized in the next iteration
            # without needing to re-encode or touch the KV cache.
            if end_think_token_id is not None and max_thinking_tokens is not None:
                tokens_so_far = total_accept_token + accept_cnt
                if tokens_so_far > max_thinking_tokens:
                    gen_so_far = x[0, prompt_len : prompt_len + tokens_so_far]
                    has_end_think = (gen_so_far == end_think_token_id).any()
                    if not has_end_think:
                        insert_pos = block_start + accept_cnt
                        x[0, insert_pos + 1:] = x[0, insert_pos:-1].clone()
                        x[0, insert_pos] = end_think_token_id

            total_accept_token += accept_cnt

            if total_accept_token >= max_new_tokens:
                break

        return x[:, : -(block_length * 2)], nfe


    @torch.no_grad()
    def linear_spec_generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        mask_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        max_thinking_tokens: Optional[int] = None,
        end_think_token_id: Optional[int] = None,
        threshold: float = 0.0,
    ):
        """Linear speculative decoding: diffusion draft + AR verification.

        Each step:
          1. Draft: forward [last_accepted, mask, ...] with bidirectional attention
             (diffusion_lm=True, use_cache=False).  Shift AR logits to get
             per-position predictions; apply confidence filtering.
          2. Verify: forward the drafted block with causal attention
             (diffusion_lm=False, use_cache=True, use_causal_mask=True).
             Accept consecutive AR-matching tokens plus one bonus token.

        Args:
            prompt_ids: Input token IDs of shape (1, prompt_len).
            max_new_tokens: Maximum number of tokens to generate.
            block_length: Number of tokens per draft/verify block.
            temperature: Sampling temperature (0 = greedy).
            mask_token_id: Override for config.mask_token_id.
            eos_token_id: Override for config.eos_token_id.
            max_thinking_tokens: Budget for thinking tokens before forcing end_think.
            end_think_token_id: Token ID inserted when thinking budget is exceeded.
            threshold: Confidence threshold for accepting draft predictions.

        Returns:
            (output_ids, nfe): output_ids includes the prompt; nfe is the number
            of forward evaluations (matching self_spec_generate interface).
        """
        if prompt_ids.shape[0] != 1:
            raise ValueError("Linear speculative decoding requires batch_size == 1")

        token_mask_id = mask_token_id if mask_token_id is not None else self.config.mask_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)

        device = prompt_ids.device
        prompt_len = prompt_ids.shape[1]
        dream_style = getattr(self.config, 'dlm_type', 'llada') == 'dream'

        def _set_diffusion_lm(val: bool):
            for layer in self.encoder.layers:
                if hasattr(layer.self_attn, 'diffusion_lm'):
                    layer.self_attn.diffusion_lm = val

        # ===== Prefill (causal) =====
        _set_diffusion_lm(False)

        enc_out = self.encoder(
            input_ids=prompt_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            use_causal_mask=True,
        )
        past_key_values = enc_out.past_key_values
        last_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)
        nfe = 1

        if temperature > 0:
            probs = torch.softmax(last_logit / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            output_ids = torch.cat([prompt_ids, next_token], dim=1)
            return output_ids, nfe

        generated = [next_token]
        total_gen = 1

        # ===== Main loop =====
        while total_gen < max_new_tokens:
            cache_len = past_key_values.get_seq_length()

            block = torch.full(
                (1, block_length), token_mask_id, dtype=torch.long, device=device
            )
            block[0, 0] = next_token.item()

            # -------- Draft (bidirectional, don't update cache) --------
            _set_diffusion_lm(True)
            enc_out = self.encoder(
                input_ids=block,
                past_key_values=past_key_values,
                use_cache=False,
            )
            nfe += 1

            draft_logits = self.diffusion_head(enc_out.last_hidden_state)
            if dream_style:
                # DREAM: logit[i] predicts position i+1 → shift to self-prediction
                draft_logits = torch.cat(
                    [draft_logits[:, :1, :], draft_logits[:, :-1, :]], dim=1
                )
            # LLaDA: logit[i] already predicts position i → no shift needed

            if temperature > 0:
                draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
                draft_tokens = torch.multinomial(
                    draft_probs.view(-1, draft_probs.shape[-1]), num_samples=1
                ).view(1, block_length)
            else:
                draft_tokens = draft_logits.argmax(dim=-1)
                draft_probs = torch.softmax(draft_logits, dim=-1)

            draft_conf = torch.gather(
                draft_probs, -1, draft_tokens.unsqueeze(-1)
            ).squeeze(-1)

            is_mask = block == token_mask_id
            draft_conf = torch.where(is_mask, draft_conf, -torch.inf)
            unmask = draft_conf > threshold

            if unmask.sum() > 0:
                block[unmask] = draft_tokens[unmask]
            else:
                raise AssertionError(
                    "No mask token above threshold for prediction"
                )

            # -------- Verify (causal, update cache) --------
            _set_diffusion_lm(False)
            enc_out = self.encoder(
                input_ids=block,
                past_key_values=past_key_values,
                use_cache=True,
                use_causal_mask=True,
            )
            past_key_values = enc_out.past_key_values
            nfe += 1

            verify_logits = self.diffusion_head(enc_out.last_hidden_state)
            if temperature > 0:
                verify_probs = torch.softmax(verify_logits / temperature, dim=-1)
                ar_tokens = torch.multinomial(
                    verify_probs.view(-1, verify_probs.shape[-1]), num_samples=1
                ).view(1, block_length)
            else:
                ar_tokens = verify_logits.argmax(dim=-1)

            accepted = 0
            for i in range(block_length - 1):
                if ar_tokens[0, i].item() == block[0, i + 1].item():
                    accepted += 1
                else:
                    break
            accepted += 1  # bonus token from AR verification

            accepted_toks = ar_tokens[:, :accepted]
            generated.append(accepted_toks)
            total_gen += accepted

            _crop_dynamic_cache(past_key_values, cache_len + accepted)

            next_token = ar_tokens[:, accepted - 1 : accepted]

            # -------- EOS check --------
            if eos_token_id is not None:
                eos_pos = (accepted_toks[0] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    first_eos = eos_pos[0].item()
                    generated[-1] = accepted_toks[:, : first_eos + 1]
                    total_gen = total_gen - accepted + first_eos + 1
                    break

            # -------- Thinking budget enforcement --------
            if end_think_token_id is not None and max_thinking_tokens is not None:
                if total_gen > max_thinking_tokens:
                    all_gen = torch.cat(generated, dim=1)
                    if not (all_gen == end_think_token_id).any():
                        next_token = torch.tensor(
                            [[end_think_token_id]], device=device
                        )

            if total_gen >= max_new_tokens:
                break

        all_generated = torch.cat(generated, dim=1)
        output_ids = torch.cat([prompt_ids, all_generated], dim=1)
        
        return output_ids, nfe
