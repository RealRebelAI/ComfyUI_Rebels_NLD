# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import torch
import torch.nn.functional as F

from transformers.utils import ModelOutput
from dataclasses import dataclass
from transformers.cache_utils import Cache, DynamicCache
@dataclass
class SimpleOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    causal_logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None

from .nemotron_diffusion_image_utils import maybe_truncate_last_dim, pad_along_last_dim


def wte(model,x,t2i_inference=False,gen_shape=None,x_gen=None,inputs_embeds_curr=None,new_token_mask=None):

    if t2i_inference:
        assert x_gen is not None
        if new_token_mask is None:
            new_token_mask = x >= INT_MAX
        # if x_gen is  None:
        #     x_gen = x[new_token_mask] - OFFSET
        # else:
        #     x_gen = x_gen - OFFSET

        gen_latents_comp_embeds = model.call_gen_embedding(x_gen,gen_shape)
        if inputs_embeds_curr is None:
            x_txt_only = x.clone()
            
            # replace consequtent [1] * 4096 to [1] * 1024

            x_txt_only[new_token_mask] = 0
            inputs_embeds_curr = model.embed_tokens(x_txt_only)
        inputs_embeds_curr[new_token_mask] = pad_along_last_dim(gen_latents_comp_embeds,inputs_embeds_curr.shape[-1]).view(-1,inputs_embeds_curr.shape[-1])
    else:
        inputs_embeds_curr = model.embed_tokens(x)
        new_token_mask = None
    return inputs_embeds_curr,new_token_mask
    

INT_MAX = 1_000_000
def get_logits(model,input_emnbeddings,modality_indices=None,t2i_inference=False,past_key_values=None,gen_shape=None,timesteps=None,input_modality_indices=None):
    if t2i_inference:
        if input_modality_indices is None:
            input_modality_indices =modality_indices
        output = model(None,input_embeddings=input_emnbeddings,modality_indices=input_modality_indices,output_hidden_states=True,past_key_values=past_key_values,
                        is_training=False,
                        overwrite_attn_impl='flash_attn'
        )
        hidden_states = output.hidden_states[-1]
        gen_hidden_states = hidden_states[modality_indices]
        gen_hidden_states = maybe_truncate_last_dim(gen_hidden_states,model.config.d_model_gen)
        gen_logits = model.call_gen_predictor(gen_hidden_states,gen_shape,timesteps=timesteps) # * 8 D
        seq_len_per_img = np.prod(gen_shape)
        if len(gen_logits.shape) == 2:
            gen_logits = gen_logits.view(-1,seq_len_per_img,gen_logits.shape[-1])
        else:
            gen_logits = gen_logits.view(-1,seq_len_per_img,*gen_logits.shape[-2:])
            # N L 8 D
        return gen_logits
        
        
        final_logits = torch.zeros(*gen_logits.shape[:-1],OFFSET+gen_logits.shape[-1],dtype=output.logits.dtype,device=output.logits.device)
        final_logits[:] = float('-inf')
        final_logits[...,OFFSET:] = gen_logits
        # breakpoint()
        # inal_logits = torch.zeros(*hidden_states.shape[:-1],OFFSET+gen_logits.shape[-1],dtype=output.logits.dtype,device=output.logits.device)
        
        # final_logits = final_logits + float('-inf')
        # final_logits[...,:output.logits.shape[-1]] = output.logits
        # final_logits[modality_indices] = float('-inf')
        # local = final_logits[modality_indices]

        # local[...,OFFSET:] = gen_logits
        # final_logits[modality_indices] = local

        logits = final_logits
        return logits
    else:
        modality_indices = torch.zeros(input_emnbeddings.shape[:-1],device=input_emnbeddings.device,dtype=torch.bool)
        logits = model(None,input_embeddings=input_emnbeddings,modality_indices=modality_indices,past_key_values=past_key_values).logits
    return logits

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None, neg_entropy=False):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        # p = F.softmax(logits.to(torch.float64), dim=-1)
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'top_p_margin':
        # Compute probabilities
        p = F.softmax(logits, dim=-1)                       # (B, L, V)
        # Top-2 per position
        top2 = torch.topk(p, k=2, dim=-1).values            # (B, L, 2)
        margin = top2[..., 0] - top2[..., 1]                # (B, L)

        # Normalize margin to [0,1] over MASKED positions per row
        plus_inf  = torch.full_like(margin, float('inf'))
        minus_inf = torch.full_like(margin, float('-inf'))
        masked_for_min = torch.where(mask_index, margin, plus_inf)
        masked_for_max = torch.where(mask_index, margin, minus_inf)
        row_min = masked_for_min.amin(dim=1, keepdim=True)  # (B, 1)
        row_max = masked_for_max.amax(dim=1, keepdim=True)  # (B, 1)
        denom = (row_max - row_min)

        # If denom==0 (all equal), set normalized=1 on masked; 0 elsewhere by default
        normalized = torch.zeros_like(margin)
        nonzero = denom > 0
        normalized = torch.where(
            mask_index & nonzero,
            (margin - row_min) / (denom + 1e-12),
            normalized
        )
        normalized = torch.where(
            mask_index & (~nonzero),
            torch.ones_like(normalized),
            normalized
        )
        x0_p = normalized  # ∈ [0,1] on masked positions
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    # Calculate negative entropy if requested
    if neg_entropy:
        # p = F.softmax(logits.to(torch.float64), dim=-1)
        p = F.softmax(logits, dim=-1)
        epsilon = 1e-10
        log_probs = torch.log(p + epsilon)
        confidence_scores = torch.sum(p * log_probs, dim=-1)  # negative entropy per position
    else:
        confidence_scores = x0_p
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence_scores, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    # print(f'confidence: {confidence}')
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_num_transfer_tokens(mask_index, steps: int):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : int(remainder[i])] += 1
    return num_transfer_tokens

def simple_fwd(model,input_ids=None,inputs_embeds=None,attention_mask=None,position_ids=None,past_key_values=None,**kwargs):
    enc_out  = model.encoder(
        past_key_values=past_key_values,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        is_training=False,
        overwrite_attn_impl='flash_attn',
        # overwrite_attn_impl='flash_attn',
        # overwrite_block_mask='full',
        **kwargs,
    )
    logits = model.diffusion_head(enc_out.last_hidden_state)

    return SimpleOutputWithPast(
        loss=logits,
        logits=logits,
        causal_logits=None,
        past_key_values=enc_out.past_key_values,
        hidden_states=None,
        attentions=None,
    )


@torch.no_grad()
def generate_with_prefix_cache_block_diff(
    model,
    prompt=None,
    prompt_embeds=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    threshold=None,
    factor=None,
    shift_logits=False,
    neg_entropy=False,
    causal_context=False,
    eos_token_id=None,
    max_thinking_tokens=None,
    end_think_token_id=None,
):
    dream_style=shift_logits
    if (prompt is None) == (prompt_embeds is None):
        raise ValueError("Exactly one of `prompt` or `prompt_embeds` must be provided.")

    if prompt is not None:
        prompt_ids = prompt
        prompt_len = prompt_ids.shape[1]
        x_accum = prompt_ids.clone()
        B = prompt_ids.shape[0]
        token_device = prompt_ids.device
        token_dtype = prompt_ids.dtype
    else:
        prompt_ids = None
        prompt_len = prompt_embeds.shape[1]
        B = prompt_embeds.shape[0]
        token_device = prompt_embeds.device
        token_dtype = torch.long
        # Keep prefix slots so block slicing by prompt_len stays identical.
        x_accum = torch.full((B, prompt_len), mask_id, dtype=token_dtype, device=token_device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    model_module = model.module if hasattr(model, "module") else model
    for layer in model_module.encoder.layers:
        layer.self_attn.mode = 'bidirectional'

    if causal_context:
        for layer in model_module.encoder.layers:
            if hasattr(layer.self_attn, 'diffusion_lm'):
                layer.self_attn.diffusion_lm=False

    # Compute KV cache for the prompt initially
    output = simple_fwd(model,
        input_ids=prompt_ids,
        inputs_embeds=prompt_embeds,
        use_cache=True,
        use_causal_mask=causal_context,
    )
    past_key_values = output.past_key_values

    if causal_context:
        for layer in model_module.encoder.layers:
            if hasattr(layer.self_attn, 'diffusion_lm'):
                layer.self_attn.diffusion_lm=True

    # Causal prefill: next token from last position (same as linear_spec_generate).
    next_token = None
    if causal_context:
        last_logit = output.logits[:, -1, :]
        if temperature > 0:
            probs = torch.softmax(last_logit / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

    # For dream_style: store the "next token logit" of the context
    next_logits_context = None
    if dream_style:
        next_logits_context = output.logits[:, -1:, :]  # (B, 1, V)

    for num_block in range(num_blocks):
        # Create a new block with mask tokens; under causal context, seed position 0
        # with the next-token prediction from the previous causal forward (prefill or
        # post-block encode), matching linear_spec_generate.
        mask_block = torch.ones(
            (B, block_length),
            dtype=token_dtype,
            device=token_device,
        ) * mask_id
        if causal_context:
            mask_block[:, 0] = next_token[:, 0]

        # Append the block of masks
        x_accum = torch.cat([x_accum, mask_block], dim=1)
        current_block_start = prompt_len + num_block * block_length
        block_slice = slice(current_block_start, current_block_start + block_length)

        # ---- thinking budget enforcement ----
        # If we've generated >= max_thinking_tokens without a </think>, inject one.
        if end_think_token_id is not None and max_thinking_tokens is not None:
            tokens_before_block = num_block * block_length
            tokens_after_block = tokens_before_block + block_length
            if tokens_after_block > max_thinking_tokens:
                gen_so_far = x_accum[:, prompt_len:current_block_start]
                has_end_think = (
                    (gen_so_far == end_think_token_id).any(dim=1)
                    if gen_so_far.size(1) > 0
                    else torch.zeros(B, dtype=torch.bool, device=token_device)
                )
                if not has_end_think.all():
                    if tokens_before_block < max_thinking_tokens:
                        offset = max_thinking_tokens - tokens_before_block
                    else:
                        offset = 0
                    inject_pos = current_block_start + offset
                    for b in range(B):
                        if not has_end_think[b]:
                            x_accum[b, inject_pos] = end_think_token_id

        # Build the initial mask for this block
        mask_block_idx0 = (x_accum[:, block_slice] == mask_id)  # (B, Lb)

        # Precompute the transfer schedule for this block
        if dream_style:
            # masked positions only (position 0 may be causal-seeded, not mask_id)
            schedule_mask = mask_block_idx0
        else:
            schedule_mask = mask_block_idx0

        num_transfer_tokens = get_num_transfer_tokens(schedule_mask, steps_per_block)  # (B, steps)

        # Denoise the current block
        for i in range(steps_per_block):
            mask_block_idx = (x_accum[:, block_slice] == mask_id)  # (B, Lb)
            if mask_block_idx.sum() == 0:
                break

            nfe += 1

            # Forward only the current noisy block using cached context
            logits_block = simple_fwd(model,
                x_accum[:, block_slice],
                past_key_values=past_key_values,
                use_cache=False
            ).logits

            if dream_style:
                # Align logits so that each masked position has a predictor:
                # prepend context-next logit, then use logits_block[:-1]
                if block_length == 1:
                    logits_use = next_logits_context              # (B, 1, V)
                else:
                    logits_use = torch.cat(
                        [next_logits_context, logits_block[:, :-1, :]],
                        dim=1
                    )  # (B, Lb, V)

                mask_use = mask_block_idx                        # (B, Lb)
                x_use   = x_accum[:, block_slice]                # (B, Lb)

                x0, transfer_idx = get_transfer_index(
                    logits_use, temperature, remasking, mask_use, x_use,
                    num_transfer_tokens=num_transfer_tokens[:, i],
                    threshold=threshold, neg_entropy=neg_entropy
                )
                cur = x_accum[:, block_slice].clone()
                cur[transfer_idx] = x0[transfer_idx]
                x_accum[:, block_slice] = cur

            else:
                # non-AR (same-position) case
                x0, transfer_idx = get_transfer_index(
                    logits_block, temperature, remasking, mask_block_idx,
                    x_accum[:, block_slice],
                    num_transfer_tokens=num_transfer_tokens[:, i],
                    threshold=threshold, neg_entropy=neg_entropy
                )
                cur = x_accum[:, block_slice].clone()
                cur[transfer_idx] = x0[transfer_idx]
                x_accum[:, block_slice] = cur

            if eos_token_id is not None:
                block_tokens = x_accum[:, block_slice]              # (B, Lb)
                eos_mask = (block_tokens == eos_token_id)           # (B, Lb)
                any_eos = eos_mask.any(dim=1)                       # (B,)
                if any_eos.any():
                    after_eos = eos_mask.cumsum(dim=1).bool()       # (B, Lb)
                    mask_before = (block_tokens == mask_id) & ~after_eos
                    if (any_eos & ~mask_before.any(dim=1)).any():
                        break

        if causal_context:
            for layer in model_module.encoder.layers:
                if hasattr(layer.self_attn, 'diffusion_lm'):
                    layer.self_attn.diffusion_lm=False

        # after block is fully denoised, update KV cache
        output = simple_fwd(model,
            x_accum[:, block_slice],
            past_key_values=past_key_values,
            use_cache=True,
            use_causal_mask=causal_context
        )
        past_key_values = output.past_key_values
        nfe += 1

        if causal_context:
            for layer in model_module.encoder.layers:
                if hasattr(layer.self_attn, 'diffusion_lm'):
                    layer.self_attn.diffusion_lm=True
            # Next block's first position = greedy/sampled next token from this causal encode
            last_logit = output.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(last_logit / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        if dream_style and num_block < num_blocks - 1:
            # refresh context-next logit for the next block
            next_logits_context = output.logits[:, -1:, :]  # (B, 1, V)

        if eos_token_id is not None:
            gen_so_far = x_accum[:, prompt_len:]                        # (B, gen_len_so_far)
            is_eos = (gen_so_far == eos_token_id)                       # (B, gen_len_so_far)
            has_eos = is_eos.any(dim=1)                                 # (B,)
            if has_eos.all():
                first_eos_pos = is_eos.to(torch.int64).argmax(dim=1)    # (B,)
                max_eos = first_eos_pos.max().item()
                if prompt_ids is None:
                    return x_accum[:, prompt_len : prompt_len + max_eos + 1], nfe
                return x_accum[:, : prompt_len + max_eos + 1], nfe

    if prompt_ids is None:
        return x_accum[:, prompt_len:], nfe
    return x_accum, nfe
