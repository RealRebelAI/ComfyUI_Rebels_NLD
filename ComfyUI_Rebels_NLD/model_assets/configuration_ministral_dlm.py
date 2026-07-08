# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ministral DLM model configuration"""

from transformers.configuration_utils import PretrainedConfig
try:
    from transformers.modeling_rope_utils import rope_config_validation
except ImportError:
    rope_config_validation = None
from transformers.utils import logging


logger = logging.get_logger(__name__)


class MinistralDLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Ministral3Model`] for diffusion language models.
    It is used to instantiate a Ministral model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the Ministral model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 34):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key_value heads for Grouped Query Attention.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            The maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        rope_parameters (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
            Default uses YaRN scaling with factor=16, original_max_position_embeddings=16384.
        attention_bias (`bool`, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers.
        sliding_window (`int`, *optional*, defaults to None):
            Sliding window attention size.
        mask_token_id (`int`, *optional*, defaults to -1):
            Token ID for masking in diffusion.
        dlm_type (`str`, *optional*, defaults to 'llada'):
            Type of diffusion language model ('llada', 'dream').
        random_length_prob (`float`, *optional*):
            Probability of using random lengths during training.
        num_ar_layers (`int`, *optional*, defaults to 0):
            Number of autoregressive layers.
        num_diffusion_layers (`int`, *optional*, defaults to 0):
            Number of diffusion layers.
        diff_loss_weight (`float`, *optional*, defaults to 1):
            Weight for diffusion loss.
        enforce_mask (`bool`, *optional*, defaults to False):
            Whether to enforce masking.
        prefix_ratio (`float`, *optional*, defaults to 0.8):
            Ratio for prefix in prefix_bidirectional mode.
        dlm_paradigm (`str`, *optional*, defaults to 'bidirectional'):
            Paradigm for diffusion ('bidirectional', 'autoregressive', 'prefix_bidirectional', 'efficient_block_diff', 'block_diff', 'sbd_block_diff').
        dlm_arch (`str`, *optional*, defaults to 'encoder'):
            Architecture type ('encoder', 'encoder_decoder').
        block_size (`int`, *optional*, defaults to 32):
            Block size for block diffusion paradigms.
        tok_mask_half_life_ratio (`float`, *optional*):
            Half-life ratio for token masking.
        adaptive_mask_rate (`bool`, *optional*, defaults to False):
            Whether to use adaptive mask rate.
        multi_sampling (`int`, *optional*):
            Number of samples for multi-sampling.
        num_skip_loss_tokens (`int`, *optional*, defaults to 0):
            Number of tokens to skip in loss calculation.
        dlm_loss_weight (`float`, *optional*):
            Weight for diffusion LM loss.
        ar_loss_weight (`float`, *optional*, defaults to 1.0):
            Weight for autoregressive loss in sbd_block_diff paradigm. Use 10000 to only use AR loss.
        global_loss_avg (`bool`, *optional*, defaults to False):
            Whether to use global loss average.
        dp_varying_mask_ratio (`bool`, *optional*, defaults to False):
            Whether to use varying mask ratio for each DP rank during sampling.
        ada_perm_ratio_per_block (`float`, *optional*):
            Adaptive permutation ratio for each block.
        ada_perm_ratio_global (`float`, *optional*):
            Adaptive permutation ratio for global.
        enable_self_spec (`bool`, *optional*, defaults to `False`):
            Force MinistralFlexAttention for all paradigms (including bidirectional/autoregressive).
            Required for self speculative generation; leave False for standard eval to use faster SDPA kernels.
    """

    model_type = "ministral_dlm"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Ministral`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=34,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_parameters=None,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        sliding_window=None,
        attn_implementation="sdpa",
        mask_token_id=None,
        dlm_type='llada',
        random_length_prob=None,
        num_ar_layers=0,
        num_diffusion_layers=0,
        diff_loss_weight=1,
        enforce_mask=False,
        prefix_ratio=0.8,
        dlm_paradigm='bidirectional',
        dlm_arch='encoder',
        block_size=32,
        tok_mask_half_life_ratio=None,
        adaptive_mask_rate=False,
        multi_sampling=None,
        num_skip_loss_tokens=0,
        dlm_loss_weight=None,
        ar_loss_weight=1.0,
        global_loss_avg=False,
        dp_varying_mask_ratio=False,
        ada_perm_ratio_per_block=None,
        ada_perm_ratio_global=None,
        ada_dlm_loss_ratio=None,
        enable_self_spec=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        if rope_parameters is None and rope_scaling is not None:
            rope_parameters = dict(rope_scaling)
        # llama_4_scaling_beta is used directly by the attention layer; do not strip it.
        self.rope_parameters = rope_parameters
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.sliding_window = sliding_window
        
        self.attn_implementation = attn_implementation
        
        self.mask_token_id = mask_token_id
        self.dlm_type = dlm_type
        self.random_length_prob = random_length_prob
        self.num_ar_layers = num_ar_layers
        self.num_diffusion_layers = num_diffusion_layers
        self.diff_loss_weight = diff_loss_weight
        self.enforce_mask = enforce_mask
        self.prefix_ratio = prefix_ratio
        self.dlm_paradigm = dlm_paradigm
        self.dlm_arch = dlm_arch
        self.block_size = block_size
        self.tok_mask_half_life_ratio = tok_mask_half_life_ratio
        self.adaptive_mask_rate = adaptive_mask_rate
        self.multi_sampling = multi_sampling
        self.num_skip_loss_tokens = num_skip_loss_tokens
        self.dlm_loss_weight = dlm_loss_weight
        self.ar_loss_weight = ar_loss_weight
        self.global_loss_avg = global_loss_avg
        self.dp_varying_mask_ratio = dp_varying_mask_ratio
        self.ada_perm_ratio_per_block = ada_perm_ratio_per_block
        self.ada_perm_ratio_global = ada_perm_ratio_global
        self.ada_dlm_loss_ratio = ada_dlm_loss_ratio
        self.enable_self_spec = enable_self_spec
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Transformers>=4.57 expects standardized/validated rope_parameters.
        if hasattr(self, "standardize_rope_params"):
            self.standardize_rope_params()
        if hasattr(self, "validate_rope"):
            self.validate_rope()
        elif rope_config_validation is not None:
            rope_config_validation(self)


__all__ = ["MinistralDLMConfig"]

