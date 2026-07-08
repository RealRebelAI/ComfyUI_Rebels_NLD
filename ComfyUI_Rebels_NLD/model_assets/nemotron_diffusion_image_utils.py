# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch


def pad_along_last_dim(tensor, size):
    pad_size = size - tensor.shape[-1]
    if pad_size <= 0:
        return tensor
    padding = torch.zeros(*tensor.shape[:-1], pad_size, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=-1)


def maybe_truncate_last_dim(tensor, size):
    if size >= tensor.shape[-1]:
        return tensor
    return tensor[..., :size]
    return tensor[..., :size]
