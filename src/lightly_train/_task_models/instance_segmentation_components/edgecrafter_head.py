#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RF-DETR (https://github.com/roboflow/rf-detr)
Copyright (c) 2025 Roboflow. All Rights Reserved.
Licensed under the Apache License, Version 2.0 [see LICENSE for details]
"""

# Modifications Copyright 2026 Lightly AG:
# - Added typed interfaces.
# - Renamed the head to EdgeCrafterInstanceSegmentationHead.
# - Removed exporting.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DepthwiseConvBlock(nn.Module):
    """Simplified ConvNeXt block with depthwise convolution and residual output."""

    def __init__(self, dim: int, layer_scale_init_value: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return x + identity


class MLPBlock(nn.Module):
    """Residual MLP block used to refine decoder query features."""

    def __init__(self, dim: int, layer_scale_init_value: float = 0.0) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            ]
        )
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        if self.gamma is not None:
            x = self.gamma * x
        return x + identity


class EdgeCrafterInstanceSegmentationHead(nn.Module):
    """EdgeCrafter mask head for LTDETR instance segmentation.

    The head refines a high-resolution spatial feature cumulatively, projects
    spatial and query features into a shared interaction dimension, and computes
    mask logits as spatial-query dot products. It returns one mask-logit tensor
    per consumed decoder layer.
    """

    def __init__(
        self,
        in_dim: int,
        num_blocks: int,
        bottleneck_ratio: int | None = 1,
        downsample_ratio: int = 4,
        image_size: tuple[int, int] = (640, 640),
        layer_scale_init_value: float = 0.0,
    ) -> None:
        """Initializes the EdgeCrafter instance segmentation head.

        Args:
            in_dim: Channel dimension of the spatial and query features.
            num_blocks: Number of cumulative spatial refinement blocks.
            bottleneck_ratio: Optional ratio for reducing the interaction
                dimension. Set to ``None`` to keep the full feature dimension.
            downsample_ratio: Output mask stride relative to ``image_size``.
            image_size: Fixed image size used to derive the mask-logit size.
            layer_scale_init_value: Initial value for optional layer-scale
                parameters in the residual blocks. Disabled when set to ``0``.
        """
        super().__init__()

        self.downsample_ratio = downsample_ratio
        self.image_size = image_size
        self.interaction_dim = (
            in_dim // bottleneck_ratio if bottleneck_ratio is not None else in_dim
        )
        self.blocks = nn.ModuleList(
            [
                DepthwiseConvBlock(
                    dim=in_dim,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for _ in range(num_blocks)
            ]
        )
        self.spatial_features_proj = (
            nn.Identity()
            if bottleneck_ratio is None
            else nn.Conv2d(in_dim, self.interaction_dim, kernel_size=1)
        )
        self.query_features_block = MLPBlock(
            dim=in_dim,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.query_features_proj = (
            nn.Identity()
            if bottleneck_ratio is None
            else nn.Linear(in_dim, self.interaction_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        spatial_features: Tensor,
        query_features: list[Tensor],
    ) -> list[Tensor]:
        """Computes per-layer mask logits.

        Args:
            spatial_features: Spatial feature tensor of shape ``(B, C, H, W)``.
            query_features: Per-layer decoder query states with shape
                ``(B, Q, C)``.

        Returns:
            A list of mask-logit tensors with shape ``(B, Q, Hm, Wm)``, where
            ``Hm`` and ``Wm`` are derived from ``image_size // downsample_ratio``.
        """
        target_size = (
            self.image_size[0] // self.downsample_ratio,
            self.image_size[1] // self.downsample_ratio,
        )
        spatial_features = F.interpolate(
            spatial_features,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        mask_logits: list[Tensor] = []
        for block, query_feature in zip(self.blocks, query_features):
            spatial_features = block(spatial_features)
            spatial_features_proj = self.spatial_features_proj(spatial_features)
            query_feature = self.query_features_proj(
                self.query_features_block(query_feature)
            )
            mask_logits.append(
                torch.einsum("bchw,bqc->bqhw", spatial_features_proj, query_feature)
                + self.bias
            )
        return mask_logits
