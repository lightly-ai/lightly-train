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
# limitations under the License.#

"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.

# Modifications Copyright 2025 Lightly AG:
- Added typing to functions.
- Updated the imports.
- Asserts the number of returned layers is 3.
- Remove printing.
- Added comments and fixed typing issues.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    GELU,
    Conv2d,
    MaxPool2d,
    Module,
    ModuleList,
    Sequential,
    SyncBatchNorm,
)

from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer,
)


class SpatialPriorModulev2(Module):
    def __init__(self, inplanes: int = 16) -> None:
        super().__init__()

        # 1/4
        self.stem = Sequential(
            *[
                Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                SyncBatchNorm(inplanes),
                GELU(),
                MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = Sequential(
            *[
                Conv2d(
                    inplanes,
                    2 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = Sequential(
            *[
                GELU(),
                Conv2d(
                    2 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = Sequential(
            *[
                GELU(),
                Conv2d(
                    4 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        c1 = self.stem(x)
        c2 = self.conv2(c1)  # 1/8
        c3 = self.conv3(c2)  # 1/16
        c4 = self.conv4(c3)  # 1/32

        return c2, c3, c4


class DINOv3STAs(Module):
    def __init__(
        self,
        model: DinoVisionTransformer,
        interaction_indexes: list[int] = [5, 8, 11],
        finetune: bool = True,
        patch_size: int = 16,
        use_sta: bool = True,
        conv_inplane: int = 16,
        hidden_dim: int | None = None,
    ):
        super(DINOv3STAs, self).__init__()

        self.dinov3 = model
        embed_dim = self.dinov3.embed_dim

        assert len(interaction_indexes) == 3
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # init the feature pyramid
        self.use_sta = use_sta
        if use_sta:
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = ModuleList(
            [
                Conv2d(
                    embed_dim + conv_inplane * 2,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                Conv2d(
                    embed_dim + conv_inplane * 4,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                Conv2d(
                    embed_dim + conv_inplane * 4,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
            ]
        )
        # norm
        self.norms = ModuleList(
            [
                SyncBatchNorm(hidden_dim),
                SyncBatchNorm(hidden_dim),
                SyncBatchNorm(hidden_dim),
            ]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Code for matching with oss
        H_c, W_c = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, _, _, _ = x.shape

        if len(self.interaction_indexes) > 0:
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            # With the assert in the __init__ this branch is never used.
            all_layers = self.dinov3(x)

        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = (
                feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
            )  # [B, D, H, W]
            resize_H, resize_W = (
                int(H_c * 2 ** (num_scales - i)),
                int(W_c * 2 ** (num_scales - i)),
            )
            sem_feat = F.interpolate(
                sem_feat,
                size=[resize_H, resize_W],
                mode="bilinear",
                align_corners=False,
            )
            sem_feats.append(sem_feat)

        # Normalize sem feats type to tensors.
        # If feat is a Tensor it is the spatial tokens
        # If feat is a Tuple, the first entry contains the spatial tokens.
        # With the default args from get_intermediate_layers it is a Tensor.
        sem_feats_t: list[torch.Tensor] = [
            feat if isinstance(feat, torch.Tensor) else feat[0] for feat in sem_feats
        ]

        # fusion
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(sem_feats_t, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats_t

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4
