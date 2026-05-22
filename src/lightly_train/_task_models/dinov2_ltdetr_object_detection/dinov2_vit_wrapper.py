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

import logging
from typing import Any, Mapping

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

from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper

logger = logging.getLogger(__name__)


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


class DINOv2STAs(Module):
    def __init__(
        self,
        model_wrapper: DINOv2ViTModelWrapper,
        interaction_indexes: list[int] = [5, 8, 11],
        finetune: bool = True,
        use_sta: bool = True,
        conv_inplane: int = 16,
        hidden_dim: int | None = None,
    ):
        super().__init__()

        self._model_wrapper = model_wrapper
        embed_dim = self._model_wrapper.feature_dim()

        assert len(interaction_indexes) == 3
        self.interaction_indexes = interaction_indexes
        self.patch_size = model_wrapper._model.patch_size

        if not finetune:
            model_wrapper.eval()
            model_wrapper.requires_grad_(False)

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

    @property
    def backbone_model(self) -> Module:
        return self._model_wrapper._model

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError:
            old_prefix = "dinov2."
            new_prefix = "_model_wrapper._model."
            if any(k.startswith(old_prefix) for k in state_dict):
                logger.info(
                    "Detected old DINOv2STAs checkpoint format "
                    "(dinov2. → _model_wrapper._model.). Remapping keys."
                )
                remapped = {}
                for k, v in state_dict.items():
                    if k.startswith(old_prefix):
                        k = new_prefix + k[len(old_prefix) :]
                    remapped[k] = v
                try:
                    return super().load_state_dict(
                        remapped, strict=strict, assign=assign
                    )
                except RuntimeError:
                    raise
            raise

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        H_c, W_c = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size

        sem_feats = self._model_wrapper.forward_multiscale_features(
            x, self.interaction_indexes
        )

        resized_feats: list[Tensor] = []
        num_scales = len(sem_feats)
        for i, sem_feat in enumerate(sem_feats):
            feat = sem_feat["features"]
            resize_H, resize_W = (
                int(H_c * 2 ** (num_scales - i)),
                int(W_c * 2 ** (num_scales - i)),
            )
            resized_feats.append(
                F.interpolate(
                    feat,
                    size=[resize_H, resize_W],
                    mode="bilinear",
                    align_corners=False,
                )
            )

        # fusion
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for semantic_feat, detail_feat in zip(resized_feats, detail_feats):
                detail_feat_interpolated = F.interpolate(
                    detail_feat,
                    size=semantic_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                fused_feats.append(
                    torch.cat(
                        [
                            semantic_feat,
                            detail_feat_interpolated,
                        ],
                        dim=1,
                    )
                )
        else:
            fused_feats = resized_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4
