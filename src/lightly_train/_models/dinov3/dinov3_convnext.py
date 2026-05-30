#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, Conv2d, Module

from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt
from lightly_train._models.model_wrapper import (
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
    MultiScaleFeatureCNN,
)


class DINOv3VConvNeXtModelWrapper(Module, ModelWrapper, MultiScaleFeatureCNN):
    def __init__(self, model: ConvNeXt) -> None:
        super().__init__()
        self._model = model
        self._feature_dim = int(self._model.embed_dim)
        self._pool = AdaptiveAvgPool2d((1, 1))

    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(
        self, x: Tensor, masks: Tensor | None = None, n_blocks: int = 1
    ) -> ForwardFeaturesOutput:
        if n_blocks > 1:
            # ConvNeXt blocks may produce feature maps at different spatial resolutions
            # (e.g. 14×14 and 7×7). Interpolate all to the last block's resolution.
            x_list = list(
                self._model.get_intermediate_layers(
                    x, n=n_blocks, reshape=True, return_class_token=True
                )
            )
            h_last, w_last = x_list[-1][0].shape[-2:]
            x_locals, x_globals = [], []
            for x_local, x_global in x_list:
                if x_local.shape[-2:] != (h_last, w_last):
                    x_local = F.interpolate(
                        x_local,
                        size=(h_last, w_last),
                        mode="bilinear",
                        align_corners=False,
                    )
                x_locals.append(x_local)
                x_globals.append(x_global)
            return {
                "features": torch.cat(x_locals, dim=1),  # (B, n*D, H, W)
                "cls_token": torch.cat(x_globals, dim=1),  # (B, n*D)
            }

        rt = self._model(x, masks, is_training=True)  # forcing to return all patches
        if rt["x_norm_patchtokens"].dim() == 3:
            x_norm_patchtokens = rt["x_norm_patchtokens"]
            b = x_norm_patchtokens.shape[0]
            d = x_norm_patchtokens.shape[2]
            h, w = rt["x_norm_patchtokens_hw"]

            features_reshaped = x_norm_patchtokens.permute(0, 2, 1).reshape(b, d, h, w)
        else:
            raise ValueError(
                f"Unexpected shape for x_norm_patchtokens: {rt['x_norm_patchtokens'].shape}"
            )
        return {"features": features_reshaped, "cls_token": rt["x_norm_clstoken"]}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def get_model(self) -> ConvNeXt:
        return self._model

    def make_teacher(self) -> None:
        pass

    def multiscale_feature_strides(self) -> list[int]:
        strides = []
        cumulative = 1
        for stage in self._model.downsample_layers:
            for mod in stage:
                if isinstance(mod, Conv2d):
                    cumulative *= mod.stride[0]
            strides.append(cumulative)
        return strides

    def multiscale_feature_dims(self) -> list[int]:
        return list(self._model.embed_dims)

    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Sequence[int]
    ) -> list[ForwardFeaturesOutput]:
        rt = self._model.get_intermediate_layers(
            x, n=list(layer_indices), reshape=True, return_class_token=True
        )
        return [{"features": feat, "cls_token": cls} for feat, cls in rt]
