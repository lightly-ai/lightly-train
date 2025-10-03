#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, Module

from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt
from lightly_train._models.model_wrapper import (
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
)


class DINOv3VConvNeXtModelWrapper(Module, ModelWrapper):
    def __init__(self, model: ConvNeXt) -> None:
        super().__init__()
        self._model = model
        self._feature_dim = int(self._model.embed_dim)
        self._pool = AdaptiveAvgPool2d((1, 1))

    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(
        self, x: Tensor, masks: Tensor | None = None
    ) -> ForwardFeaturesOutput:
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
