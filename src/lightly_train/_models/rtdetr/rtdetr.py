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

from lightly_train._models.model_wrapper import (
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
)


class RTDETRModelWrapper(Module, ModelWrapper):
    def __init__(self, model: Module):
        super().__init__()
        self._model = [model]
        self._backbone = self._model[0].backbone
        self._pool = AdaptiveAvgPool2d((1, 1))

    def get_model(self) -> Module:
        return self._model[0]

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        features = self._backbone(x)[-1]
        return {"features": features}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def feature_dim(self) -> int:
        feat_dim = self._backbone.out_channels[-1]
        assert isinstance(feat_dim, int)
        return feat_dim
