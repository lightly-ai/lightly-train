#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from lightly_train._models.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
)

logger = logging.getLogger(__name__)


class FastViTModelWrapper(Module, ModelWrapper, ArchitectureInfoGettable):
    def __init__(self, model: Module) -> None:
        for attr in ("forward_embeddings", "forward_tokens", "conv_exp", "gap"):
            if not hasattr(model, attr):
                raise ValueError(f"Model must have a '{attr}' attribute")
        super().__init__()
        self._model = model

    def feature_dim(self) -> int:
        return _get_feature_dim(self._model)

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        x = self._model.forward_embeddings(x)  # type: ignore[operator]
        x = self._model.forward_tokens(x)  # type: ignore[operator]
        x = self._model.conv_exp(x)  # type: ignore[operator]
        return {"features": x}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        features: Tensor = self._model.gap(x["features"])  # type: ignore[operator]
        return {"pooled_features": features}

    def get_model(self) -> Module:
        return self._model

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "hybrid", "norm_type": "batchnorm"}  # type: ignore[return-value]


def _get_feature_dim(model: Module) -> int:
    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        return int(head.in_features)
    raise ValueError(
        "Cannot determine feature_dim for FastViT model: "
        "model has no 'head' nn.Linear attribute."
    )
