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


class FasterViTModelWrapper(Module, ModelWrapper, ArchitectureInfoGettable):
    def __init__(self, model: Module) -> None:
        if not hasattr(model, "forward_features"):
            raise ValueError("Model must have a 'forward_features' method")
        if not hasattr(model, "avgpool"):
            raise ValueError("Model must have an 'avgpool' attribute")
        super().__init__()
        self._model = model
        self._pool: Module = model.avgpool  # type: ignore[assignment]

    def feature_dim(self) -> int:
        return _get_feature_dim(self._model)

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        features: Tensor = self._model.forward_features(x)  # type: ignore[operator]
        return {"features": features}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        features = self._pool(x["features"])
        while len(features.shape) < 4:
            features = features.unsqueeze(-1)
        return {"pooled_features": features}

    def get_model(self) -> Module:
        return self._model

    def architecture_info(self) -> ArchitectureInfo:
        norm = getattr(self._model, "norm", None)
        norm_type: str = "batchnorm" if isinstance(norm, nn.BatchNorm2d) else "layernorm"
        return {"model_type": "hybrid", "norm_type": norm_type}  # type: ignore[return-value]


def _get_feature_dim(model: Module) -> int:
    """Return the feature dimension of a FasterViT model."""
    head = getattr(model, "head", None)
    if head is not None and hasattr(head, "in_features"):
        return int(head.in_features)
    # Fallback: read from norm layer (BatchNorm2d has num_features, LayerNorm has normalized_shape)
    norm = getattr(model, "norm", None)
    if norm is not None:
        if hasattr(norm, "num_features"):
            return int(norm.num_features)
        if hasattr(norm, "normalized_shape"):
            return int(norm.normalized_shape[0])
    raise ValueError(
        "Cannot determine feature_dim for FasterViT model: "
        "model has no 'head.in_features' or recognizable 'norm' attribute."
    )
