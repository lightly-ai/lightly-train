#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch import Tensor
from torch.nn import Module

from lightly_train._models.fastvit.fastvit import FastViTModelWrapper


class FastViTBackboneWrapper(Module):
    """Expose FastViT's stride-8/16/32 stages to LT-DETR."""

    def __init__(self, model_wrapper: FastViTModelWrapper) -> None:
        super().__init__()
        self._model_wrapper = model_wrapper

    @property
    def backbone_model(self) -> Module:
        return self._model_wrapper.get_model()

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        features = self._model_wrapper.forward_multiscale_features(x, [1, 2, 3])
        return tuple(feature["features"] for feature in features)
