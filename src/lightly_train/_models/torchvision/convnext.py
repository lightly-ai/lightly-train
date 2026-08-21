#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch import Tensor
from torchvision.models import ConvNeXt

from lightly_train._models.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
)
from lightly_train._models.torchvision.torchvision import TorchvisionModelWrapper

# Indices of the stage outputs in `model.features`. Each stage is preceded by a
# downsampling block, so the four stage outputs are at the odd indices 1, 3, 5, 7.
_STAGE_INDICES = (1, 3, 5, 7)


class ConvNeXtModelWrapper(TorchvisionModelWrapper, ArchitectureInfoGettable):
    _torchvision_models = [ConvNeXt]
    _torchvision_model_name_pattern = r"convnext.*"

    def __init__(self, model: ConvNeXt) -> None:
        super().__init__()
        self._model = [model]
        self._features = model.features
        self._pool = model.avgpool
        # Use linear layer from classifier to get feature dimension as last layer of
        # `model.features` is different depending on model configuration, making it hard
        # to get the feature dimension from there.
        self._feature_dim: int = model.classifier[-1].in_features

    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        return {"features": self._features(x)}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def get_model(self) -> ConvNeXt:
        return self._model[0]

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "convolutional", "norm_type": "layernorm"}

    def _extract_multiscale_stages(self, x: Tensor) -> list[Tensor]:
        stages: list[Tensor] = []
        out = x
        for index, module in enumerate(self._features):
            out = module(out)
            if index in _STAGE_INDICES:
                stages.append(out)
        return stages
