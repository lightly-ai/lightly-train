#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch import Tensor
from torchvision.models import ResNet
from torchvision.models._utils import IntermediateLayerGetter

from lightly_train._models.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
)
from lightly_train._models.torchvision.torchvision import TorchvisionModelWrapper

# Output of every residual stage. The dict values are the stage indices used by the
# multi-scale feature interface, from earliest (0) to last (3).
_RETURN_LAYERS = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}


class ResNetModelWrapper(TorchvisionModelWrapper, ArchitectureInfoGettable):
    _torchvision_models = [ResNet]
    _torchvision_model_name_pattern = r"resnet.*"

    def __init__(self, model: ResNet) -> None:
        super().__init__()
        self._model = [model]
        self._features = IntermediateLayerGetter(
            model=model, return_layers=dict(_RETURN_LAYERS)
        )
        self._pool = model.avgpool
        self._feature_dim: int = model.fc.in_features

    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        return {"features": self._features(x)["3"]}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def get_model(self) -> ResNet:
        return self._model[0]

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "convolutional", "norm_type": "batchnorm"}

    def _extract_multiscale_stages(self, x: Tensor) -> list[Tensor]:
        features = self._features(x)
        return [features[index] for index in _RETURN_LAYERS.values()]
