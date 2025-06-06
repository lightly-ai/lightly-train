#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import Tensor
from torchvision.models import ResNet
from torchvision.models._utils import IntermediateLayerGetter

from lightly_train._models.model_wrapper import (
    ForwardFeaturesOutput,
    ForwardPoolOutput,
)
from lightly_train._models.torchvision.torchvision import TorchvisionModelWrapper


class ResNetModelWrapper(TorchvisionModelWrapper):
    _torchvision_models = [ResNet]
    _torchvision_model_name_pattern = r"resnet.*"

    def __init__(self, model: ResNet) -> None:
        super().__init__()
        self._model = [model]
        self._features = IntermediateLayerGetter(
            model=model, return_layers={"layer4": "out"}
        )
        self._pool = model.avgpool
        self._feature_dim: int = model.fc.in_features

    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        return {"features": self._features(x)["out"]}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def get_model(self) -> ResNet:
        return self._model[0]
