#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import Tensor
from torchvision.models import ConvNeXt

from lightly_train._models.model_wrapper import (
    ForwardFeaturesOutput,
    ForwardPoolOutput,
)
from lightly_train._models.torchvision.torchvision import TorchvisionModelWrapper


class ConvNeXtModelWrapper(TorchvisionModelWrapper):
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
