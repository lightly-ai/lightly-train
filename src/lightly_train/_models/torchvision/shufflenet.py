#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch import Tensor
from torch.nn import Module
from torchvision.models import ShuffleNetV2

from lightly_train._models.model_wrapper import ForwardFeaturesOutput, ForwardPoolOutput
from lightly_train._models.torchvision.torchvision import TorchvisionModelWrapper


class ShuffleNetV2ModelWrapper(TorchvisionModelWrapper):
    _torchvision_models = [ShuffleNetV2]
    _torchvision_model_name_pattern = r"shufflenet_v2.*"

    def __init__(self, model: Module):
        super().__init__()
        self._model = model

    def get_model(self) -> Module:
        return self._model

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        x = self._model.conv1(x)
        x = self._model.maxpool(x)
        x = self._model.stage2(x)
        x = self._model.stage3(x)
        x = self._model.stage4(x)
        x = self._model.conv5(x)
        return {"features": x}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": x["features"].mean([2, 3], keepdim=True)}

    def feature_dim(self) -> int:
        feature_dim: int = self._model.fc.in_features
        return feature_dim
