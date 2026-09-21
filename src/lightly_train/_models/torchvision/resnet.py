#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Sequence

from torch import Tensor
from torchvision.models import ResNet
from torchvision.models._utils import IntermediateLayerGetter

from lightly_train._models.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    MultiScaleFeatureCNN,
)
from lightly_train._models.torchvision.torchvision import (
    TorchvisionModelWrapper,
    _last_conv_out_channels,
    _max_conv_stride,
    _validate_layer_indices,
)

# Output of every residual stage. The dict values are the stage indices used by the
# multi-scale feature interface, from earliest (0) to last (3).
_RETURN_LAYERS = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}


class ResNetModelWrapper(
    TorchvisionModelWrapper, ArchitectureInfoGettable, MultiScaleFeatureCNN
):
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
        return {"features": self._features(x)[_RETURN_LAYERS["layer4"]]}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def get_model(self) -> ResNet:
        return self._model[0]

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "convolutional", "norm_type": "batchnorm"}

    def multiscale_feature_dims(self) -> list[int]:
        model = self.get_model()
        # The last block of a stage has no downsampling branch, so its last Conv2d
        # outputs the stage's feature dimension. This holds for both the BasicBlock of
        # resnet18/34 and the Bottleneck of resnet50 and larger.
        return [
            _last_conv_out_channels(getattr(model, layer_name)[-1])
            for layer_name in _RETURN_LAYERS
        ]

    def multiscale_feature_strides(self) -> list[int]:
        model = self.get_model()
        maxpool_stride = model.maxpool.stride
        stride = model.conv1.stride[0] * (
            maxpool_stride[0] if isinstance(maxpool_stride, tuple) else maxpool_stride
        )
        strides = []
        for layer_name in _RETURN_LAYERS:
            # Only the first block of a stage downsamples. Stages built with
            # replace_stride_with_dilation do not downsample at all, in which case the
            # stride stays the same as for the previous stage.
            stride *= _max_conv_stride(getattr(model, layer_name)[0])
            strides.append(stride)
        return strides

    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Sequence[int]
    ) -> list[ForwardFeaturesOutput]:
        _validate_layer_indices(
            layer_indices=layer_indices, num_layers=len(_RETURN_LAYERS)
        )
        features = self._features(x)
        stages = [features[index] for index in _RETURN_LAYERS.values()]
        return [{"features": stages[index]} for index in layer_indices]
