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
from torch.nn import Conv2d, Module
from torchvision.models import ShuffleNetV2

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

# Names of the stages in `model`, from earliest (0) to last (3). conv5 has the same
# stride as stage4 and only expands the feature dimension, but it is the last stage of
# the backbone and is therefore included.
_STAGE_NAMES = ("stage2", "stage3", "stage4", "conv5")


class ShuffleNetV2ModelWrapper(
    TorchvisionModelWrapper, ArchitectureInfoGettable, MultiScaleFeatureCNN
):
    _torchvision_models = [ShuffleNetV2]
    _torchvision_model_name_pattern = r"shufflenet_v2.*"

    def __init__(self, model: ShuffleNetV2):
        super().__init__()
        self._model = model

    def get_model(self) -> ShuffleNetV2:
        return self._model

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        x = self._forward_stem(x)
        for stage_name in _STAGE_NAMES:
            x = getattr(self._model, stage_name)(x)
        return {"features": x}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": x["features"].mean([2, 3], keepdim=True)}

    def feature_dim(self) -> int:
        feature_dim: int = self._model.fc.in_features
        return feature_dim

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "convolutional", "norm_type": "batchnorm"}

    def multiscale_feature_dims(self) -> list[int]:
        return [self._stage_out_channels(stage_name) for stage_name in _STAGE_NAMES]

    def multiscale_feature_strides(self) -> list[int]:
        model = self._model
        maxpool_stride = model.maxpool.stride
        stride = _max_conv_stride(model.conv1) * (
            maxpool_stride[0] if isinstance(maxpool_stride, tuple) else maxpool_stride
        )
        strides = []
        for stage_name in _STAGE_NAMES:
            stride *= _max_conv_stride(self._first_block(stage_name))
            strides.append(stride)
        return strides

    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Sequence[int]
    ) -> list[ForwardFeaturesOutput]:
        _validate_layer_indices(
            layer_indices=layer_indices, num_layers=len(_STAGE_NAMES)
        )
        x = self._forward_stem(x)
        stages = []
        for stage_name in _STAGE_NAMES:
            x = getattr(self._model, stage_name)(x)
            stages.append(x)
        return [{"features": stages[index]} for index in layer_indices]

    def _forward_stem(self, x: Tensor) -> Tensor:
        x = self._model.conv1(x)
        x = self._model.maxpool(x)
        return x

    def _first_block(self, stage_name: str) -> Module:
        """Returns the block of the stage that defines its stride and dimension.

        The stages are sequences of blocks of which only the first one downsamples and
        changes the feature dimension. conv5 is a single block.
        """
        stage: Module = getattr(self._model, stage_name)
        return stage if stage_name == "conv5" else stage[0]  # type: ignore[index]

    def _stage_out_channels(self, stage_name: str) -> int:
        """Returns the feature dimension of the stage.

        The downsampling block of a stage concatenates the outputs of its two branches,
        so the stage's feature dimension is the sum over the branches. conv5 has a
        single branch.
        """
        block = self._first_block(stage_name)
        return sum(
            _last_conv_out_channels(branch)
            for _, branch in block.named_children()
            if any(isinstance(module, Conv2d) for module in branch.modules())
        )
