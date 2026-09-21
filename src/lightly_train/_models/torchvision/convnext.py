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
from torchvision.models import ConvNeXt

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


class ConvNeXtModelWrapper(
    TorchvisionModelWrapper, ArchitectureInfoGettable, MultiScaleFeatureCNN
):
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

    def multiscale_feature_dims(self) -> list[int]:
        # The downsampling blocks set the feature dimension of the stage that follows
        # them and the blocks within a stage keep it.
        return [
            _last_conv_out_channels(self._features[index])
            for index in self._downsample_indices()
        ]

    def multiscale_feature_strides(self) -> list[int]:
        strides = []
        stride = 1
        for index in self._downsample_indices():
            stride *= _max_conv_stride(self._features[index])
            strides.append(stride)
        return strides

    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Sequence[int]
    ) -> list[ForwardFeaturesOutput]:
        stage_indices = self._stage_indices()
        _validate_layer_indices(
            layer_indices=layer_indices, num_layers=len(stage_indices)
        )
        stages = []
        for index, module in enumerate(self._features):
            x = module(x)
            if index in stage_indices:
                stages.append(x)
        return [{"features": stages[index]} for index in layer_indices]

    def _stage_indices(self) -> range:
        """Indices of the stages in `model.features`.

        `model.features` alternates between a downsampling block and a stage, starting
        with the stem, so the stages are at the odd indices.
        """
        return range(1, len(self._features), 2)

    def _downsample_indices(self) -> range:
        """Indices of the stem and the downsampling blocks in `model.features`."""
        return range(0, len(self._features), 2)
