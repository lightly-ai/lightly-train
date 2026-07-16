#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, Module

from lightly_train._models.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
)


class RadioModelWrapper(Module, ModelWrapper, ArchitectureInfoGettable):
    """Adapter for NVIDIA RADIO Torch Hub models."""

    def __init__(self, model: Module) -> None:
        super().__init__()
        self._model = model
        self._feature_dim = int(getattr(model, "embed_dim"))
        self._pool = AdaptiveAvgPool2d(output_size=1)

    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        min_resolution_step = int(getattr(self._model, "min_resolution_step"))
        height, width = x.shape[-2:]
        if height % min_resolution_step != 0 or width % min_resolution_step != 0:
            raise ValueError(
                "RADIO input dimensions must be multiples of "
                f"min_resolution_step={min_resolution_step}, got "
                f"({height}, {width}). Set a compatible transform image size."
            )

        output = self._model(x, feature_fmt="NCHW")
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "RADIO returned an unexpected output. Adaptors and custom necks are "
                "not supported by the LightlyTrain RADIO wrapper."
            )
        _, features = output
        if features.ndim != 4:
            raise RuntimeError(
                "RADIO returned spatial features with unexpected shape "
                f"{tuple(features.shape)}. Expected NCHW features."
            )
        return {"features": features}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def get_model(self) -> Module:
        return self._model

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "transformer", "norm_type": "layernorm"}
