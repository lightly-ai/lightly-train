#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torch import Tensor
from torch.nn import Module

from lightly_train._models.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
)


class RadioModelWrapper(Module, ModelWrapper, ArchitectureInfoGettable):
    """Adapter for NVIDIA C-RADIO models."""

    _input_mean: Tensor
    _input_std: Tensor

    def __init__(self, model: Module) -> None:
        super().__init__()
        self._model = model
        self.register_buffer(
            "_input_mean",
            torch.tensor(IMAGENET_NORMALIZE["mean"]).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_input_std",
            torch.tensor(IMAGENET_NORMALIZE["std"]).view(1, -1, 1, 1),
            persistent=False,
        )
        summary_dim = getattr(model, "summary_dim", None)
        if summary_dim is None:
            summary_dim = getattr(model, "embed_dim")
        self._feature_dim = int(summary_dim)

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

        # LightlyTrain's default transforms use ImageNet normalization. NVIDIA's
        # built-in RADIO input conditioner expects values in the [0, 1] range.
        x = x * self._input_std.to(dtype=x.dtype) + self._input_mean.to(dtype=x.dtype)

        output = self._model(x, feature_fmt="NCHW")
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(
                "RADIO returned an unexpected output. Adaptors and custom necks are "
                "not supported by the LightlyTrain RADIO wrapper."
            )
        cls_token, features = output
        if cls_token.ndim != 2:
            raise RuntimeError(
                "C-RADIO returned summary features with unexpected shape "
                f"{tuple(cls_token.shape)}. Expected (B, C)."
            )
        if features.ndim != 4:
            raise RuntimeError(
                "C-RADIO returned spatial features with unexpected shape "
                f"{tuple(features.shape)}. Expected NCHW features."
            )
        return {"features": features, "cls_token": cls_token}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": x["cls_token"][..., None, None]}

    def get_model(self) -> Module:
        return self._model

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "transformer", "norm_type": "layernorm"}
