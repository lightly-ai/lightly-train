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
    ForwardFeaturesOutput,
    ForwardPoolOutput,
)
from lightly_train._models.super_gradients.super_gradients import (
    SuperGradientsModelWrapper,
)

try:
    from super_gradients.training.models import (
        PPLiteSegBase,
    )
except ImportError:
    PPLiteSegBase = None


class SegmentationModuleModelWrapper(Module, SuperGradientsModelWrapper):
    _SUPPORTED_MODEL_CLASSES = (
        (PPLiteSegBase,) if PPLiteSegBase is not None else tuple()
    )

    def __init__(self, model: Module) -> None:
        super().__init__()
        self._model = [model]
        self._backbone = model.backbone
        self._pool = AdaptiveAvgPool2d((1, 1))

    @classmethod
    def is_supported_model_cls(cls, model_cls: type[Module]) -> bool:
        return issubclass(model_cls, cls._SUPPORTED_MODEL_CLASSES)

    @classmethod
    def supported_model_classes(cls) -> tuple[type[Module], ...]:
        return cls._SUPPORTED_MODEL_CLASSES

    def feature_dim(self) -> int:
        out_channels: list[int] | int = (
            self._backbone.get_backbone_output_number_of_channels()
        )
        return out_channels[-1] if isinstance(out_channels, list) else out_channels

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        features: list[Tensor | None] | Tensor = self._backbone(x)
        if isinstance(features, Tensor):
            features = [features]

        # Some models can return None outputs.
        feats = [f for f in features if f is not None][-1]
        return {"features": feats}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def get_model(self) -> Module:
        return self._model[0]
