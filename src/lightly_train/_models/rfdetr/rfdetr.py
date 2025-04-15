#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging

from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, Module

from lightly_train._models.feature_extractor import (
    FeatureExtractor,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
)

logger = logging.getLogger(__name__)


class RFDETRFeatureExtractor(Module, FeatureExtractor):
    def __init__(self, model: Module) -> None:
        super().__init__()
        from rfdetr.models.backbone import Backbone
        from rfdetr.models.backbone.dinov2 import DinoV2
        from rfdetr.models.lwdetr import LWDETR

        assert isinstance(model, LWDETR)

        backbone = model.backbone[0]
        assert isinstance(backbone, Backbone)

        encoder = backbone.encoder
        assert isinstance(encoder, DinoV2)

        feature_dim = encoder._out_feature_channels[-1]
        assert isinstance(feature_dim, int)

        self._model = model
        # Set model to training mode. This is necessary for RFDETR pretrained
        # models as the DINOv2 backbone is in eval mode by default.
        self._model.train()

        self._encoder = encoder
        self._feature_dim = feature_dim
        self._pool = AdaptiveAvgPool2d((1, 1))

    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        """We use the last output of different stages of the DINOv2 encoder as the output feature."""
        return {"features": self._encoder(x)[-1]}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}
