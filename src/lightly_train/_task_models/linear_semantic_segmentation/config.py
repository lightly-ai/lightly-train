#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Any

from pydantic import Field

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig
from lightly_train._configs.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class LinearSemanticSegmentationConfig(PydanticConfig):
    backbone_args: dict[str, Any] = Field(default_factory=dict)


LINEAR_SEG_MODEL_REGISTRY: ModelRegistry[LinearSemanticSegmentationConfig] = ModelRegistry()


class LinearSegConfigRegistry(ConfigsNamespace):
    @LINEAR_SEG_MODEL_REGISTRY.register(
        "dinov2/vits14-linear",
        "dinov2/vitb14-linear",
        "dinov2/vitl14-linear",
        "dinov2/vitg14-linear",
        "dinov2/vits14-notpretrained-linear",
        "dinov2/vitb14-notpretrained-linear",
        "dinov2/vitl14-notpretrained-linear",
        "dinov2/vitg14-notpretrained-linear",
    )
    class DINOv2(LinearSemanticSegmentationConfig):
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )

    @LINEAR_SEG_MODEL_REGISTRY.register(
        # ViT variants
        "dinov3/vitt16-linear",
        "dinov3/vitt16plus-linear",
        "dinov3/vitt16-distillationv1-linear",
        "dinov3/vitt16plus-distillationv1-linear",
        "dinov3/vits16-linear",
        "dinov3/vits16plus-linear",
        "dinov3/vitb16-linear",
        "dinov3/vitl16-linear",
        "dinov3/vith16plus-linear",
        "dinov3/vit7b16-linear",
        "dinov3/vitl16-sat493m-linear",
        "dinov3/vit7b16-sat493m-linear",
        "dinov3/vitt16-eupe-linear",
        "dinov3/vits16-eupe-linear",
        "dinov3/vitb16-eupe-linear",
        "dinov3/vitt16-notpretrained-linear",
        "dinov3/vitt16plus-notpretrained-linear",
        # ConvNeXt variants
        "dinov3/convnext-tiny-linear",
        "dinov3/convnext-small-linear",
        "dinov3/convnext-base-linear",
        "dinov3/convnext-large-linear",
        "dinov3/convnext-tiny-eupe-linear",
        "dinov3/convnext-small-eupe-linear",
        "dinov3/convnext-base-eupe-linear",
    )
    class DINOv3(LinearSemanticSegmentationConfig):
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    class Fallback(LinearSemanticSegmentationConfig):
        backbone_args: dict[str, Any] = Field(default_factory=dict)
