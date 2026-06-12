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
    backbone_name: str = ""  # full "package/backbone" string, e.g. "dinov2/vits14"
    backbone_args: dict[str, Any] = Field(default_factory=dict)


LINEAR_SEG_MODEL_REGISTRY: ModelRegistry[LinearSemanticSegmentationConfig] = ModelRegistry()


class LinearSegConfigRegistry(ConfigsNamespace):
    # --- DINOv2 ---
    @LINEAR_SEG_MODEL_REGISTRY.register(
        "dinov2/vits14-linear",
        "dinov2/vits14-notpretrained-linear",
    )
    class DINOv2ViTS14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vits14"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )

    @LINEAR_SEG_MODEL_REGISTRY.register(
        "dinov2/vitb14-linear",
        "dinov2/vitb14-notpretrained-linear",
    )
    class DINOv2ViTB14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitb14"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )

    @LINEAR_SEG_MODEL_REGISTRY.register(
        "dinov2/vitl14-linear",
        "dinov2/vitl14-notpretrained-linear",
    )
    class DINOv2ViTL14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitl14"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )

    @LINEAR_SEG_MODEL_REGISTRY.register(
        "dinov2/vitg14-linear",
        "dinov2/vitg14-notpretrained-linear",
    )
    class DINOv2ViTG14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitg14"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )

    # --- DINOv3 ViT variants ---
    @LINEAR_SEG_MODEL_REGISTRY.register(
        "dinov3/vitt16-linear",
        "dinov3/vitt16-notpretrained-linear",
    )
    class DINOv3ViTT16(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16"

    @LINEAR_SEG_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-linear",
        "dinov3/vitt16plus-notpretrained-linear",
    )
    class DINOv3ViTT16Plus(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitt16-distillationv1-linear")
    class DINOv3ViTT16Distillationv1(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-distillationv1"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitt16plus-distillationv1-linear")
    class DINOv3ViTT16PlusDistillationv1(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus-distillationv1"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vits16-linear")
    class DINOv3ViTS16(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vits16"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vits16plus-linear")
    class DINOv3ViTS16Plus(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vits16plus"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitb16-linear")
    class DINOv3ViTB16(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitb16"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitl16-linear")
    class DINOv3ViTL16(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitl16"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vith16plus-linear")
    class DINOv3ViTH16Plus(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vith16plus"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vit7b16-linear")
    class DINOv3ViT7B16(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vit7b16"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitl16-sat493m-linear")
    class DINOv3ViTL16Sat493m(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitl16-sat493m"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vit7b16-sat493m-linear")
    class DINOv3ViT7B16Sat493m(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vit7b16-sat493m"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitt16-eupe-linear")
    class DINOv3ViTT16Eupe(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-eupe"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vits16-eupe-linear")
    class DINOv3ViTS16Eupe(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vits16-eupe"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitb16-eupe-linear")
    class DINOv3ViTB16Eupe(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitb16-eupe"

    # --- DINOv3 ConvNeXt variants ---
    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-tiny-linear")
    class DINOv3ConvNextTiny(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-tiny"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-small-linear")
    class DINOv3ConvNextSmall(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-small"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-base-linear")
    class DINOv3ConvNextBase(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-base"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-large-linear")
    class DINOv3ConvNextLarge(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-large"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-tiny-eupe-linear")
    class DINOv3ConvNextTinyEupe(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-tiny-eupe"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-small-eupe-linear")
    class DINOv3ConvNextSmallEupe(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-small-eupe"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-base-eupe-linear")
    class DINOv3ConvNextBaseEupe(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-base-eupe"

    class Fallback(LinearSemanticSegmentationConfig):
        pass  # backbone_name stays "", parsed from model_name at runtime
