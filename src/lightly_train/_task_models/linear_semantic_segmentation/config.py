#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator
from typing_extensions import Self

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig
from lightly_train._configs.model_registry import ModelRegistry


class LinearSemanticSegmentationConfig(PydanticConfig):
    backbone_name: str = ""  # full "package/backbone" string, e.g. "dinov2/vits14"
    backbone_args: dict[str, Any] = Field(default_factory=dict)
    freeze_mask_token: bool = False

    @model_validator(mode="after")
    def _check_freeze_mask_token(self) -> Self:
        if self.freeze_mask_token and not self.backbone_name.startswith("dinov2/"):
            raise ValueError(
                f"freeze_mask_token=True is only supported for DINOv2 backbones "
                f"(backbone_name must start with 'dinov2/'), got '{self.backbone_name}'."
            )
        if self.backbone_name.startswith("dinov2/") and not self.freeze_mask_token:
            raise ValueError(
                f"DINOv2 backbones require freeze_mask_token=True, got freeze_mask_token={self.freeze_mask_token}."
            )
        return self


LINEAR_SEG_MODEL_REGISTRY: ModelRegistry[LinearSemanticSegmentationConfig] = (
    ModelRegistry()
)


class LinearSegConfigRegistry(ConfigsNamespace):
    # --- DINOv2 (pretrained) ---
    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vits14-linear")
    class DINOv2ViTS14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vits14"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vitb14-linear")
    class DINOv2ViTB14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitb14"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vitl14-linear")
    class DINOv2ViTL14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitl14"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vitg14-linear")
    class DINOv2ViTG14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitg14"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    # --- TIPSv2 (pretrained vision backbones) ---
    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vitb14-tipsv2-linear")
    class TIPSv2ViTB14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitb14-tipsv2"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vitl14-tipsv2-linear")
    class TIPSv2ViTL14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitl14-tipsv2"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vitso400m14-tipsv2-linear")
    class TIPSv2ViTSO400M14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitso400m14-tipsv2"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vitg14-tipsv2-linear")
    class TIPSv2ViTG14(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitg14-tipsv2"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    # --- DINOv2 (not pretrained) ---
    # Map to the "-notpretrained" backbones so no pretrained weights are loaded.
    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vits14-notpretrained-linear")
    class DINOv2ViTS14NotPretrained(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vits14-notpretrained"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vitb14-notpretrained-linear")
    class DINOv2ViTB14NotPretrained(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitb14-notpretrained"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vitl14-notpretrained-linear")
    class DINOv2ViTL14NotPretrained(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitl14-notpretrained"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov2/vitg14-notpretrained-linear")
    class DINOv2ViTG14NotPretrained(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov2/vitg14-notpretrained"
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"drop_path_rate": 0.0}
        )
        freeze_mask_token: bool = True

    # --- DINOv3 ViT variants ---
    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitt16-linear")
    class DINOv3ViTT16(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitt16-notpretrained-linear")
    class DINOv3ViTT16NotPretrained(DINOv3ViTT16):
        backbone_name: str = "dinov3/vitt16-notpretrained"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitt16plus-linear")
    class DINOv3ViTT16Plus(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitt16plus-notpretrained-linear")
    class DINOv3ViTT16PlusNotPretrained(DINOv3ViTT16Plus):
        backbone_name: str = "dinov3/vitt16plus-notpretrained"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitt16-distillationv1-linear")
    class DINOv3ViTT16Distillationv1(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-distillationv1"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitt16plus-distillationv1-linear")
    class DINOv3ViTT16PlusDistillationv1(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus-distillationv1"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vits16-linear")
    class DINOv3ViTS16(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vits16"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vits16-notpretrained-linear")
    class DINOv3ViTS16NotPretrained(DINOv3ViTS16):
        backbone_name: str = "dinov3/vits16-notpretrained"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vits16plus-linear")
    class DINOv3ViTS16Plus(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vits16plus"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitb16-linear")
    class DINOv3ViTB16(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitb16"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitb16-notpretrained-linear")
    class DINOv3ViTB16NotPretrained(DINOv3ViTB16):
        backbone_name: str = "dinov3/vitb16-notpretrained"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitl16-linear")
    class DINOv3ViTL16(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitl16"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitl16-notpretrained-linear")
    class DINOv3ViTL16NotPretrained(DINOv3ViTL16):
        backbone_name: str = "dinov3/vitl16-notpretrained"

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

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vits16-lingbot-linear")
    class DINOv3ViTS16Lingbot(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vits16-lingbot"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitb16-lingbot-linear")
    class DINOv3ViTB16Lingbot(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitb16-lingbot"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/vitl16-lingbot-linear")
    class DINOv3ViTL16Lingbot(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitl16-lingbot"

    # --- DINOv3 ConvNeXt variants ---
    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-tiny-linear")
    class DINOv3ConvNextTiny(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-tiny"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-tiny-notpretrained-linear")
    class DINOv3ConvNextTinyNotPretrained(DINOv3ConvNextTiny):
        backbone_name: str = "dinov3/convnext-tiny-notpretrained"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-small-linear")
    class DINOv3ConvNextSmall(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-small"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-small-notpretrained-linear")
    class DINOv3ConvNextSmallNotPretrained(DINOv3ConvNextSmall):
        backbone_name: str = "dinov3/convnext-small-notpretrained"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-base-linear")
    class DINOv3ConvNextBase(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-base"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-base-notpretrained-linear")
    class DINOv3ConvNextBaseNotPretrained(DINOv3ConvNextBase):
        backbone_name: str = "dinov3/convnext-base-notpretrained"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-large-linear")
    class DINOv3ConvNextLarge(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-large"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-large-notpretrained-linear")
    class DINOv3ConvNextLargeNotPretrained(DINOv3ConvNextLarge):
        backbone_name: str = "dinov3/convnext-large-notpretrained"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-tiny-eupe-linear")
    class DINOv3ConvNextTinyEupe(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-tiny-eupe"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-small-eupe-linear")
    class DINOv3ConvNextSmallEupe(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-small-eupe"

    @LINEAR_SEG_MODEL_REGISTRY.register("dinov3/convnext-base-eupe-linear")
    class DINOv3ConvNextBaseEupe(LinearSemanticSegmentationConfig):
        backbone_name: str = "dinov3/convnext-base-eupe"

    # --- Torchvision (CNN backbones) ---
    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/convnext_tiny-linear")
    class TorchvisionConvNeXtTiny(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/convnext_tiny"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/convnext_small-linear")
    class TorchvisionConvNeXtSmall(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/convnext_small"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/convnext_base-linear")
    class TorchvisionConvNeXtBase(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/convnext_base"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/convnext_large-linear")
    class TorchvisionConvNeXtLarge(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/convnext_large"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/resnet18-linear")
    class TorchvisionResNet18(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/resnet18"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/resnet34-linear")
    class TorchvisionResNet34(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/resnet34"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/resnet50-linear")
    class TorchvisionResNet50(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/resnet50"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/resnet101-linear")
    class TorchvisionResNet101(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/resnet101"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/resnet152-linear")
    class TorchvisionResNet152(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/resnet152"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/shufflenet_v2_x0_5-linear")
    class TorchvisionShuffleNetV2X05(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/shufflenet_v2_x0_5"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/shufflenet_v2_x1_0-linear")
    class TorchvisionShuffleNetV2X10(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/shufflenet_v2_x1_0"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/shufflenet_v2_x1_5-linear")
    class TorchvisionShuffleNetV2X15(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/shufflenet_v2_x1_5"

    @LINEAR_SEG_MODEL_REGISTRY.register("torchvision/shufflenet_v2_x2_0-linear")
    class TorchvisionShuffleNetV2X20(LinearSemanticSegmentationConfig):
        backbone_name: str = "torchvision/shufflenet_v2_x2_0"

    class Fallback(LinearSemanticSegmentationConfig):
        pass  # backbone_name stays "", parsed from model_name at runtime
