#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig
from lightly_train._configs.model_registry import ModelRegistry


class EoMTInstanceSegmentationConfig(PydanticConfig):
    backbone_name: str


DINOV2_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY: ModelRegistry[
    EoMTInstanceSegmentationConfig
] = ModelRegistry()


class DINOv2EoMTInstanceSegmentationConfigRegistry(ConfigsNamespace):
    @DINOV2_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register("dinov2/_vittest14-eomt")
    class ViTTest(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov2/_vittest14"

    @DINOV2_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register("dinov2/vits14-eomt")
    class ViTSmall(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov2/vits14"

    @DINOV2_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register("dinov2/vitb14-eomt")
    class ViTBase(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov2/vitb14"

    @DINOV2_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register("dinov2/vitl14-eomt")
    class ViTLarge(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov2/vitl14"

    @DINOV2_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register("dinov2/vitg14-eomt")
    class ViTGiant(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov2/vitg14"

    @DINOV2_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov2/vits14-notpretrained-eomt"
    )
    class ViTSmallNotPretrained(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov2/vits14-notpretrained"

    @DINOV2_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov2/vitb14-notpretrained-eomt"
    )
    class ViTBaseNotPretrained(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov2/vitb14-notpretrained"

    @DINOV2_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov2/vitl14-notpretrained-eomt"
    )
    class ViTLargeNotPretrained(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov2/vitl14-notpretrained"

    @DINOV2_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov2/vitg14-notpretrained-eomt"
    )
    class ViTGiantNotPretrained(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov2/vitg14-notpretrained"
