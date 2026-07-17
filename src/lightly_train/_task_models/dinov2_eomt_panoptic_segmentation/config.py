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


class EoMTPanopticSegmentationConfig(PydanticConfig):
    backbone_name: str
    num_joint_blocks: int


DINOV2_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY: ModelRegistry[
    EoMTPanopticSegmentationConfig
] = ModelRegistry()


class DINOv2EoMTPanopticSegmentationConfigRegistry(ConfigsNamespace):
    @DINOV2_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov2/_vittest14-eomt")
    class ViTTest(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov2/_vittest14"
        num_joint_blocks: int = 3

    @DINOV2_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov2/vits14-eomt")
    class ViTSmall(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov2/vits14"
        num_joint_blocks: int = 3

    @DINOV2_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov2/vitb14-eomt")
    class ViTBase(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov2/vitb14"
        num_joint_blocks: int = 3

    @DINOV2_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov2/vitl14-eomt")
    class ViTLarge(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov2/vitl14"
        num_joint_blocks: int = 4

    @DINOV2_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov2/vitg14-eomt")
    class ViTGiant(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov2/vitg14"
        num_joint_blocks: int = 5

    @DINOV2_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov2/vits14-notpretrained-eomt"
    )
    class ViTSmallNotPretrained(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov2/vits14-notpretrained"
        num_joint_blocks: int = 3

    @DINOV2_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov2/vitb14-notpretrained-eomt"
    )
    class ViTBaseNotPretrained(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov2/vitb14-notpretrained"
        num_joint_blocks: int = 3

    @DINOV2_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov2/vitl14-notpretrained-eomt"
    )
    class ViTLargeNotPretrained(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov2/vitl14-notpretrained"
        num_joint_blocks: int = 4

    @DINOV2_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov2/vitg14-notpretrained-eomt"
    )
    class ViTGiantNotPretrained(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov2/vitg14-notpretrained"
        num_joint_blocks: int = 5
