#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig
from lightly_train._configs.model_registry import (
    DownloadableCheckpoint,
    ModelAlias,
    ModelRegistry,
)


class EoMTPanopticSegmentationConfig(PydanticConfig):
    backbone_name: str
    num_joint_blocks: int


DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY: ModelRegistry[
    EoMTPanopticSegmentationConfig
] = ModelRegistry()

_DINOV3_VITT16_PANOPTIC_COCO_URL = "dinov3_vitt16_eomt_panoptic_coco_260113_770c0a1f.pt"
_DINOV3_VITT16_PANOPTIC_COCO_SHA256 = (
    "770c0a1f024b9a78a6669d44968e2ab15b6d812839ce0c28732889ec5370ceea"
)
_DINOV3_VITT16PLUS_PANOPTIC_COCO_URL = (
    "dinov3_vitt16plus_eomt_panoptic_coco_260113_25765911.pt"
)
_DINOV3_VITT16PLUS_PANOPTIC_COCO_SHA256 = (
    "25765911e4ebc6d735f385e8350a1c9924b4ccf08657d3868fbaa95ff4cc64e9"
)
_DINOV3_VITS16_PANOPTIC_COCO_URL = "dinov3_vits16_eomt_panoptic_coco_251219_89e8a64f.pt"
_DINOV3_VITS16_PANOPTIC_COCO_SHA256 = (
    "89e8a64fb601c509df76d09ed6ddb6789e080147cadcff9700cf5792dfc20167"
)
_DINOV3_VITB16_PANOPTIC_COCO_URL = "dinov3_vitb16_eomt_panoptic_coco_251209_05948298.pt"
_DINOV3_VITB16_PANOPTIC_COCO_SHA256 = (
    "0594829822a23935079c35304f3bd1c7fede802114bc1a699780df693f2dea6c"
)
_DINOV3_VITL16_PANOPTIC_COCO_URL = "dinov3_vitl16_eomt_panoptic_coco_251209_e0c1e6ae.pt"
_DINOV3_VITL16_PANOPTIC_COCO_SHA256 = (
    "e0c1e6aeb245dbe6fd8735ffea48b81978b66b1a320533498de4375c18ad4368"
)
_DINOV3_VITL16_PANOPTIC_COCO_1280_URL = (
    "dinov3_vitl16_eomt_panoptic_coco_1280_251209_3da0b210.pt"
)
_DINOV3_VITL16_PANOPTIC_COCO_1280_SHA256 = (
    "3da0b21000bba3747bcb3e4ac4ee1e38641614022281f4b710d7442c643182f2"
)


class DINOv3EoMTPanopticSegmentationConfigRegistry(ConfigsNamespace):
    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/_vittest16-eomt")
    class ViTTest(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/_vittest16"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitt16-eomt-panoptic-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITT16_PANOPTIC_COCO_URL,
                sha256=_DINOV3_VITT16_PANOPTIC_COCO_SHA256,
            ),
        ),
        "dinov3/vitt16-eomt",
    )
    class ViTTiny(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitt16plus-eomt-panoptic-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITT16PLUS_PANOPTIC_COCO_URL,
                sha256=_DINOV3_VITT16PLUS_PANOPTIC_COCO_SHA256,
            ),
        ),
        "dinov3/vitt16plus-eomt",
    )
    class ViTTinyPlus(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16-notpretrained-eomt"
    )
    class ViTTinyNotPretrained(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-notpretrained"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-notpretrained-eomt"
    )
    class ViTTinyPlusNotPretrained(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus-notpretrained"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16-distillationv1-eomt"
    )
    class ViTTinyDistillationV1(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-distillationv1"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-distillationv1-eomt"
    )
    class ViTTinyPlusDistillationV1(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus-distillationv1"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vits16-eomt-panoptic-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITS16_PANOPTIC_COCO_URL,
                sha256=_DINOV3_VITS16_PANOPTIC_COCO_SHA256,
            ),
        ),
        "dinov3/vits16-eomt",
    )
    class ViTSmall(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vits16"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vits16plus-eomt")
    class ViTSmallPlus(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vits16plus"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitb16-eomt-panoptic-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITB16_PANOPTIC_COCO_URL,
                sha256=_DINOV3_VITB16_PANOPTIC_COCO_SHA256,
            ),
        ),
        "dinov3/vitb16-eomt",
    )
    class ViTBase(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitb16"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitl16-eomt-panoptic-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITL16_PANOPTIC_COCO_URL,
                sha256=_DINOV3_VITL16_PANOPTIC_COCO_SHA256,
            ),
        ),
        ModelAlias(
            name="dinov3/vitl16-eomt-panoptic-coco-1280",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITL16_PANOPTIC_COCO_1280_URL,
                sha256=_DINOV3_VITL16_PANOPTIC_COCO_1280_SHA256,
            ),
        ),
        "dinov3/vitl16-eomt",
    )
    class ViTLarge(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitl16"
        num_joint_blocks: int = 4

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vith16plus-eomt")
    class ViTHugePlus(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vith16plus"
        num_joint_blocks: int = 5

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vit7b16-eomt")
    class ViT7B(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vit7b16"
        num_joint_blocks: int = 5

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitl16-sat493m-eomt"
    )
    class ViTLargeSAT493M(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitl16-sat493m"
        num_joint_blocks: int = 4

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vit7b16-sat493m-eomt"
    )
    class ViT7BSAT493M(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vit7b16-sat493m"
        num_joint_blocks: int = 5

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16-eupe-eomt"
    )
    class ViTTinyEUPE(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-eupe"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vits16-eupe-eomt"
    )
    class ViTSmallEUPE(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vits16-eupe"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitb16-eupe-eomt"
    )
    class ViTBaseEUPE(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitb16-eupe"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vitt32-eomt")
    class ViTTiny32(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitt32"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vitt32plus-eomt")
    class ViTTinyPlus32(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitt32plus"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vits32-eomt")
    class ViTSmall32(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vits32"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vitb32-eomt")
    class ViTBase32(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitb32"
        num_joint_blocks: int = 3

    @DINOV3_EOMT_PANOPTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vitl32-eomt")
    class ViTLarge32(EoMTPanopticSegmentationConfig):
        backbone_name: str = "dinov3/vitl32"
        num_joint_blocks: int = 4
