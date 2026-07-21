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


class EoMTInstanceSegmentationConfig(PydanticConfig):
    backbone_name: str


DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY: ModelRegistry[
    EoMTInstanceSegmentationConfig
] = ModelRegistry()

_DINOV3_VITT16_INST_COCO_URL = "dinov3_vitt16_eomt_inst_coco_260109_45e0aff8.pt"
_DINOV3_VITT16_INST_COCO_SHA256 = (
    "45e0aff8c5c8054a3240fcbc368b4e7f87e8066c1e100e3ef9d9c60c7d949a17"
)
_DINOV3_VITT16PLUS_INST_COCO_URL = "dinov3_vitt16plus_eomt_inst_coco_260109_0e20aa05.pt"
_DINOV3_VITT16PLUS_INST_COCO_SHA256 = (
    "0e20aa05ef15003d7d9462400d32ecc671e7a8d256ae061d42dd4f8978feb621"
)
_DINOV3_VITS16_INST_COCO_URL = "/dinov3_eomt/dinov3_vits16_eomt_inst_coco.pt"
_DINOV3_VITS16_INST_COCO_SHA256 = (
    "b54dafb12d550958cc5c9818b061fba0d8b819423581d02080221d0199e1cc37"
)
_DINOV3_VITB16_INST_COCO_URL = "/dinov3_eomt/dinov3_vitb16_eomt_inst_coco.pt"
_DINOV3_VITB16_INST_COCO_SHA256 = (
    "a57b5e7afd5cd64422d74d400f30693f80f96fa63184960250fb0878afd3c7f6"
)
_DINOV3_VITL16_INST_COCO_URL = "/dinov3_eomt/dinov3_vitl16_eomt_inst_coco.pt"
_DINOV3_VITL16_INST_COCO_SHA256 = (
    "1aac5ac16dcbc1a12cc6f8d4541bea5e7940937a49f0b1dcea7394956b6e46e5"
)


class DINOv3EoMTInstanceSegmentationConfigRegistry(ConfigsNamespace):
    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register("dinov3/_vittest16-eomt")
    class ViTTest(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/_vittest16"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitt16-eomt-inst-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITT16_INST_COCO_URL,
                sha256=_DINOV3_VITT16_INST_COCO_SHA256,
            ),
        ),
        "dinov3/vitt16-eomt",
    )
    class ViTTiny(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitt16"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitt16plus-eomt-inst-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITT16PLUS_INST_COCO_URL,
                sha256=_DINOV3_VITT16PLUS_INST_COCO_SHA256,
            ),
        ),
        "dinov3/vitt16plus-eomt",
    )
    class ViTTinyPlus(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vits16-eomt-inst-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITS16_INST_COCO_URL,
                sha256=_DINOV3_VITS16_INST_COCO_SHA256,
            ),
        ),
        "dinov3/vits16-eomt",
    )
    class ViTSmall(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vits16"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitb16-eomt-inst-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITB16_INST_COCO_URL,
                sha256=_DINOV3_VITB16_INST_COCO_SHA256,
            ),
        ),
        "dinov3/vitb16-eomt",
    )
    class ViTBase(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitb16"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitl16-eomt-inst-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITL16_INST_COCO_URL,
                sha256=_DINOV3_VITL16_INST_COCO_SHA256,
            ),
        ),
        "dinov3/vitl16-eomt",
    )
    class ViTLarge(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitl16"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16-notpretrained-eomt"
    )
    class ViTTinyNotPretrained(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-notpretrained"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-notpretrained-eomt"
    )
    class ViTTinyPlusNotPretrained(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus-notpretrained"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16-distillationv1-eomt"
    )
    class ViTTinyDistillationV1(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-distillationv1"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-distillationv1-eomt"
    )
    class ViTTinyPlusDistillationV1(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus-distillationv1"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vits16plus-eomt")
    class ViTSmallPlus(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vits16plus"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vith16plus-eomt")
    class ViTHugePlus(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vith16plus"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vit7b16-eomt")
    class ViT7B(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vit7b16"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitl16-sat493m-eomt"
    )
    class ViTLargeSAT493M(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitl16-sat493m"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vit7b16-sat493m-eomt"
    )
    class ViT7BSAT493M(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vit7b16-sat493m"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16-eupe-eomt"
    )
    class ViTTinyEUPE(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-eupe"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vits16-eupe-eomt"
    )
    class ViTSmallEUPE(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vits16-eupe"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitb16-eupe-eomt"
    )
    class ViTBaseEUPE(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitb16-eupe"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vits16-lingbot-eomt"
    )
    class ViTSmallLingBot(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vits16-lingbot"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitb16-lingbot-eomt"
    )
    class ViTBaseLingBot(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitb16-lingbot"

    @DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitl16-lingbot-eomt"
    )
    class ViTLargeLingBot(EoMTInstanceSegmentationConfig):
        backbone_name: str = "dinov3/vitl16-lingbot"
