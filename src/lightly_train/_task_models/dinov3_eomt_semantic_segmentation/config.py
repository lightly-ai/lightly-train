#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pydantic import Field

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig
from lightly_train._configs.model_registry import (
    DownloadableCheckpoint,
    ModelAlias,
    ModelRegistry,
)


class BackboneArgs(PydanticConfig):
    patch_size: int = 16


class EoMTSemanticSegmentationConfig(PydanticConfig):
    backbone_name: str
    backbone_args: BackboneArgs = Field(default_factory=BackboneArgs)


DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY: ModelRegistry[
    EoMTSemanticSegmentationConfig
] = ModelRegistry()

_DINOV3_VITT16_COCO_URL = "dinov3_vitt16_eomt_coco_260106_104e563e.pt"
_DINOV3_VITT16_COCO_SHA256 = (
    "104e563ebcd8b7d2842db5f0cc6f8d0e67f1607a063ab818725e9af6f6fe7c27"
)
_DINOV3_VITT16PLUS_COCO_URL = "dinov3_vitt16plus_eomt_coco_260106_68339a7d.pt"
_DINOV3_VITT16PLUS_COCO_SHA256 = (
    "68339a7d5baa0dd6fdd88660410939eb78fc8a8c9332145b9b8ac91a2291950b"
)
_DINOV3_VITS16_COCO_URL = "dinov3_vits16_eomt_coco_260105_11be50b5.pt"
_DINOV3_VITS16_COCO_SHA256 = (
    "11be50b578251c974b1fdb413c76e2cd7cfe1e154f6118556bd87477ea205d5a"
)
_DINOV3_VITB16_COCO_URL = "dinov3_vitb16_eomt_coco_260105_92de5e05.pt"
_DINOV3_VITB16_COCO_SHA256 = (
    "92de5e0550f51647e201eef3537a35a8bba75b4e41323b9a7df3c54e6ab400b9"
)
_DINOV3_VITL16_COCO_URL = "dinov3_vitl16_eomt_coco_260105_6169fdd8.pt"
_DINOV3_VITL16_COCO_SHA256 = (
    "6169fdd8edf7d4648c45c6aa1d09b9a4e917ba51dcbd36acf8fbf04a25d1e516"
)
_DINOV3_VITT32_COCO_URL = "dinov3_vitt32_eomt_coco_260106_3ce75c95.pt"
_DINOV3_VITT32_COCO_SHA256 = (
    "3ce75c958aa0d31e3ac14d0bc1e0ca34ccb5b9ab5b141ec40c7f83c1950a2186"
)
_DINOV3_VITT32PLUS_COCO_URL = "dinov3_vitt32plus_eomt_coco_260106_68e19609.pt"
_DINOV3_VITT32PLUS_COCO_SHA256 = (
    "68e196093301bc8a4e73005cebe1cccca75f5c14e58e732d1d9c555ea44e2088"
)
_DINOV3_VITS32_COCO_URL = "dinov3_vits32_eomt_coco_260106_06595b53.pt"
_DINOV3_VITS32_COCO_SHA256 = (
    "06595b53b0ee63032e8f7882a2d1e877c84b996c8313727a6694abf42e871d05"
)
_DINOV3_VITB32_COCO_URL = "dinov3_vitb32_eomt_coco_260106_62cf509e.pt"
_DINOV3_VITB32_COCO_SHA256 = (
    "62cf509e156257347274837087592f27743ba51722c4949bec90688859cc6b6a"
)
_DINOV3_VITL32_COCO_URL = "dinov3_vitl32_eomt_coco_260106_f51348fb.pt"
_DINOV3_VITL32_COCO_SHA256 = (
    "f51348fb4c794889ae35b8d9e2cfe383b42e09e975d2854f2e96fed155edd7d9"
)
_DINOV3_VITS16_CITYSCAPES_URL = (
    "dinov3_eomt/lightlytrain_dinov3_eomt_vits16_cityscapes.pt"
)
_DINOV3_VITS16_CITYSCAPES_SHA256 = (
    "ef7d54eac202bb0a6707fd7115b689a748d032037eccaa3a6891b57b83f18b7e"
)
_DINOV3_VITB16_CITYSCAPES_URL = (
    "dinov3_eomt/lightlytrain_dinov3_eomt_vitb16_cityscapes.pt"
)
_DINOV3_VITB16_CITYSCAPES_SHA256 = (
    "e78e6b1f372ac15c860f64445d8265fd5e9d60271509e106a92b7162096c9560"
)
_DINOV3_VITL16_CITYSCAPES_URL = (
    "dinov3_eomt/lightlytrain_dinov3_eomt_vitl16_cityscapes.pt"
)
_DINOV3_VITL16_CITYSCAPES_SHA256 = (
    "3f397e6ca0af4555adb1da9efa489b734e35fbeac15b4c18e408c63922b41f6c"
)
_DINOV3_VITS16_ADE20K_URL = (
    "dinov3_eomt/lightlytrain_dinov3_eomt_vits16_autolabel_sun397.pt"
)
_DINOV3_VITS16_ADE20K_SHA256 = (
    "f9f002e5adff875e0a97a3b310c26fe5e10c26d69af4e830a4a67aa7dda330aa"
)
_DINOV3_VITB16_ADE20K_URL = (
    "dinov3_eomt/lightlytrain_dinov3_eomt_vitb16_autolabel_sun397.pt"
)
_DINOV3_VITB16_ADE20K_SHA256 = (
    "400f7a1b42a7b67babf253d6aade0be334173d70e7351a01159698ac2d2335ca"
)
_DINOV3_VITL16_ADE20K_URL = "dinov3_eomt/lightlytrain_dinov3_eomt_vitl16_ade20k.pt"
_DINOV3_VITL16_ADE20K_SHA256 = (
    "eb31183c70edd4df8923cba54ce2eefa517ae328cf3caf0106d2795e34382f8f"
)


class DINOv3EoMTSemanticSegmentationConfigRegistry(ConfigsNamespace):
    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/_vittest16-eomt")
    class ViTTest(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/_vittest16"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitt16-eomt-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITT16_COCO_URL,
                sha256=_DINOV3_VITT16_COCO_SHA256,
            ),
        ),
        "dinov3/vitt16-eomt",
    )
    class ViTTiny16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitt16plus-eomt-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITT16PLUS_COCO_URL,
                sha256=_DINOV3_VITT16PLUS_COCO_SHA256,
            ),
        ),
        "dinov3/vitt16plus-eomt",
    )
    class ViTTinyPlus16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16-notpretrained-eomt"
    )
    class ViTTinyNotPretrained16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-notpretrained"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-notpretrained-eomt"
    )
    class ViTTinyPlusNotPretrained16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus-notpretrained"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16-distillationv1-eomt"
    )
    class ViTTinyDistillationV116(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-distillationv1"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-distillationv1-eomt"
    )
    class ViTTinyPlusDistillationV116(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16plus-distillationv1"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vits16-eomt-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITS16_COCO_URL,
                sha256=_DINOV3_VITS16_COCO_SHA256,
            ),
        ),
        ModelAlias(
            name="dinov3/vits16-eomt-cityscapes",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITS16_CITYSCAPES_URL,
                sha256=_DINOV3_VITS16_CITYSCAPES_SHA256,
            ),
        ),
        ModelAlias(
            name="dinov3/vits16-eomt-ade20k",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITS16_ADE20K_URL,
                sha256=_DINOV3_VITS16_ADE20K_SHA256,
            ),
        ),
        "dinov3/vits16-eomt",
    )
    class ViTSmall16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vits16"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vits16plus-eomt")
    class ViTSmallPlus16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vits16plus"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitb16-eomt-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITB16_COCO_URL,
                sha256=_DINOV3_VITB16_COCO_SHA256,
            ),
        ),
        ModelAlias(
            name="dinov3/vitb16-eomt-cityscapes",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITB16_CITYSCAPES_URL,
                sha256=_DINOV3_VITB16_CITYSCAPES_SHA256,
            ),
        ),
        ModelAlias(
            name="dinov3/vitb16-eomt-ade20k",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITB16_ADE20K_URL,
                sha256=_DINOV3_VITB16_ADE20K_SHA256,
            ),
        ),
        "dinov3/vitb16-eomt",
    )
    class ViTBase16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitb16"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitl16-eomt-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITL16_COCO_URL,
                sha256=_DINOV3_VITL16_COCO_SHA256,
            ),
        ),
        ModelAlias(
            name="dinov3/vitl16-eomt-cityscapes",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITL16_CITYSCAPES_URL,
                sha256=_DINOV3_VITL16_CITYSCAPES_SHA256,
            ),
        ),
        ModelAlias(
            name="dinov3/vitl16-eomt-ade20k",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITL16_ADE20K_URL,
                sha256=_DINOV3_VITL16_ADE20K_SHA256,
            ),
        ),
        "dinov3/vitl16-eomt",
    )
    class ViTLarge16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitl16"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vith16plus-eomt")
    class ViTHugePlus16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vith16plus"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register("dinov3/vit7b16-eomt")
    class ViT7B16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vit7b16"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitl16-sat493m-eomt"
    )
    class ViTLargeSAT493M16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitl16-sat493m"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vit7b16-sat493m-eomt"
    )
    class ViT7BSAT493M16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vit7b16-sat493m"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitt16-eupe-eomt"
    )
    class ViTTinyEUPE16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt16-eupe"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vits16-eupe-eomt"
    )
    class ViTSmallEUPE16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vits16-eupe"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        "dinov3/vitb16-eupe-eomt"
    )
    class ViTBaseEUPE16(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitb16-eupe"

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitt32-eomt-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITT32_COCO_URL,
                sha256=_DINOV3_VITT32_COCO_SHA256,
            ),
        ),
        "dinov3/vitt32-eomt",
    )
    class ViTTiny32(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt32"
        backbone_args: BackboneArgs = Field(
            default_factory=lambda: BackboneArgs(patch_size=32)
        )

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitt32plus-eomt-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITT32PLUS_COCO_URL,
                sha256=_DINOV3_VITT32PLUS_COCO_SHA256,
            ),
        ),
        "dinov3/vitt32plus-eomt",
    )
    class ViTTinyPlus32(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitt32plus"
        backbone_args: BackboneArgs = Field(
            default_factory=lambda: BackboneArgs(patch_size=32)
        )

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vits32-eomt-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITS32_COCO_URL,
                sha256=_DINOV3_VITS32_COCO_SHA256,
            ),
        ),
        "dinov3/vits32-eomt",
    )
    class ViTSmall32(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vits32"
        backbone_args: BackboneArgs = Field(
            default_factory=lambda: BackboneArgs(patch_size=32)
        )

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitb32-eomt-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITB32_COCO_URL,
                sha256=_DINOV3_VITB32_COCO_SHA256,
            ),
        ),
        "dinov3/vitb32-eomt",
    )
    class ViTBase32(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitb32"
        backbone_args: BackboneArgs = Field(
            default_factory=lambda: BackboneArgs(patch_size=32)
        )

    @DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitl32-eomt-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITL32_COCO_URL,
                sha256=_DINOV3_VITL32_COCO_SHA256,
            ),
        ),
        "dinov3/vitl32-eomt",
    )
    class ViTLarge32(EoMTSemanticSegmentationConfig):
        backbone_name: str = "dinov3/vitl32"
        backbone_args: BackboneArgs = Field(
            default_factory=lambda: BackboneArgs(patch_size=32)
        )
