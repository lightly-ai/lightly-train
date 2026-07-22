#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig
from lightly_train._configs.model_registry import (
    DownloadableCheckpoint,
    ModelAlias,
    ModelRegistry,
)

LTDETR_SEG_MODEL_REGISTRY: ModelRegistry[SegmentorConfig] = ModelRegistry()

# COCO-pretrained ECViT LT-DETR instance-segmentation weights. The URLs are paths
# relative to DOWNLOADABLE_MODEL_BASE_URL (they live in the ecvit_ltdetrv2_seg_coco/
# subfolder of the S3 bucket).
_LTDETRV2_SEG_S_COCO_URL = "ecvit_ltdetrv2_seg_coco/ltdetrv2-seg-s-coco.pt"
_LTDETRV2_SEG_S_COCO_SHA256 = (
    "5c7e00895e10a5b8a14cb9ad1c164232a16af302719fd7a2f7de241264155c15"
)
_LTDETRV2_SEG_M_COCO_URL = "ecvit_ltdetrv2_seg_coco/ltdetrv2-seg-m-coco.pt"
_LTDETRV2_SEG_M_COCO_SHA256 = (
    "4527278b7e1d819fecbf72fb90554f665a506f178ef30b32f22c227107970384"
)
_LTDETRV2_SEG_L_COCO_URL = "ecvit_ltdetrv2_seg_coco/ltdetrv2-seg-l-coco.pt"
_LTDETRV2_SEG_L_COCO_SHA256 = (
    "601b9d8b51d73105ad11feae0dfc4d8d085a12d0afac3f991e7d83f9f493d58b"
)
_LTDETRV2_SEG_X_COCO_URL = "ecvit_ltdetrv2_seg_coco/ltdetrv2-seg-x-coco.pt"
_LTDETRV2_SEG_X_COCO_SHA256 = (
    "d55b16f48f05f18e6dd03e3c5c2a3894d0bdeaf468dc80280a727edf5086edcd"
)


class HybridEncoderConfig(PydanticConfig):
    in_channels: list[int]
    feat_strides: list[int] | Literal["auto"] = "auto"
    hidden_dim: int
    use_encoder_idx: list[int]
    num_encoder_layers: int
    nhead: int
    dim_feedforward: int
    dropout: float
    enc_act: str
    expansion: float
    depth_mult: float
    act: str
    upsample: bool = True

    def resolve_auto(self, patch_size: int | None) -> None:
        patch_size = patch_size or 16
        if self.feat_strides == "auto":
            self.feat_strides = [
                int(patch_size * (2 ** (i - 1))) for i in range(len(self.in_channels))
            ]


class LTDETRHybridEncoderConfig(ConfigsNamespace):
    class ViTTiny(HybridEncoderConfig):
        in_channels: list[int] = [192, 192, 192]
        hidden_dim: int = 192
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 512
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 0.34
        depth_mult: float = 0.67
        act: str = "silu"

    class ViTTinyPlus(HybridEncoderConfig):
        in_channels: list[int] = [256, 256, 256]
        hidden_dim: int = 256
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 512
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 0.67
        depth_mult: float = 1.0
        act: str = "silu"


class ECSegTransformerConfig(PydanticConfig):
    feat_channels: list[int]
    feat_strides: list[int] | Literal["auto"] = "auto"
    hidden_dim: int
    num_levels: int = 3
    num_layers: int
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [3, 6, 3]
    query_select_method: str = "default"
    cross_attn_method: str = "default"
    dim_feedforward: int
    reg_max: int = 32
    reg_scale: float = 4.0
    layer_scale: float = 1.0

    def resolve_auto(self, patch_size: int | None) -> None:
        patch_size = patch_size or 16
        if self.feat_strides == "auto":
            self.feat_strides = [
                int(patch_size * (2 ** (i - 1))) for i in range(self.num_levels)
            ]


class LTDETRECSegTransformerConfig(ConfigsNamespace):
    class ViTTiny(ECSegTransformerConfig):
        feat_channels: list[int] = [192, 192, 192]
        hidden_dim: int = 192
        num_layers: int = 4
        dim_feedforward: int = 512

    class ViTTinyPlus(ECSegTransformerConfig):
        feat_channels: list[int] = [256, 256, 256]
        hidden_dim: int = 256
        num_layers: int = 4
        dim_feedforward: int = 512


class ECViTBackboneWrapperConfig(PydanticConfig):
    finetune: bool = True

    def resolve_auto(self, patch_size: int | None) -> None:
        if patch_size is not None and patch_size != 16:
            raise ValueError(
                "ECViT (EdgeCrafter) backbones only support patch_size=16, "
                f"but got patch_size={patch_size}."
            )


class ECSegPostProcessorConfig(PydanticConfig):
    num_top_queries: int = 300


class SegmentorConfig(PydanticConfig):
    backbone_name: str
    hybrid_encoder: HybridEncoderConfig
    transformer: ECSegTransformerConfig
    ecseg_postprocessor: ECSegPostProcessorConfig
    backbone_wrapper: ECViTBackboneWrapperConfig
    backbone_args: dict[str, Any]

    def resolve_auto(self, patch_size: int | None) -> None:
        self.backbone_wrapper.resolve_auto(patch_size=patch_size)
        self.hybrid_encoder.resolve_auto(patch_size=patch_size)
        self.transformer.resolve_auto(patch_size=patch_size)


class LTDETRBaseConfig(ConfigsNamespace):
    class ViTTiny(SegmentorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTTiny
        )
        ecseg_postprocessor: ECSegPostProcessorConfig = Field(
            default_factory=ECSegPostProcessorConfig
        )

    class ViTTinyPlus(SegmentorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTTinyPlus
        )
        ecseg_postprocessor: ECSegPostProcessorConfig = Field(
            default_factory=ECSegPostProcessorConfig
        )


class LTDETRv2ConfigRegistry(ConfigsNamespace):
    @LTDETR_SEG_MODEL_REGISTRY.register(
        "edgecrafter/ecvitt-ltdetr-seg",
        ModelAlias(
            name="edgecrafter/ecvitt-ltdetr-seg-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_LTDETRV2_SEG_S_COCO_URL,
                sha256=_LTDETRV2_SEG_S_COCO_SHA256,
            ),
        ),
        "ltdetrv2-seg-s",
        ModelAlias(
            name="ltdetrv2-seg-s-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_LTDETRV2_SEG_S_COCO_URL,
                sha256=_LTDETRV2_SEG_S_COCO_SHA256,
            ),
        ),
    )
    class EdgeCrafterECViTTiny(LTDETRBaseConfig.ViTTiny):
        backbone_name: str = "edgecrafter/ecvitt"
        transformer: ECSegTransformerConfig = Field(
            default_factory=LTDETRECSegTransformerConfig.ViTTiny
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_SEG_MODEL_REGISTRY.register(
        "_ltdetrv2-seg-s-notpretrained", include_in_model_list=False
    )
    class EdgeCrafterECViTTinyNotPretrained(EdgeCrafterECViTTiny):
        backbone_name: str = "edgecrafter/_ecvitt-notpretrained"

    @LTDETR_SEG_MODEL_REGISTRY.register(
        "edgecrafter/ecvittplus-ltdetr-seg",
        ModelAlias(
            name="edgecrafter/ecvittplus-ltdetr-seg-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_LTDETRV2_SEG_M_COCO_URL,
                sha256=_LTDETRV2_SEG_M_COCO_SHA256,
            ),
        ),
        "ltdetrv2-seg-m",
        ModelAlias(
            name="ltdetrv2-seg-m-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_LTDETRV2_SEG_M_COCO_URL,
                sha256=_LTDETRV2_SEG_M_COCO_SHA256,
            ),
        ),
    )
    class EdgeCrafterECViTTinyPlus(LTDETRBaseConfig.ViTTinyPlus):
        backbone_name: str = "edgecrafter/ecvittplus"
        transformer: ECSegTransformerConfig = Field(
            default_factory=LTDETRECSegTransformerConfig.ViTTinyPlus
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_SEG_MODEL_REGISTRY.register(
        "edgecrafter/ecvits-ltdetr-seg",
        ModelAlias(
            name="edgecrafter/ecvits-ltdetr-seg-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_LTDETRV2_SEG_L_COCO_URL,
                sha256=_LTDETRV2_SEG_L_COCO_SHA256,
            ),
        ),
        "ltdetrv2-seg-l",
        ModelAlias(
            name="ltdetrv2-seg-l-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_LTDETRV2_SEG_L_COCO_URL,
                sha256=_LTDETRV2_SEG_L_COCO_SHA256,
            ),
        ),
    )
    class EdgeCrafterECViTSmall(LTDETRBaseConfig.ViTTinyPlus):
        backbone_name: str = "edgecrafter/ecvits"
        transformer: ECSegTransformerConfig = Field(
            default_factory=LTDETRECSegTransformerConfig.ViTTinyPlus
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_SEG_MODEL_REGISTRY.register(
        "edgecrafter/ecvitsplus-ltdetr-seg",
        ModelAlias(
            name="edgecrafter/ecvitsplus-ltdetr-seg-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_LTDETRV2_SEG_X_COCO_URL,
                sha256=_LTDETRV2_SEG_X_COCO_SHA256,
            ),
        ),
        "ltdetrv2-seg-x",
        ModelAlias(
            name="ltdetrv2-seg-x-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_LTDETRV2_SEG_X_COCO_URL,
                sha256=_LTDETRV2_SEG_X_COCO_SHA256,
            ),
        ),
    )
    class EdgeCrafterECViTSmallPlus(LTDETRBaseConfig.ViTTinyPlus):
        backbone_name: str = "edgecrafter/ecvitsplus"
        transformer: ECSegTransformerConfig = Field(
            default_factory=LTDETRECSegTransformerConfig.ViTTinyPlus
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )
