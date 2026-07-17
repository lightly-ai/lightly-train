#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import Field
from typing_extensions import Annotated

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig
from lightly_train._configs.model_registry import (
    DownloadableCheckpoint,
    ModelAlias,
    ModelRegistry,
)

logger = logging.getLogger(__name__)


LTDETR_MODEL_REGISTRY: ModelRegistry[DetectorConfig] = ModelRegistry()

# COCO-pretrained weights shared by the ltdetrv2-s-coco /
# edgecrafter-ecvitt-ltdetr-coco aliases.
_ECVITT_COCO_URL = "edgecrafter_ecvitt_ltdetr_coco_260624_f8aefe49.pt"
_ECVITT_COCO_SHA256 = "f8aefe499be1579c55bfcb288f623399ea5f4efef0c5a5f00960663efeda4f49"

_DINOV3_VITT16_COCO_URL = "dinov3_vitt16_ltdetr_coco_251218_dfd34210.pt"
_DINOV3_VITT16_COCO_SHA256 = (
    "dfd34210a1a3375793d149a55d9b49e6e8b783458bdd4cd76fd28fa2d61dbb37"
)
_DINOV3_VITT16PLUS_COCO_URL = "dinov3_vitt16plus_ltdetr_coco_251218_af499c82.pt"
_DINOV3_VITT16PLUS_COCO_SHA256 = (
    "af499c825436013098a77a028ff5cf08dbf31118f4d68b15eefa6fdd9635f5d2"
)
_DINOV3_VITS16_COCO_URL = "dinov3_vits16_ltdetr_coco_251218_4812416b.pt"
_DINOV3_VITS16_COCO_SHA256 = (
    "4812416b861a80f305889cf1408775044c8b05f1baf9be45cd4b1d0edd5d4532"
)
_DINOV3_CONVNEXT_TINY_COCO_URL = "dinov3_convnext_tiny_ltdetr_coco_251218_35bbc4fb.pt"
_DINOV3_CONVNEXT_TINY_COCO_SHA256 = (
    "35bbc4fbec3bb9fa113a33f1013abaab1952edf3335f98624b5914812d63d26c"
)
_DINOV3_CONVNEXT_SMALL_COCO_URL = "dinov3_convnext_small_ltdetr_coco_251218_8f7109ab.pt"
_DINOV3_CONVNEXT_SMALL_COCO_SHA256 = (
    "8f7109ab406aa92791e4e4ca6249ab9a863734795676c81b91dbd4cc4b1ef387"
)
_DINOV3_CONVNEXT_BASE_COCO_URL = "dinov3_convnext_base_ltdetr_coco_251218_836adb6b.pt"
_DINOV3_CONVNEXT_BASE_COCO_SHA256 = (
    "836adb6b5122665a24b6da3ee1720b9f3d0fc3c30cee44cfbd98dcb79fe0809a"
)
_DINOV3_CONVNEXT_LARGE_COCO_URL = "dinov3_convnext_large_ltdetr_coco_251218_03fe6750.pt"
_DINOV3_CONVNEXT_LARGE_COCO_SHA256 = (
    "03fe6750392daf3ecd32bbab3f144bd5c4d6cdc8bd75635f9e1c5e296e7dd8b0"
)
_DINOV2_VITS14_NOREG_COCO_URL = "dinov2_vits14_noreg_ltdetr_coco_251218_4e1f523d.pt"
_DINOV2_VITS14_NOREG_COCO_SHA256 = (
    "4e1f523db68c94516ee5b35a91f24267657af474bea58b52a7f7e51ec2d8f717"
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
    upsample: bool
    state_dict_ignore_keys: set[str] = Field(default_factory=set)

    def resolve_auto(self, patch_size: int | None) -> None:
        patch_size = patch_size or 16
        if self.feat_strides == "auto":
            self.feat_strides = [
                int(patch_size * (2 ** (i - 1))) for i in range(len(self.in_channels))
            ]


class LTDETRHybridEncoderConfig(ConfigsNamespace):
    class CNNTiny(HybridEncoderConfig):
        in_channels: list[int] = [192, 384, 768]
        feat_strides: list[int] = [8, 16, 32]
        hidden_dim: int = 384
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 2048
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 1.0
        depth_mult: float = 1
        act: str = "silu"
        upsample: bool = True

    class CNNSmall(HybridEncoderConfig):
        in_channels: list[int] = [192, 384, 768]
        feat_strides: list[int] = [8, 16, 32]
        hidden_dim: int = 384
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 2048
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 1.0
        depth_mult: float = 1
        act: str = "silu"
        upsample: bool = True

    class CNNBase(HybridEncoderConfig):
        in_channels: list[int] = [256, 512, 1024]
        feat_strides: list[int] = [8, 16, 32]
        hidden_dim: int = 384
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 2048
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 1.0
        depth_mult: float = 1
        act: str = "silu"
        upsample: bool = True

    class CNNLarge(HybridEncoderConfig):
        in_channels: list[int] = [384, 768, 1536]
        feat_strides: list[int] = [8, 16, 32]
        hidden_dim: int = 384
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 2048
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 1.0
        depth_mult: float = 1
        act: str = "silu"
        upsample: bool = True

    class ViTTest(HybridEncoderConfig):
        in_channels: list[int] = [8, 8, 8]
        hidden_dim: int = 8
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 1
        dim_feedforward: int = 32
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 1.0
        depth_mult: float = 1.0
        act: str = "silu"
        upsample: bool = True

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
        upsample: bool = True

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
        upsample: bool = True

    class ViTSmall(HybridEncoderConfig):
        in_channels: list[int] = [224, 224, 224]
        hidden_dim: int = 224
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 896
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 1.0
        depth_mult: float = 1.0
        act: str = "silu"
        upsample: bool = True

    class DINOv2ViTSmallNoRegistersLegacy(HybridEncoderConfig):
        in_channels: list[int] = [384, 384, 384]
        feat_strides: list[int] = [14, 14, 14]
        hidden_dim: int = 384
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 2048
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 1.0
        depth_mult: float = 1.0
        act: str = "silu"
        # The original checkpoint carries real (but unused) downsample_convs
        # weights. We don't allocate that module (upsample=False, since all 3
        # backbone taps already share the same spatial resolution at stride
        # 14), so its checkpoint keys are dropped before the strict state_dict
        # load.
        upsample: bool = False
        state_dict_ignore_keys: set[str] = Field(
            default_factory=lambda: {"downsample_convs."}
        )

    class ViTBase(HybridEncoderConfig):
        in_channels: list[int] = [768, 768, 768]
        hidden_dim: int = 768
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 3072
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 1.0
        depth_mult: float = 1.0
        act: str = "silu"
        upsample: bool = True

    class ViTLarge(HybridEncoderConfig):
        in_channels: list[int] = [1024, 1024, 1024]
        hidden_dim: int = 1024
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 4096
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 1.0
        depth_mult: float = 1.0
        act: str = "silu"
        upsample: bool = True

    class ViTGiant(HybridEncoderConfig):
        in_channels: list[int] = [1536, 1536, 1536]
        hidden_dim: int = 1536
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 6144
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 1.0
        depth_mult: float = 1.0
        act: str = "silu"
        upsample: bool = True


class RTDETRTransformerv2Config(PydanticConfig):
    decoder_name: Literal["rtdetrv2"] = "rtdetrv2"
    feat_channels: list[int]
    feat_strides: list[int] | Literal["auto"] = "auto"
    hidden_dim: int = 256
    num_levels: int = 3
    num_layers: int = 6
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [4, 4, 4]
    query_select_method: str = "default"

    def resolve_auto(self, patch_size: int | None) -> None:
        patch_size = patch_size or 16
        if self.feat_strides == "auto":
            self.feat_strides = [
                int(patch_size * (2 ** (i - 1))) for i in range(self.num_levels)
            ]


class LTDETRRTDETRTransformerv2Config(ConfigsNamespace):
    class CNNTiny(RTDETRTransformerv2Config):
        feat_channels: list[int] = [384, 384, 384]

    class CNNSmall(RTDETRTransformerv2Config):
        feat_channels: list[int] = [384, 384, 384]

    class CNNBase(RTDETRTransformerv2Config):
        feat_channels: list[int] = [384, 384, 384]

    class CNNLarge(RTDETRTransformerv2Config):
        feat_channels: list[int] = [384, 384, 384]

    class ViTTest(RTDETRTransformerv2Config):
        feat_channels: list[int] = [8, 8, 8]
        hidden_dim: int = 8
        num_layers: int = 1
        num_queries: int = 20
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 32

    class ViTTiny(RTDETRTransformerv2Config):
        feat_channels: list[int] = [192, 192, 192]
        hidden_dim: int = 192
        num_layers: int = 4
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 512

    class ViTTinyPlus(RTDETRTransformerv2Config):
        feat_channels: list[int] = [256, 256, 256]
        hidden_dim: int = 256
        num_layers: int = 4
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 512

    class ViTSmall(RTDETRTransformerv2Config):
        feat_channels: list[int] = [224, 224, 224]
        hidden_dim: int = 224
        num_layers: int = 4
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 1792

    class DINOv2ViTSmallNoRegistersLegacy(RTDETRTransformerv2Config):
        feat_channels: list[int] = [384, 384, 384]
        feat_strides: list[int] = [14, 14, 14]
        hidden_dim: int = 256
        num_layers: int = 6
        num_points: list[int] = [4, 4, 4]
        dim_feedforward: int = 1024

    class ViTBase(RTDETRTransformerv2Config):
        feat_channels: list[int] = [768, 768, 768]
        hidden_dim: int = 768
        num_layers: int = 4
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 6144

    class ViTLarge(RTDETRTransformerv2Config):
        feat_channels: list[int] = [1024, 1024, 1024]
        hidden_dim: int = 1024
        num_layers: int = 4
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 8192

    class ViTGiant(RTDETRTransformerv2Config):
        feat_channels: list[int] = [1536, 1536, 1536]
        hidden_dim: int = 1536
        num_layers: int = 4
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 12288


class DFINETransformerConfig(PydanticConfig):
    decoder_name: Literal["dfine"] = "dfine"
    feat_channels: list[int]
    feat_strides: list[int] | Literal["auto"] = "auto"
    hidden_dim: int = 256
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
    dim_feedforward: int = 2048
    reg_max: int = 32
    reg_scale: float = 4.0
    layer_scale: float = 1.0

    def resolve_auto(self, patch_size: int | None) -> None:
        patch_size = patch_size or 16
        if self.feat_strides == "auto":
            self.feat_strides = [
                int(patch_size * (2 ** (i - 1))) for i in range(self.num_levels)
            ]


class LTDETRDFINETransformerConfig(ConfigsNamespace):
    class CNNTiny(DFINETransformerConfig):
        feat_channels: list[int] = [384, 384, 384]
        feat_strides: list[int] = [8, 16, 32]
        num_layers: int = 3

    class CNNSmall(DFINETransformerConfig):
        feat_channels: list[int] = [384, 384, 384]
        feat_strides: list[int] = [8, 16, 32]
        num_layers: int = 3

    class CNNBase(DFINETransformerConfig):
        feat_channels: list[int] = [384, 384, 384]
        feat_strides: list[int] = [8, 16, 32]
        num_layers: int = 4

    class CNNLarge(DFINETransformerConfig):
        feat_channels: list[int] = [384, 384, 384]
        feat_strides: list[int] = [8, 16, 32]
        num_layers: int = 6
        reg_scale: float = 8.0

    class ViTTest(DFINETransformerConfig):
        feat_channels: list[int] = [8, 8, 8]
        hidden_dim: int = 8
        num_layers: int = 1
        num_queries: int = 20
        dim_feedforward: int = 32

    class ViTTiny(DFINETransformerConfig):
        feat_channels: list[int] = [192, 192, 192]
        hidden_dim: int = 192
        num_layers: int = 4
        dim_feedforward: int = 512

    class ViTTinyPlus(DFINETransformerConfig):
        feat_channels: list[int] = [256, 256, 256]
        hidden_dim: int = 256
        num_layers: int = 4
        dim_feedforward: int = 512

    class ViTSmall(DFINETransformerConfig):
        feat_channels: list[int] = [224, 224, 224]
        hidden_dim: int = 224
        num_layers: int = 4
        dim_feedforward: int = 1792

    class ViTBase(DFINETransformerConfig):
        feat_channels: list[int] = [768, 768, 768]
        hidden_dim: int = 768
        num_layers: int = 4
        dim_feedforward: int = 6144

    class ViTLarge(DFINETransformerConfig):
        feat_channels: list[int] = [1024, 1024, 1024]
        hidden_dim: int = 1024
        num_layers: int = 4
        dim_feedforward: int = 8192

    class ViTGiant(DFINETransformerConfig):
        feat_channels: list[int] = [1536, 1536, 1536]
        hidden_dim: int = 1536
        num_layers: int = 4
        dim_feedforward: int = 12288


class CNNBackboneWrapperConfig(PydanticConfig):
    finetune: bool = True

    def resolve_auto(self, patch_size: int | None) -> None:
        """CNNs don't have a patch size, so this will fail."""
        if patch_size is not None:
            raise ValueError(
                "CNNBackboneWrapperConfig does not support patch_size, "
                "but a patch_size was provided. Patch sizes are only "
                "relevant for ViT backbones."
            )


class ECViTBackboneWrapperConfig(PydanticConfig):
    finetune: bool = True

    def resolve_auto(self, patch_size: int | None) -> None:
        if patch_size is not None and patch_size != 16:
            raise ValueError(
                "ECViT (EdgeCrafter) backbones only support patch_size=16, "
                f"but got patch_size={patch_size}."
            )


class RTDETRBackboneWrapperConfig(PydanticConfig):
    interaction_indexes: list[int]
    finetune: bool
    use_sta: bool
    conv_inplane: int | Literal["auto"] = "auto"
    conv_inplane_factor: int = 2
    hidden_dim: int
    project_features: bool
    resize_features: bool = True

    def resolve_auto(self, patch_size: int | None) -> None:
        patch_size = patch_size or 16
        if self.conv_inplane == "auto":
            self.conv_inplane = self.conv_inplane_factor * patch_size


class LTDETRRTDETRBackboneWrapperConfig(ConfigsNamespace):
    class ViTTiny(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 1
        hidden_dim: int = 192
        project_features: bool = True

    class ViTTinyPlus(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 1
        hidden_dim: int = 256
        project_features: bool = True

    class ViTSmall(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 2
        hidden_dim: int = 224
        project_features: bool = True

    class ViTBase(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 4
        hidden_dim: int = 768
        project_features: bool = True

    class ViTLarge(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [11, 17, 23]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 4
        hidden_dim: int = 1024
        project_features: bool = True

    class ViTGiant(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [19, 29, 39]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 4
        hidden_dim: int = 1536
        project_features: bool = True


class LTDETRRTDETRNoSTABackboneWrapperConfig(ConfigsNamespace):
    class ViTTest(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [0, 1, 2]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 2
        hidden_dim: int = 8
        project_features: bool = True

    class ViTTiny(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 1
        hidden_dim: int = 192
        project_features: bool = True

    class ViTTinyPlus(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 1
        hidden_dim: int = 256
        project_features: bool = True

    class ViTSmall(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 2
        hidden_dim: int = 224
        project_features: bool = True

    class DINOv2ViTSmallNoRegistersLegacy(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 2
        hidden_dim: int = 384
        project_features: bool = False
        resize_features: bool = False

    class ViTBase(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 4
        hidden_dim: int = 768
        project_features: bool = True

    class ViTLarge(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [11, 17, 23]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 4
        hidden_dim: int = 1024
        project_features: bool = True

    class ViTGiant(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [19, 29, 39]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 4
        hidden_dim: int = 1536
        project_features: bool = True


class RTDETRPostProcessorConfig(PydanticConfig):
    num_top_queries: int


class LTDETRPostProcessorConfig(ConfigsNamespace):
    class Generic(RTDETRPostProcessorConfig):
        num_top_queries: int = 300

    class ViTTest(RTDETRPostProcessorConfig):
        num_top_queries: int = 20


class DetectorConfig(PydanticConfig):
    version: Literal["v1", "v2"]
    backbone_name: str  # full "package/backbone" string, e.g. "dinov3/vits16"
    hybrid_encoder: HybridEncoderConfig
    transformer: Annotated[
        RTDETRTransformerv2Config | DFINETransformerConfig,
        Field(discriminator="decoder_name"),
    ]
    rtdetr_postprocessor: RTDETRPostProcessorConfig
    backbone_wrapper: (
        RTDETRBackboneWrapperConfig
        | CNNBackboneWrapperConfig
        | ECViTBackboneWrapperConfig
    )
    backbone_args: dict[str, Any]

    def resolve_auto(self, patch_size: int | None) -> None:
        self.backbone_wrapper.resolve_auto(patch_size=patch_size)
        self.hybrid_encoder.resolve_auto(patch_size=patch_size)
        self.transformer.resolve_auto(patch_size=patch_size)


class LTDETRBaseConfig(ConfigsNamespace):
    class CNNLarge(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.CNNLarge
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )
        backbone_wrapper: CNNBackboneWrapperConfig = Field(
            default_factory=CNNBackboneWrapperConfig
        )

    class CNNBase(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.CNNBase
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )
        backbone_wrapper: CNNBackboneWrapperConfig = Field(
            default_factory=CNNBackboneWrapperConfig
        )

    class CNNSmall(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.CNNSmall
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )
        backbone_wrapper: CNNBackboneWrapperConfig = Field(
            default_factory=CNNBackboneWrapperConfig
        )

    class CNNTiny(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.CNNTiny
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )
        backbone_wrapper: CNNBackboneWrapperConfig = Field(
            default_factory=CNNBackboneWrapperConfig
        )

    class ViTTest(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTTest
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.ViTTest
        )

    class ViTTiny(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTTiny
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )

    class ViTTinyPlus(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTTinyPlus
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )

    class ViTSmall(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTSmall
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )

    class DINOv2ViTSmallNoRegistersLegacy(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.DINOv2ViTSmallNoRegistersLegacy
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )

    class ViTBase(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTBase
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )

    class ViTLarge(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTLarge
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )

    class ViTGiant(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTGiant
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )


class LTDETRFastViTHybridEncoderConfig(ConfigsNamespace):
    class Nano(HybridEncoderConfig):
        in_channels: list[int] = [96, 192, 384]
        feat_strides: list[int] = [8, 16, 32]
        hidden_dim: int = 128
        use_encoder_idx: list[int] = [2]
        num_encoder_layers: int = 1
        nhead: int = 8
        dim_feedforward: int = 512
        dropout: float = 0.0
        enc_act: str = "gelu"
        expansion: float = 0.34
        depth_mult: float = 0.67
        act: str = "silu"
        upsample: bool = True

    class Tiny(Nano):
        in_channels: list[int] = [128, 256, 512]
        hidden_dim: int = 192

    class Small(Tiny):
        hidden_dim: int = 256
        expansion: float = 0.67
        depth_mult: float = 1.0

    class Medium(Small):
        hidden_dim: int = 288
        dim_feedforward: int = 1152
        expansion: float = 1.0

    class Large(Medium):
        hidden_dim: int = 320
        dim_feedforward: int = 1280

    class LargeMA36(Large):
        in_channels: list[int] = [152, 304, 608]


class LTDETRFastViTRTDETRTransformerv2Config(ConfigsNamespace):
    class Nano(RTDETRTransformerv2Config):
        feat_channels: list[int] = [128, 128, 128]
        feat_strides: list[int] = [8, 16, 32]
        hidden_dim: int = 128
        num_layers: int = 3
        dim_feedforward: int = 512
        num_points: list[int] = [3, 6, 3]

    class Tiny(Nano):
        feat_channels: list[int] = [192, 192, 192]
        hidden_dim: int = 192
        num_layers: int = 4

    class Small(Tiny):
        feat_channels: list[int] = [256, 256, 256]
        hidden_dim: int = 256

    class Medium(Small):
        feat_channels: list[int] = [288, 288, 288]
        hidden_dim: int = 288
        dim_feedforward: int = 1152

    class Large(Medium):
        feat_channels: list[int] = [320, 320, 320]
        hidden_dim: int = 320
        num_layers: int = 6
        dim_feedforward: int = 1280


class LTDETRFastViTDFINETransformerConfig(ConfigsNamespace):
    class Nano(DFINETransformerConfig):
        feat_channels: list[int] = [128, 128, 128]
        feat_strides: list[int] = [8, 16, 32]
        hidden_dim: int = 128
        num_layers: int = 3
        dim_feedforward: int = 512

    class Tiny(Nano):
        feat_channels: list[int] = [192, 192, 192]
        hidden_dim: int = 192
        num_layers: int = 4

    class Small(Tiny):
        feat_channels: list[int] = [256, 256, 256]
        hidden_dim: int = 256

    class Medium(Small):
        feat_channels: list[int] = [288, 288, 288]
        hidden_dim: int = 288
        dim_feedforward: int = 1152

    class Large(Medium):
        feat_channels: list[int] = [320, 320, 320]
        hidden_dim: int = 320
        num_layers: int = 6
        dim_feedforward: int = 1280
        reg_scale: float = 8.0


class LTDETRFastViTBaseConfig(ConfigsNamespace):
    class Nano(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRFastViTHybridEncoderConfig.Nano
        )
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRFastViTDFINETransformerConfig.Nano
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=LTDETRPostProcessorConfig.Generic
        )
        backbone_wrapper: CNNBackboneWrapperConfig = Field(
            default_factory=CNNBackboneWrapperConfig
        )

    class Tiny(Nano):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRFastViTHybridEncoderConfig.Tiny
        )
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRFastViTDFINETransformerConfig.Tiny
        )

    class Small(Tiny):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRFastViTHybridEncoderConfig.Small
        )
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRFastViTDFINETransformerConfig.Small
        )

    class Medium(Small):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRFastViTHybridEncoderConfig.Medium
        )
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRFastViTDFINETransformerConfig.Medium
        )

    class Large(Medium):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRFastViTHybridEncoderConfig.Large
        )
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRFastViTDFINETransformerConfig.Large
        )


class LTDETRConfigRegistry(ConfigsNamespace):
    @LTDETR_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/convnext-tiny-ltdetr-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_CONVNEXT_TINY_COCO_URL,
                sha256=_DINOV3_CONVNEXT_TINY_COCO_SHA256,
            ),
        ),
        "dinov3/convnext-tiny-ltdetr",
        "dinov3/convnext-tiny-eupe-ltdetr",
    )
    class DINOv3ConvNeXtTiny(LTDETRBaseConfig.CNNTiny):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov3/convnext-tiny"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNTiny
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/convnext-small-ltdetr-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_CONVNEXT_SMALL_COCO_URL,
                sha256=_DINOV3_CONVNEXT_SMALL_COCO_SHA256,
            ),
        ),
        "dinov3/convnext-small-ltdetr",
        "dinov3/convnext-small-eupe-ltdetr",
    )
    class DINOv3ConvNeXtSmall(LTDETRBaseConfig.CNNSmall):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov3/convnext-small"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNSmall
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/convnext-base-ltdetr-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_CONVNEXT_BASE_COCO_URL,
                sha256=_DINOV3_CONVNEXT_BASE_COCO_SHA256,
            ),
        ),
        "dinov3/convnext-base-ltdetr",
        "dinov3/convnext-base-eupe-ltdetr",
    )
    class DINOv3ConvNeXtBase(LTDETRBaseConfig.CNNBase):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov3/convnext-base"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNBase
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/convnext-large-ltdetr-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_CONVNEXT_LARGE_COCO_URL,
                sha256=_DINOV3_CONVNEXT_LARGE_COCO_SHA256,
            ),
        ),
        "dinov3/convnext-large-ltdetr",
    )
    class DINOv3ConvNeXtLarge(LTDETRBaseConfig.CNNLarge):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov3/convnext-large"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNLarge
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitt16-ltdetr-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITT16_COCO_URL,
                sha256=_DINOV3_VITT16_COCO_SHA256,
            ),
        ),
        "dinov3/vitt16-ltdetr",
        "dinov3/vitt16-eupe-ltdetr",
        "dinov3/vitt16-notpretrained-ltdetr",
    )
    class DINOv3ViTTiny(LTDETRBaseConfig.ViTTiny):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov3/vitt16"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTTiny
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTTiny
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vitt16plus-ltdetr-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITT16PLUS_COCO_URL,
                sha256=_DINOV3_VITT16PLUS_COCO_SHA256,
            ),
        ),
        "dinov3/vitt16plus-ltdetr",
    )
    class DINOv3ViTTinyPlus(LTDETRBaseConfig.ViTTinyPlus):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov3/vitt16plus"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTTinyPlus
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTTinyPlus
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov3/vits16-ltdetr-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV3_VITS16_COCO_URL,
                sha256=_DINOV3_VITS16_COCO_SHA256,
            ),
        ),
        "dinov3/vits16-ltdetr",
        "dinov3/vits16-eupe-ltdetr",
    )
    class DINOv3ViTSmall(LTDETRBaseConfig.ViTSmall):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov3/vits16"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTSmall
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTSmall
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov3/vitb16-ltdetr", "dinov3/vitb16-eupe-ltdetr")
    class DINOv3ViTBase(LTDETRBaseConfig.ViTBase):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov3/vitb16"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTBase
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTBase
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov3/vitl16-ltdetr")
    class DINOv3ViTLarge(LTDETRBaseConfig.ViTLarge):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov3/vitl16"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTLarge
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTLarge
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov2/_vittest14-ltdetr")
    class ViTTest(LTDETRBaseConfig.ViTTest):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov2/_vittest14"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTTest
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTTest
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov2/vits14-ltdetr")
    class DINOv2ViTSmall(LTDETRBaseConfig.ViTSmall):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov2/vits14"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTSmall
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTSmall
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov2/vits14-notpretrained-ltdetr")
    class DINOv2ViTSmallNotPretrained(DINOv2ViTSmall):
        backbone_name: str = "dinov2/vits14-notpretrained"

    @LTDETR_MODEL_REGISTRY.register(
        ModelAlias(
            name="dinov2/vits14-noreg-ltdetr-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_DINOV2_VITS14_NOREG_COCO_URL,
                sha256=_DINOV2_VITS14_NOREG_COCO_SHA256,
            ),
        ),
        "dinov2/vits14-noreg-ltdetr",
    )
    class DINOv2ViTSmallNoRegistersLegacy(
        LTDETRBaseConfig.DINOv2ViTSmallNoRegistersLegacy
    ):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov2/vits14-noreg"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.DINOv2ViTSmallNoRegistersLegacy
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.DINOv2ViTSmallNoRegistersLegacy
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0}
        )

    @LTDETR_MODEL_REGISTRY.register(
        "dinov2/vitb14-ltdetr",
    )
    class DINOv2ViTBase(LTDETRBaseConfig.ViTBase):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov2/vitb14"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTBase
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTBase
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitb14-notpretrained-ltdetr")
    class DINOv2ViTBaseNotPretrained(DINOv2ViTBase):
        backbone_name: str = "dinov2/vitb14-notpretrained"

    @LTDETR_MODEL_REGISTRY.register(
        "dinov2/vitl14-ltdetr",
    )
    class DINOv2ViTLarge(LTDETRBaseConfig.ViTLarge):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov2/vitl14"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTLarge
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTLarge
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitl14-notpretrained-ltdetr")
    class DINOv2ViTLargeNotPretrained(DINOv2ViTLarge):
        backbone_name: str = "dinov2/vitl14-notpretrained"

    @LTDETR_MODEL_REGISTRY.register(
        "dinov2/vitg14-ltdetr",
    )
    class DINOv2ViTGiant(LTDETRBaseConfig.ViTGiant):
        version: Literal["v1"] = "v1"
        backbone_name: str = "dinov2/vitg14"
        transformer: RTDETRTransformerv2Config | DFINETransformerConfig = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTGiant
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTGiant
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitg14-notpretrained-ltdetr")
    class DINOv2ViTGiantNotPretrained(DINOv2ViTGiant):
        backbone_name: str = "dinov2/vitg14-notpretrained"


class LTDETRFastViTConfigRegistry(ConfigsNamespace):
    @LTDETR_MODEL_REGISTRY.register("fastvit/fastvit_t8-ltdetr")
    class FastViTT8(LTDETRFastViTBaseConfig.Nano):
        version: Literal["v2"] = "v2"
        backbone_name: str = "fastvit/fastvit_t8-dist-in1k"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register("fastvit/fastvit_t12-ltdetr")
    class FastViTT12(LTDETRFastViTBaseConfig.Tiny):
        version: Literal["v2"] = "v2"
        backbone_name: str = "fastvit/fastvit_t12-dist-in1k"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register("fastvit/fastvit_s12-ltdetr")
    class FastViTS12(LTDETRFastViTBaseConfig.Small):
        version: Literal["v2"] = "v2"
        backbone_name: str = "fastvit/fastvit_s12-dist-in1k"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register("fastvit/fastvit_sa12-ltdetr")
    class FastViTSA12(LTDETRFastViTBaseConfig.Small):
        version: Literal["v2"] = "v2"
        backbone_name: str = "fastvit/fastvit_sa12-dist-in1k"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register("fastvit/fastvit_sa24-ltdetr")
    class FastViTSA24(LTDETRFastViTBaseConfig.Medium):
        version: Literal["v2"] = "v2"
        backbone_name: str = "fastvit/fastvit_sa24-dist-in1k"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register("fastvit/fastvit_sa36-ltdetr")
    class FastViTSA36(LTDETRFastViTBaseConfig.Large):
        version: Literal["v2"] = "v2"
        backbone_name: str = "fastvit/fastvit_sa36-dist-in1k"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register("fastvit/fastvit_ma36-ltdetr")
    class FastViTMA36(LTDETRFastViTBaseConfig.Large):
        version: Literal["v2"] = "v2"
        backbone_name: str = "fastvit/fastvit_ma36-dist-in1k"
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRFastViTHybridEncoderConfig.LargeMA36
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)


class LTDETRv2ConfigRegistry(ConfigsNamespace):
    @LTDETR_MODEL_REGISTRY.register(
        "edgecrafter/ecvitt-ltdetr",
        ModelAlias(
            name="edgecrafter/ecvitt-ltdetr-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_ECVITT_COCO_URL,
                sha256=_ECVITT_COCO_SHA256,
            ),
        ),
        "ltdetrv2-s",
        ModelAlias(
            name="ltdetrv2-s-coco",
            downloadable_checkpoint=DownloadableCheckpoint(
                url=_ECVITT_COCO_URL,
                sha256=_ECVITT_COCO_SHA256,
            ),
        ),
    )
    class EdgeCrafterECViTTiny(LTDETRBaseConfig.ViTTiny):
        version: Literal["v2"] = "v2"
        backbone_name: str = "edgecrafter/ecvitt"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTiny
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register(
        "_ltdetrv2-s-notpretrained", include_in_model_list=False
    )
    class EdgeCrafterECViTTinyNotPretrained(EdgeCrafterECViTTiny):
        backbone_name: str = "edgecrafter/_ecvitt-notpretrained"

    @LTDETR_MODEL_REGISTRY.register(
        "edgecrafter/ecvittplus-ltdetr",
        "ltdetrv2-m",
    )
    class EdgeCrafterECViTTinyPlus(LTDETRBaseConfig.ViTTinyPlus):
        version: Literal["v2"] = "v2"
        backbone_name: str = "edgecrafter/ecvittplus"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTinyPlus
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register(
        "edgecrafter/ecvits-ltdetr",
        "ltdetrv2-l",
    )
    class EdgeCrafterECViTSmall(LTDETRBaseConfig.ViTTinyPlus):
        version: Literal["v2"] = "v2"
        backbone_name: str = "edgecrafter/ecvits"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTinyPlus
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register(
        "edgecrafter/ecvitsplus-ltdetr",
        "ltdetrv2-x",
    )
    class EdgeCrafterECViTSmallPlus(LTDETRBaseConfig.ViTTinyPlus):
        version: Literal["v2"] = "v2"
        backbone_name: str = "edgecrafter/ecvitsplus"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTinyPlus
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )
