#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig
from lightly_train._configs.model_registry import ModelRegistry

# ============================================================
# HybridEncoder Configs
# ============================================================


class HybridEncoderConfig(PydanticConfig):
    in_channels: list[int]
    feat_strides: list[int]
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


class DINOv3LTDETRHybridEncoderConfig(ConfigsNamespace):
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

    class ViTT(HybridEncoderConfig):
        in_channels: list[int] = [192, 192, 192]
        feat_strides: list[int] = [8, 16, 32]
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

    class ViTTPlus(HybridEncoderConfig):
        in_channels: list[int] = [256, 256, 256]
        feat_strides: list[int] = [8, 16, 32]
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

    class ViTS(HybridEncoderConfig):
        in_channels: list[int] = [224, 224, 224]
        feat_strides: list[int] = [8, 16, 32]
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

    class ViTB(HybridEncoderConfig):
        in_channels: list[int] = [768, 768, 768]
        feat_strides: list[int] = [8, 16, 32]
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

    class ViTL(HybridEncoderConfig):
        in_channels: list[int] = [1024, 1024, 1024]
        feat_strides: list[int] = [8, 16, 32]
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


# ============================================================
# RTDETRTransformerv2 Configs
# ============================================================


class RTDETRTransformerv2Config(PydanticConfig):
    decoder_name: Literal["rtdetrv2"] = "rtdetrv2"
    feat_channels: list[int] = [256, 256, 256]
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

    def resolve_auto(self, patch_size: int | None) -> None:
        patch_size = patch_size or 16
        if self.feat_strides == "auto":
            self.feat_strides = [
                int(patch_size * (2 ** (i - 1))) for i in range(self.num_levels)
            ]


class DINOv3LTDETRRTDETRTransformerv2Config(ConfigsNamespace):
    class CNNTiny(RTDETRTransformerv2Config):
        feat_channels: list[int] = [384, 384, 384]

    class CNNSmall(RTDETRTransformerv2Config):
        feat_channels: list[int] = [384, 384, 384]

    class CNNBase(RTDETRTransformerv2Config):
        feat_channels: list[int] = [384, 384, 384]

    class CNNLarge(RTDETRTransformerv2Config):
        feat_channels: list[int] = [384, 384, 384]

    class ViTT(RTDETRTransformerv2Config):
        feat_channels: list[int] = [192, 192, 192]
        hidden_dim: int = 192
        num_layers: int = 4
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 512

    class ViTTPlus(RTDETRTransformerv2Config):
        feat_channels: list[int] = [256, 256, 256]
        hidden_dim: int = 256
        num_layers: int = 4
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 512

    class ViTS(RTDETRTransformerv2Config):
        feat_channels: list[int] = [224, 224, 224]
        hidden_dim: int = 224
        num_layers: int = 4
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 1792

    class ViTB(RTDETRTransformerv2Config):
        feat_channels: list[int] = [768, 768, 768]
        hidden_dim: int = 768
        num_layers: int = 4
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 6144

    class ViTL(RTDETRTransformerv2Config):
        feat_channels: list[int] = [1024, 1024, 1024]
        hidden_dim: int = 1024
        num_layers: int = 4
        num_points: list[int] = [3, 6, 3]
        dim_feedforward: int = 8192


# ============================================================
# DFINETransformer Configs
# ============================================================


class DFINETransformerConfig(PydanticConfig):
    decoder_name: Literal["dfine"] = "dfine"
    feat_channels: list[int] = [256, 256, 256]
    feat_strides: list[int] | Literal["auto"] = "auto"
    hidden_dim: int = 256
    num_levels: int = 3
    num_layers: int = 6
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


class DINOv3LTDETRDFINETransformerConfig(ConfigsNamespace):
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

    class ViTT(DFINETransformerConfig):
        feat_channels: list[int] = [192, 192, 192]
        hidden_dim: int = 192
        num_layers: int = 4
        dim_feedforward: int = 512

    class ViTTPlus(DFINETransformerConfig):
        feat_channels: list[int] = [256, 256, 256]
        hidden_dim: int = 256
        num_layers: int = 4
        dim_feedforward: int = 512

    class ViTS(DFINETransformerConfig):
        feat_channels: list[int] = [224, 224, 224]
        hidden_dim: int = 224
        num_layers: int = 4
        dim_feedforward: int = 1792

    class ViTB(DFINETransformerConfig):
        feat_channels: list[int] = [768, 768, 768]
        hidden_dim: int = 768
        num_layers: int = 4
        dim_feedforward: int = 6144

    class ViTL(DFINETransformerConfig):
        feat_channels: list[int] = [1024, 1024, 1024]
        hidden_dim: int = 1024
        num_layers: int = 4
        dim_feedforward: int = 8192


# ============================================================
# BackboneWrapper Configs (ViT only)
# ============================================================


class RTDETRBackboneWrapperConfig(PydanticConfig):
    interaction_indexes: list[int]
    finetune: bool
    conv_inplane: int
    hidden_dim: int


class DINOv3LTDETRBackboneWrapperConfig(ConfigsNamespace):
    class ViTT(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        conv_inplane: int = 16
        hidden_dim: int = 192

    class ViTTPlus(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        conv_inplane: int = 16
        hidden_dim: int = 256

    class ViTS(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        conv_inplane: int = 32
        hidden_dim: int = 224

    class ViTB(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        conv_inplane: int = 64
        hidden_dim: int = 768

    class ViTL(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [11, 17, 23]
        finetune: bool = True
        conv_inplane: int = 64
        hidden_dim: int = 1024


# ============================================================
# RTDETRPostProcessor Config
# ============================================================


class RTDETRPostProcessorConfig(PydanticConfig):
    num_top_queries: int = 300


# ============================================================
# DetectorConfig — top-level, with discriminated union transformer
# ============================================================


class DetectorConfig(PydanticConfig):
    hybrid_encoder: HybridEncoderConfig
    transformer: Annotated[
        RTDETRTransformerv2Config | DFINETransformerConfig,
        Field(discriminator="decoder_name"),
    ]
    rtdetr_postprocessor: RTDETRPostProcessorConfig
    backbone_wrapper: RTDETRBackboneWrapperConfig | None = None

    def resolve_auto(self, patch_size: int | None) -> None:
        self.transformer.resolve_auto(patch_size=patch_size)


# ============================================================
# DINOv3LTDETRBaseConfig — pre-assembled DetectorConfig variants
# ============================================================


class DINOv3LTDETRBaseConfig(ConfigsNamespace):
    class CNNTiny(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=DINOv3LTDETRHybridEncoderConfig.CNNTiny
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )

    class CNNSmall(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=DINOv3LTDETRHybridEncoderConfig.CNNSmall
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )

    class CNNBase(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=DINOv3LTDETRHybridEncoderConfig.CNNBase
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )

    class CNNLarge(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=DINOv3LTDETRHybridEncoderConfig.CNNLarge
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )

    class ViTT(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=DINOv3LTDETRHybridEncoderConfig.ViTT
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = Field(
            default_factory=DINOv3LTDETRBackboneWrapperConfig.ViTT
        )

    class ViTTPlus(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=DINOv3LTDETRHybridEncoderConfig.ViTTPlus
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = Field(
            default_factory=DINOv3LTDETRBackboneWrapperConfig.ViTTPlus
        )

    class ViTS(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=DINOv3LTDETRHybridEncoderConfig.ViTS
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = Field(
            default_factory=DINOv3LTDETRBackboneWrapperConfig.ViTS
        )

    class ViTB(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=DINOv3LTDETRHybridEncoderConfig.ViTB
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = Field(
            default_factory=DINOv3LTDETRBackboneWrapperConfig.ViTB
        )

    class ViTL(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=DINOv3LTDETRHybridEncoderConfig.ViTL
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = Field(
            default_factory=DINOv3LTDETRBackboneWrapperConfig.ViTL
        )


# ============================================================
# Model Registry
# ============================================================

DINOV3_LTDETR_MODEL_REGISTRY: ModelRegistry[DetectorConfig] = ModelRegistry()


class DINOv3LTDETRConfigRegistry(ConfigsNamespace):
    # ——— DINOv3 ConvNeXt + DFINE decoder ———

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-tiny-ltdetr",
        "dinov3/convnext-tiny-eupe-ltdetr",
    )
    class DINOv3ConvNeXtTiny(DINOv3LTDETRBaseConfig.CNNTiny):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.CNNTiny
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-small-ltdetr",
        "dinov3/convnext-small-eupe-ltdetr",
    )
    class DINOv3ConvNeXtSmall(DINOv3LTDETRBaseConfig.CNNSmall):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.CNNSmall
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-base-ltdetr",
        "dinov3/convnext-base-eupe-ltdetr",
    )
    class DINOv3ConvNeXtBase(DINOv3LTDETRBaseConfig.CNNBase):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.CNNBase
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register("dinov3/convnext-large-ltdetr")
    class DINOv3ConvNeXtLarge(DINOv3LTDETRBaseConfig.CNNLarge):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.CNNLarge
        )

    # ——— DINOv3 ViT + DFINE decoder ———

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16-ltdetr",
        "dinov3/vitt16-eupe-ltdetr",
        "dinov3/vitt16-notpretrained-ltdetr",
    )
    class DINOv3ViTT(DINOv3LTDETRBaseConfig.ViTT):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.ViTT
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-ltdetr",
        "dinov3/vitt16plus-notpretrained-ltdetr",
    )
    class DINOv3ViTTPlus(DINOv3LTDETRBaseConfig.ViTTPlus):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.ViTTPlus
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/vits16-ltdetr",
        "dinov3/vits16-eupe-ltdetr",
    )
    class DINOv3ViTS(DINOv3LTDETRBaseConfig.ViTS):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.ViTS
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitb16-ltdetr",
        "dinov3/vitb16-eupe-ltdetr",
    )
    class DINOv3ViTB(DINOv3LTDETRBaseConfig.ViTB):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.ViTB
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register("dinov3/vitl16-ltdetr")
    class DINOv3ViTL(DINOv3LTDETRBaseConfig.ViTL):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.ViTL
        )

    # ——— ECViT (EdgeCrafter) + DFINE decoder ———
    # ECViT backbones do not use the backbone_wrapper config.
    # ecvitt -> ViTT channel shape (192), ecvittplus/ecvits/ecvitsplus -> ViTTPlus (256).

    @DINOV3_LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvitt-ltdetr")
    class ECViTT(DINOv3LTDETRBaseConfig.ViTT):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.ViTT
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = None

    @DINOV3_LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvittplus-ltdetr")
    class ECViTTPlus(DINOv3LTDETRBaseConfig.ViTTPlus):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.ViTTPlus
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = None

    @DINOV3_LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvits-ltdetr")
    class ECViTS(DINOv3LTDETRBaseConfig.ViTTPlus):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.ViTTPlus
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = None

    @DINOV3_LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvitsplus-ltdetr")
    class ECViTSPlus(DINOv3LTDETRBaseConfig.ViTTPlus):
        transformer: DFINETransformerConfig = Field(
            default_factory=DINOv3LTDETRDFINETransformerConfig.ViTTPlus
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = None

    # ——— DINOv3 ConvNeXt + RTDETRv2 decoder ———
    # Internal keys (suffixed with "-rtdetrv2") used when decoder_name="rtdetrv2".

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-tiny-ltdetr-rtdetrv2",
        "dinov3/convnext-tiny-eupe-ltdetr-rtdetrv2",
    )
    class DINOv3ConvNeXtTiny_RTDETRv2(DINOv3LTDETRBaseConfig.CNNTiny):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.CNNTiny
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-small-ltdetr-rtdetrv2",
        "dinov3/convnext-small-eupe-ltdetr-rtdetrv2",
    )
    class DINOv3ConvNeXtSmall_RTDETRv2(DINOv3LTDETRBaseConfig.CNNSmall):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.CNNSmall
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-base-ltdetr-rtdetrv2",
        "dinov3/convnext-base-eupe-ltdetr-rtdetrv2",
    )
    class DINOv3ConvNeXtBase_RTDETRv2(DINOv3LTDETRBaseConfig.CNNBase):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.CNNBase
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register("dinov3/convnext-large-ltdetr-rtdetrv2")
    class DINOv3ConvNeXtLarge_RTDETRv2(DINOv3LTDETRBaseConfig.CNNLarge):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.CNNLarge
        )

    # ——— DINOv3 ViT + RTDETRv2 decoder ———

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16-ltdetr-rtdetrv2",
        "dinov3/vitt16-eupe-ltdetr-rtdetrv2",
        "dinov3/vitt16-notpretrained-ltdetr-rtdetrv2",
    )
    class DINOv3ViTT_RTDETRv2(DINOv3LTDETRBaseConfig.ViTT):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.ViTT
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-ltdetr-rtdetrv2",
        "dinov3/vitt16plus-notpretrained-ltdetr-rtdetrv2",
    )
    class DINOv3ViTTPlus_RTDETRv2(DINOv3LTDETRBaseConfig.ViTTPlus):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.ViTTPlus
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/vits16-ltdetr-rtdetrv2",
        "dinov3/vits16-eupe-ltdetr-rtdetrv2",
    )
    class DINOv3ViTS_RTDETRv2(DINOv3LTDETRBaseConfig.ViTS):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.ViTS
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitb16-ltdetr-rtdetrv2",
        "dinov3/vitb16-eupe-ltdetr-rtdetrv2",
    )
    class DINOv3ViTB_RTDETRv2(DINOv3LTDETRBaseConfig.ViTB):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.ViTB
        )

    @DINOV3_LTDETR_MODEL_REGISTRY.register("dinov3/vitl16-ltdetr-rtdetrv2")
    class DINOv3ViTL_RTDETRv2(DINOv3LTDETRBaseConfig.ViTL):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.ViTL
        )

    # ——— ECViT + RTDETRv2 decoder ———

    @DINOV3_LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvitt-ltdetr-rtdetrv2")
    class ECViTT_RTDETRv2(DINOv3LTDETRBaseConfig.ViTT):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.ViTT
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = None

    @DINOV3_LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvittplus-ltdetr-rtdetrv2")
    class ECViTTPlus_RTDETRv2(DINOv3LTDETRBaseConfig.ViTTPlus):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.ViTTPlus
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = None

    @DINOV3_LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvits-ltdetr-rtdetrv2")
    class ECViTS_RTDETRv2(DINOv3LTDETRBaseConfig.ViTTPlus):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.ViTTPlus
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = None

    @DINOV3_LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvitsplus-ltdetr-rtdetrv2")
    class ECViTSPlus_RTDETRv2(DINOv3LTDETRBaseConfig.ViTTPlus):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=DINOv3LTDETRRTDETRTransformerv2Config.ViTTPlus
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig | None = None
