#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from pydantic import Field

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig
from lightly_train._configs.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


LTDETR_MODEL_REGISTRY: ModelRegistry[PydanticConfig] = ModelRegistry()


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

    class ViTG(HybridEncoderConfig):
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

    class ViTG(RTDETRTransformerv2Config):
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

    class ViTG(DFINETransformerConfig):
        feat_channels: list[int] = [1536, 1536, 1536]
        hidden_dim: int = 1536
        num_layers: int = 4
        dim_feedforward: int = 12288


class CNNBackboneWrapperConfig(PydanticConfig):
    finetune: bool = True

    def resolve_auto(self, patch_size: int | None) -> None:
        """No-op since CNNs don't have a patch size."""
        pass


class RTDETRBackboneWrapperConfig(PydanticConfig):
    interaction_indexes: list[int]
    finetune: bool
    use_sta: bool
    conv_inplane: int | Literal["auto"] = "auto"
    conv_inplane_factor: int = 2
    hidden_dim: int

    def resolve_auto(self, patch_size: int | None) -> None:
        patch_size = patch_size or 16
        if self.conv_inplane == "auto":
            self.conv_inplane = self.conv_inplane_factor * patch_size


class LTDETRRTDETRBackboneWrapperConfig(ConfigsNamespace):
    class ViTT(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 1
        hidden_dim: int = 192

    class ViTTPlus(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 1
        hidden_dim: int = 256

    class ViTS(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 2
        hidden_dim: int = 224

    class ViTB(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 4
        hidden_dim: int = 768

    class ViTL(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [11, 17, 23]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 4
        hidden_dim: int = 1024

    class ViTG(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [19, 29, 39]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 4
        hidden_dim: int = 1536


class LTDETRRTDETRNoSTABackboneWrapperConfig(ConfigsNamespace):
    class ViTT(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 1
        hidden_dim: int = 192

    class ViTTPlus(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 1
        hidden_dim: int = 256

    class ViTS(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 2
        hidden_dim: int = 224

    class ViTB(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 4
        hidden_dim: int = 768

    class ViTL(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [11, 17, 23]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 4
        hidden_dim: int = 1024

    class ViTG(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [19, 29, 39]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 4
        hidden_dim: int = 1536


class RTDETRPostProcessorConfig(PydanticConfig):
    num_top_queries: int = 300


class DetectorConfig(PydanticConfig):
    hybrid_encoder: HybridEncoderConfig
    transformer: Annotated[
        RTDETRTransformerv2Config | DFINETransformerConfig,
        Field(discriminator="decoder_name"),
    ]
    rtdetr_postprocessor: RTDETRPostProcessorConfig
    backbone_wrapper: RTDETRBackboneWrapperConfig | CNNBackboneWrapperConfig

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
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: CNNBackboneWrapperConfig = Field(
            default_factory=CNNBackboneWrapperConfig
        )
        backbone_args: dict[str, Any]

    class CNNBase(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.CNNBase
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: CNNBackboneWrapperConfig = Field(
            default_factory=CNNBackboneWrapperConfig
        )
        backbone_args: dict[str, Any]

    class CNNSmall(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.CNNSmall
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: CNNBackboneWrapperConfig = Field(
            default_factory=CNNBackboneWrapperConfig
        )
        backbone_args: dict[str, Any]

    class CNNTiny(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.CNNTiny
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: CNNBackboneWrapperConfig = Field(
            default_factory=CNNBackboneWrapperConfig
        )
        backbone_args: dict[str, Any]

    class ViTT(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTT
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTT
        )
        backbone_args: dict[str, Any]

    class ViTTPlus(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTTPlus
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTTPlus
        )
        backbone_args: dict[str, Any]

    class ViTS(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTS
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTS
        )
        backbone_args: dict[str, Any]

    class ViTB(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTB
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTB
        )
        backbone_args: dict[str, Any]

    class ViTL(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTL
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTL
        )
        backbone_args: dict[str, Any]

    class ViTG(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTG
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTG
        )
        backbone_args: dict[str, Any]

    class DINOv2ViTS(ViTS):
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTS
        )

    class DINOv2ViTB(ViTB):
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTB
        )

    class DINOv2ViTL(ViTL):
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTL
        )

    class DINOv2ViTG(ViTG):
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTG
        )


class LTDETRConfigRegistry(ConfigsNamespace):
    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-tiny-ltdetr-coco",
        "dinov3/convnext-tiny-ltdetr",
        "dinov3/convnext-tiny-eupe-ltdetr",
    )
    class DINOv3ConvNeXtTiny(LTDETRBaseConfig.CNNTiny):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNTiny
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-small-ltdetr-coco",
        "dinov3/convnext-small-ltdetr",
        "dinov3/convnext-small-eupe-ltdetr",
    )
    class DINOv3ConvNeXtSmall(LTDETRBaseConfig.CNNSmall):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNSmall
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-base-ltdetr-coco",
        "dinov3/convnext-base-ltdetr",
        "dinov3/convnext-base-eupe-ltdetr",
    )
    class DINOv3ConvNeXtBase(LTDETRBaseConfig.CNNBase):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNBase
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-large-ltdetr-coco", "dinov3/convnext-large-ltdetr"
    )
    class DINOv3ConvNeXtLarge(LTDETRBaseConfig.CNNLarge):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNLarge
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16-ltdetr-coco", "dinov3/vitt16-ltdetr", "dinov3/vitt16-eupe-ltdetr"
    )
    class DINOv3ViTT(LTDETRBaseConfig.ViTT):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTT
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-ltdetr-coco", "dinov3/vitt16plus-ltdetr"
    )
    class DINOv3ViTTPlus(LTDETRBaseConfig.ViTTPlus):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTTPlus
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vits16-ltdetr-coco", "dinov3/vits16-ltdetr", "dinov3/vits16-eupe-ltdetr"
    )
    class DINOv3ViTS(LTDETRBaseConfig.ViTS):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTS
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register("dinov3/vitb16-ltdetr", "dinov3/vitb16-eupe-ltdetr")
    class DINOv3ViTB(LTDETRBaseConfig.ViTB):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTB
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register("dinov3/vitl16-ltdetr")
    class DINOv3ViTL(LTDETRBaseConfig.ViTL):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTL
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vits14-ltdetr")
    class DINOv2ViTS(LTDETRBaseConfig.DINOv2ViTS):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTS
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitb14-ltdetr")
    class DINOv2ViTB(LTDETRBaseConfig.DINOv2ViTB):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTB
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitl14-ltdetr")
    class DINOv2ViTL(LTDETRBaseConfig.DINOv2ViTL):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTL
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitg14-ltdetr")
    class DINOv2ViTG(LTDETRBaseConfig.DINOv2ViTG):
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTG
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})


class LTDETRv2ConfigRegistry(ConfigsNamespace):
    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-tiny-ltdetrv2-coco",
        "dinov3/convnext-tiny-ltdetrv2",
        "dinov3/convnext-tiny-eupe-ltdetrv2",
    )
    class DINOv3ConvNeXtTiny(LTDETRBaseConfig.CNNTiny):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNTiny
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-small-ltdetrv2-coco",
        "dinov3/convnext-small-ltdetrv2",
        "dinov3/convnext-small-eupe-ltdetrv2",
    )
    class DINOv3ConvNeXtSmall(LTDETRBaseConfig.CNNSmall):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNSmall
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-base-ltdetrv2-coco",
        "dinov3/convnext-base-ltdetrv2",
        "dinov3/convnext-base-eupe-ltdetrv2",
    )
    class DINOv3ConvNeXtBase(LTDETRBaseConfig.CNNBase):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNBase
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-large-ltdetrv2-coco",
        "dinov3/convnext-large-ltdetrv2",
    )
    class DINOv3ConvNeXtLarge(LTDETRBaseConfig.CNNLarge):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNLarge
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16-ltdetrv2-coco",
        "dinov3/vitt16-ltdetrv2",
        "dinov3/vitt16-eupe-ltdetrv2",
    )
    class DINOv3ViTT(LTDETRBaseConfig.ViTT):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTT
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-ltdetrv2-coco",
        "dinov3/vitt16plus-ltdetrv2",
    )
    class DINOv3ViTTPlus(LTDETRBaseConfig.ViTTPlus):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTPlus
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vits16-ltdetrv2-coco",
        "dinov3/vits16-ltdetrv2",
        "dinov3/vits16-eupe-ltdetrv2",
    )
    class DINOv3ViTS(LTDETRBaseConfig.ViTS):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTS
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitb16-ltdetrv2",
        "dinov3/vitb16-eupe-ltdetrv2",
    )
    class DINOv3ViTB(LTDETRBaseConfig.ViTB):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTB
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register("dinov3/vitl16-ltdetrv2")
    class DINOv3ViTL(LTDETRBaseConfig.ViTL):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTL
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vits14-ltdetrv2")
    class DINOv2ViTS(LTDETRBaseConfig.DINOv2ViTS):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTS
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitb14-ltdetrv2")
    class DINOv2ViTB(LTDETRBaseConfig.DINOv2ViTB):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTB
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitl14-ltdetrv2")
    class DINOv2ViTL(LTDETRBaseConfig.DINOv2ViTL):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTL
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitg14-ltdetrv2")
    class DINOv2ViTG(LTDETRBaseConfig.DINOv2ViTG):
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTG
        )
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})