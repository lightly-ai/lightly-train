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
from lightly_train._configs.model_registry import ModelRegistry
from lightly_train._models import package_helpers

logger = logging.getLogger(__name__)


LTDETR_MODEL_REGISTRY: ModelRegistry[DetectorConfig] = ModelRegistry()


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

    class ViTTinyPlus(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 1
        hidden_dim: int = 256

    class ViTSmall(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 2
        hidden_dim: int = 224

    class ViTBase(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 4
        hidden_dim: int = 768

    class ViTLarge(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [11, 17, 23]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 4
        hidden_dim: int = 1024

    class ViTGiant(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [19, 29, 39]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane_factor: int = 4
        hidden_dim: int = 1536


class LTDETRRTDETRNoSTABackboneWrapperConfig(ConfigsNamespace):
    class ViTTiny(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 1
        hidden_dim: int = 192

    class ViTTinyPlus(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 1
        hidden_dim: int = 256

    class ViTSmall(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 2
        hidden_dim: int = 224

    class ViTBase(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [5, 8, 11]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 4
        hidden_dim: int = 768

    class ViTLarge(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [11, 17, 23]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 4
        hidden_dim: int = 1024

    class ViTGiant(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [19, 29, 39]
        finetune: bool = True
        use_sta: bool = False
        conv_inplane_factor: int = 4
        hidden_dim: int = 1536


class RTDETRPostProcessorConfig(PydanticConfig):
    num_top_queries: int = 300


class DetectorConfig(PydanticConfig):
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


def resolve_transformer_config(
    *, config: DetectorConfig, decoder_name: Literal["rtdetrv2", "dfine"] | None
) -> RTDETRTransformerv2Config | DFINETransformerConfig:
    """Resolve the transformer config for fresh and legacy checkpoints."""
    resolved_decoder_name = decoder_name or config.transformer.decoder_name
    if resolved_decoder_name == config.transformer.decoder_name:
        return config.transformer

    package_name, backbone_name = package_helpers.parse_model_name(config.backbone_name)
    if package_name == "dinov2" and resolved_decoder_name == "dfine":
        raise ValueError(
            f"decoder_name='dfine' is not supported for model {config.backbone_name!r}."
        )

    if resolved_decoder_name == "rtdetrv2":
        config_namespace = LTDETRRTDETRTransformerv2Config
    else:
        config_namespace = LTDETRDFINETransformerConfig

    transformer_config_cls = {
        "convnext-tiny": config_namespace.CNNTiny,
        "convnext-small": config_namespace.CNNSmall,
        "convnext-base": config_namespace.CNNBase,
        "convnext-large": config_namespace.CNNLarge,
        "vitt16": config_namespace.ViTTiny,
        "vitt16plus": config_namespace.ViTTinyPlus,
        "vits16": config_namespace.ViTSmall,
        "vitb16": config_namespace.ViTBase,
        "vitl16": config_namespace.ViTLarge,
        "vits14": config_namespace.ViTSmall,
        "vitb14": config_namespace.ViTBase,
        "vitl14": config_namespace.ViTLarge,
        "vitg14": config_namespace.ViTGiant,
        "ecvitt": config_namespace.ViTTiny,
        "ecvittplus": config_namespace.ViTTinyPlus,
        "ecvits": config_namespace.ViTTinyPlus,
        "ecvitsplus": config_namespace.ViTTinyPlus,
    }.get(backbone_name)
    if transformer_config_cls is None:
        raise ValueError(
            f"decoder_name={resolved_decoder_name!r} is not supported for model "
            f"{config.backbone_name!r}."
        )
    return transformer_config_cls()


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

    class ViTTiny(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTTiny
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )

    class ViTTinyPlus(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTTinyPlus
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )

    class ViTSmall(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTSmall
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )

    class ViTBase(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTBase
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )

    class ViTLarge(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTLarge
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )

    class ViTGiant(DetectorConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.ViTGiant
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )


class LTDETRConfigRegistry(ConfigsNamespace):
    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-tiny-ltdetr-coco",
        "dinov3/convnext-tiny-ltdetr",
        "dinov3/convnext-tiny-eupe-ltdetr",
    )
    class DINOv3ConvNeXtTiny(LTDETRBaseConfig.CNNTiny):
        backbone_name: str = "dinov3/convnext-tiny"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNTiny
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-small-ltdetr-coco",
        "dinov3/convnext-small-ltdetr",
        "dinov3/convnext-small-eupe-ltdetr",
    )
    class DINOv3ConvNeXtSmall(LTDETRBaseConfig.CNNSmall):
        backbone_name: str = "dinov3/convnext-small"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNSmall
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-base-ltdetr-coco",
        "dinov3/convnext-base-ltdetr",
        "dinov3/convnext-base-eupe-ltdetr",
    )
    class DINOv3ConvNeXtBase(LTDETRBaseConfig.CNNBase):
        backbone_name: str = "dinov3/convnext-base"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNBase
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-large-ltdetr-coco", "dinov3/convnext-large-ltdetr"
    )
    class DINOv3ConvNeXtLarge(LTDETRBaseConfig.CNNLarge):
        backbone_name: str = "dinov3/convnext-large"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNLarge
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16-ltdetr-coco",
        "dinov3/vitt16-ltdetr",
        "dinov3/vitt16-eupe-ltdetr",
        "dinov3/vitt16-notpretrained-ltdetr",
    )
    class DINOv3ViTTiny(LTDETRBaseConfig.ViTTiny):
        backbone_name: str = "dinov3/vitt16"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTiny
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTTiny
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-ltdetr-coco", "dinov3/vitt16plus-ltdetr"
    )
    class DINOv3ViTTinyPlus(LTDETRBaseConfig.ViTTinyPlus):
        backbone_name: str = "dinov3/vitt16plus"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTinyPlus
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTTinyPlus
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vits16-ltdetr-coco", "dinov3/vits16-ltdetr", "dinov3/vits16-eupe-ltdetr"
    )
    class DINOv3ViTSmall(LTDETRBaseConfig.ViTSmall):
        backbone_name: str = "dinov3/vits16"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTSmall
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTSmall
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov3/vitb16-ltdetr", "dinov3/vitb16-eupe-ltdetr")
    class DINOv3ViTBase(LTDETRBaseConfig.ViTBase):
        backbone_name: str = "dinov3/vitb16"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTBase
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTBase
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov3/vitl16-ltdetr")
    class DINOv3ViTLarge(LTDETRBaseConfig.ViTLarge):
        backbone_name: str = "dinov3/vitl16"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTLarge
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRBackboneWrapperConfig.ViTLarge
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 16}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov2/vits14-ltdetr")
    class DINOv2ViTSmall(LTDETRBaseConfig.ViTSmall):
        backbone_name: str = "dinov2/vits14"
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTSmall
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTSmall
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitb14-ltdetr")
    class DINOv2ViTBase(LTDETRBaseConfig.ViTBase):
        backbone_name: str = "dinov2/vitb14"
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTBase
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTBase
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitl14-ltdetr")
    class DINOv2ViTLarge(LTDETRBaseConfig.ViTLarge):
        backbone_name: str = "dinov2/vitl14"
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTLarge
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTLarge
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0}
        )

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitg14-ltdetr")
    class DINOv2ViTGiant(LTDETRBaseConfig.ViTGiant):
        backbone_name: str = "dinov2/vitg14"
        transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTGiant
        )
        backbone_wrapper: RTDETRBackboneWrapperConfig = Field(
            default_factory=LTDETRRTDETRNoSTABackboneWrapperConfig.ViTGiant
        )
        backbone_args: dict[str, Any] = Field(
            default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0}
        )


class LTDETRv2ConfigRegistry(ConfigsNamespace):
    @LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvitt-ltdetr", "ltdetrv2-s")
    class EdgeCrafterECViTTiny(LTDETRBaseConfig.ViTTiny):
        backbone_name: str = "edgecrafter/ecvitt"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTiny
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvittplus-ltdetr", "ltdetrv2-m")
    class EdgeCrafterECViTTinyPlus(LTDETRBaseConfig.ViTTinyPlus):
        backbone_name: str = "edgecrafter/ecvittplus"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTinyPlus
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvits-ltdetr", "ltdetrv2-l")
    class EdgeCrafterECViTSmall(LTDETRBaseConfig.ViTTinyPlus):
        backbone_name: str = "edgecrafter/ecvits"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTinyPlus
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvitsplus-ltdetr", "ltdetrv2-x")
    class EdgeCrafterECViTSmallPlus(LTDETRBaseConfig.ViTTinyPlus):
        backbone_name: str = "edgecrafter/ecvitsplus"
        transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTinyPlus
        )
        backbone_wrapper: ECViTBackboneWrapperConfig = Field(
            default_factory=ECViTBackboneWrapperConfig
        )
        backbone_args: dict[str, Any] = Field(default_factory=dict)
