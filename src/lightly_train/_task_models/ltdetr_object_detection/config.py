#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Any, Literal, Type

from pydantic import Field

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig

logger = logging.getLogger(__name__)

_LTDETRDecoderName = Literal["rtdetrv2", "dfine"]


class ModelRegistry:
    def __init__(self):
        # Maps an alias' to
        self._registry: dict[str, Type] = {}

    def register(self, *aliases: str):
        """
        A decorator to register a dataclass under one or multiple aliases.
        Raises a ValueError if any alias is already taken.
        """

        def decorator(cls: Type):
            for alias in aliases:
                # Enforce uniqueness
                if alias in self._registry:
                    existing_cls = self._registry[alias].__name__
                    raise ValueError(
                        f"Conflict detected! The alias '{alias}' is already registered "
                        f"to the class '{existing_cls}'."
                    )
                self._registry[alias] = cls
            return cls

        return decorator

    def get(self, alias: str) -> Type:
        """Retrieve the dataclass associated with the alias."""
        if alias not in self._registry:
            raise KeyError(
                f"No model configuration registered under the alias '{alias}'."
            )
        return self._registry[alias]

    def list_aliases(self) -> dict[str, str]:
        """Returns a mapping of current aliases to their class names for debugging."""
        return {alias: cls.__name__ for alias, cls in self._registry.items()}


# Create singleton instance of the registry to be used across the package.
LTDETR_MODEL_REGISTRY = ModelRegistry()


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
    decoder_name: _LTDETRDecoderName
    hybrid_encoder: HybridEncoderConfig
    transformer: RTDETRTransformerv2Config | DFINETransformerConfig
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
        # TODO: Discriminated Union
        rtdetr_transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNLarge
        )
        dfine_transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNLarge
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
        rtdetr_transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNBase
        )
        dfine_transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNBase
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
        rtdetr_transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNSmall
        )
        dfine_transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNSmall
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
        rtdetr_transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNTiny
        )
        dfine_transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNTiny
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
        rtdetr_transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTT
        )
        dfine_transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTT
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
        rtdetr_transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTTPlus
        )
        dfine_transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTTPlus
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
        rtdetr_transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTS
        )
        dfine_transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTS
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
        rtdetr_transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTB
        )
        dfine_transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTB
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
        rtdetr_transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTL
        )
        dfine_transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTL
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
        rtdetr_transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.ViTG
        )
        dfine_transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.ViTG
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
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-small-ltdetr-coco",
        "dinov3/convnext-small-ltdetr",
        "dinov3/convnext-small-eupe-ltdetr",
    )
    class DINOv3ConvNeXtSmall(LTDETRBaseConfig.CNNSmall):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-base-ltdetr-coco",
        "dinov3/convnext-base-ltdetr",
        "dinov3/convnext-base-eupe-ltdetr",
    )
    class DINOv3ConvNeXtBase(LTDETRBaseConfig.CNNBase):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-large-ltdetr-coco", "dinov3/convnext-large-ltdetr"
    )
    class DINOv3ConvNeXtLarge(LTDETRBaseConfig.CNNLarge):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16-ltdetr-coco", "dinov3/vitt16-ltdetr", "dinov3/vitt16-eupe-ltdetr"
    )
    class DINOv3ViTT(LTDETRBaseConfig.ViTT):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-ltdetr-coco", "dinov3/vitt16plus-ltdetr"
    )
    class DINOv3ViTTPlus(LTDETRBaseConfig.ViTTPlus):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vits16-ltdetr-coco", "dinov3/vits16-ltdetr", "dinov3/vits16-eupe-ltdetr"
    )
    class DINOv3ViTS(LTDETRBaseConfig.ViTS):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register("dinov3/vitb16-ltdetr", "dinov3/vitb16-eupe-ltdetr")
    class DINOv3ViTB(LTDETRBaseConfig.ViTB):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register("dinov3/vitl16-ltdetr")
    class DINOv3ViTL(LTDETRBaseConfig.ViTL):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vits14-ltdetr")
    class DINOv2ViTS(LTDETRBaseConfig.DINOv2ViTS):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitb14-ltdetr")
    class DINOv2ViTB(LTDETRBaseConfig.DINOv2ViTB):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitl14-ltdetr")
    class DINOv2ViTL(LTDETRBaseConfig.DINOv2ViTL):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitg14-ltdetr")
    class DINOv2ViTG(LTDETRBaseConfig.DINOv2ViTG):
        decoder_name: _LTDETRDecoderName = "rtdetrv2"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})


class LTDETRv2ConfigRegistry(ConfigsNamespace):
    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-tiny-ltdetrv2-coco",
        "dinov3/convnext-tiny-ltdetrv2",
        "dinov3/convnext-tiny-eupe-ltdetrv2",
    )
    class DINOv3ConvNeXtTiny(LTDETRBaseConfig.CNNTiny):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-small-ltdetrv2-coco",
        "dinov3/convnext-small-ltdetrv2",
        "dinov3/convnext-small-eupe-ltdetrv2",
    )
    class DINOv3ConvNeXtSmall(LTDETRBaseConfig.CNNSmall):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-base-ltdetrv2-coco",
        "dinov3/convnext-base-ltdetrv2",
        "dinov3/convnext-base-eupe-ltdetrv2",
    )
    class DINOv3ConvNeXtBase(LTDETRBaseConfig.CNNBase):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/convnext-large-ltdetrv2-coco",
        "dinov3/convnext-large-ltdetrv2",
    )
    class DINOv3ConvNeXtLarge(LTDETRBaseConfig.CNNLarge):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=dict)

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16-ltdetrv2-coco",
        "dinov3/vitt16-ltdetrv2",
        "dinov3/vitt16-eupe-ltdetrv2",
    )
    class DINOv3ViTT(LTDETRBaseConfig.ViTT):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitt16plus-ltdetrv2-coco",
        "dinov3/vitt16plus-ltdetrv2",
    )
    class DINOv3ViTTPlus(LTDETRBaseConfig.ViTTPlus):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vits16-ltdetrv2-coco",
        "dinov3/vits16-ltdetrv2",
        "dinov3/vits16-eupe-ltdetrv2",
    )
    class DINOv3ViTS(LTDETRBaseConfig.ViTS):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register(
        "dinov3/vitb16-ltdetrv2",
        "dinov3/vitb16-eupe-ltdetrv2",
    )
    class DINOv3ViTB(LTDETRBaseConfig.ViTB):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register("dinov3/vitl16-ltdetrv2")
    class DINOv3ViTL(LTDETRBaseConfig.ViTL):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 16})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vits14-ltdetrv2")
    class DINOv2ViTS(LTDETRBaseConfig.DINOv2ViTS):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitb14-ltdetrv2")
    class DINOv2ViTB(LTDETRBaseConfig.DINOv2ViTB):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitl14-ltdetrv2")
    class DINOv2ViTL(LTDETRBaseConfig.DINOv2ViTL):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})

    @LTDETR_MODEL_REGISTRY.register("dinov2/vitg14-ltdetrv2")
    class DINOv2ViTG(LTDETRBaseConfig.DINOv2ViTG):
        decoder_name: _LTDETRDecoderName = "dfine"
        backbone_args: dict[str, Any] = Field(default_factory=lambda: {"patch_size": 14, "drop_path_rate": 0.0})