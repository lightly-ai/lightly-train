#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Literal, Type

from pydantic import Field

from lightly_train._configs.config import ConfigsNamespace, PydanticConfig

logger = logging.getLogger(__name__)

_LTDETRDecoderName = Literal["rtdetrv2", "dfine"]


class ModelRegistry:
    def __init__(self):
        # Maps an alias (str) to the specific dataclass Type
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
            raise KeyError(f"No model configuration registered under the alias '{alias}'.")
        return self._registry[alias]

    def list_aliases(self) -> dict[str, str]:
        """Returns a mapping of current aliases to their class names for debugging."""
        return {alias: cls.__name__ for alias, cls in self._registry.items()}

# Create singleton instance of the registry to be used across the module.
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
        conv_inplane: int = 16
        hidden_dim: int = 192

    class ViTTPlus(RTDETRBackboneWrapperConfig):
        interaction_indexes: list[int] = [3, 7, 11]
        finetune: bool = True
        use_sta: bool = True
        conv_inplane: int = 16
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
        conv_inplane: int = 64
        hidden_dim: int = 1536


class RTDETRPostProcessorConfig(PydanticConfig):
    num_top_queries: int = 300


class ObjectDetectionConfig(PydanticConfig):
    decoder_name: _LTDETRDecoderName = "dfine"
    hybrid_encoder: HybridEncoderConfig
    rtdetr_transformer: RTDETRTransformerv2Config
    dfine_transformer: DFINETransformerConfig
    rtdetr_postprocessor: RTDETRPostProcessorConfig

    def resolve_auto(self, patch_size: int | None) -> None:
        wrapper = getattr(self, "backbone_wrapper", None)
        if wrapper is not None:
            wrapper.resolve_auto(patch_size=patch_size)
        self.hybrid_encoder.resolve_auto(patch_size=patch_size)
        self.rtdetr_transformer.resolve_auto(patch_size=patch_size)
        self.dfine_transformer.resolve_auto(patch_size=patch_size)


class LTDETRObjectDetectionConfig(ConfigsNamespace):
    LTDETR_MODEL_REGISTRY.register("convnext-large")
    class CNNLarge(ObjectDetectionConfig):
        hybrid_encoder: HybridEncoderConfig = Field(
            default_factory=LTDETRHybridEncoderConfig.CNNLarge
        )
        rtdetr_transformer: RTDETRTransformerv2Config = Field(
            default_factory=LTDETRRTDETRTransformerv2Config.CNNLarge
        )
        dfine_transformer: DFINETransformerConfig = Field(
            default_factory=LTDETRDFINETransformerConfig.CNNLarge
        )
        rtdetr_postprocessor: RTDETRPostProcessorConfig = Field(
            default_factory=RTDETRPostProcessorConfig
        )
    
    LTDETR_MODEL_REGISTRY.register("convnext-base")
    class CNNBase(ObjectDetectionConfig):
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

    LTDETR_MODEL_REGISTRY.register("convnext-small")
    class CNNSmall(ObjectDetectionConfig):
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

    LTDETR_MODEL_REGISTRY.register("convnext-tiny")
    class CNNTiny(ObjectDetectionConfig):
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

    LTDETR_MODEL_REGISTRY.register("vitt16")
    class ViTT(ObjectDetectionConfig):
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

    LTDETR_MODEL_REGISTRY.register("vitt16-plus")
    class ViTTPlus(ObjectDetectionConfig):
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

    LTDETR_MODEL_REGISTRY.register("vits16", "vits14", "_vittest14")
    class ViTS(ObjectDetectionConfig):
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

    LTDETR_MODEL_REGISTRY.register("vitb16", "vitb14")
    class ViTB(ObjectDetectionConfig):
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

    LTDETR_MODEL_REGISTRY.register("vitl16", "vitl14")
    class ViTL(ObjectDetectionConfig):
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

    LTDETR_MODEL_REGISTRY.register("vitg14")
    class ViTG(ObjectDetectionConfig):
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
