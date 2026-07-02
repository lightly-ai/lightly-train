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
from lightly_train._configs.model_registry import ModelRegistry

LTDETR_MODEL_REGISTRY: ModelRegistry[SegmentorConfig] = ModelRegistry()


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
    @LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvitt-ltdetr-seg", "ltdetrv2-seg-s")
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

    @LTDETR_MODEL_REGISTRY.register(
        "edgecrafter/ecvittplus-ltdetr-seg", "ltdetrv2-seg-m"
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

    @LTDETR_MODEL_REGISTRY.register("edgecrafter/ecvits-ltdetr-seg", "ltdetrv2-seg-l")
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

    @LTDETR_MODEL_REGISTRY.register(
        "edgecrafter/ecvitsplus-ltdetr-seg", "ltdetrv2-seg-x"
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
