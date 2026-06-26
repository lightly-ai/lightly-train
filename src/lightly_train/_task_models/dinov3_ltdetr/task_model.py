#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal, NoReturn, cast

import torch
from PIL.Image import Image as PILImage
from pydantic import Field
from torch import Tensor, nn
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import Self

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._models import package_helpers
from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper
from lightly_train._models.dinov3.dinov3_package import DINOV3_PACKAGE
from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt
from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper
from lightly_train._models.ecvit.ecvit import ECViTModelWrapper
from lightly_train._models.ecvit.ecvit_package import EDGE_CRAFTER_PACKAGE
from lightly_train._task_models.dinov3_ltdetr_object_detection.dinov3_convnext_wrapper import (
    DINOv3ConvNextWrapper,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.dinov3_vit_wrapper import (
    DINOv3STAs,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.ecvit_vit_wrapper import (
    ECViTBackboneWrapper,
)
from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

_LTDETRDecoderName = Literal["rtdetrv2", "dfine"]


class _HybridEncoderConfig(PydanticConfig):
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


class _HybridEncoderTinyConfig(_HybridEncoderConfig):
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


class _HybridEncoderSmallConfig(_HybridEncoderConfig):
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


class _HybridEncoderBaseConfig(_HybridEncoderConfig):
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


class _HybridEncoderLargeConfig(_HybridEncoderConfig):
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


class _HybridEncoderViTTConfig(_HybridEncoderConfig):
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


class _HybridEncoderViTTPlusConfig(_HybridEncoderConfig):
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


class _HybridEncoderViTSConfig(_HybridEncoderConfig):
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


class _HybridEncoderViTBConfig(_HybridEncoderConfig):
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


class _HybridEncoderViTLConfig(_HybridEncoderConfig):
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


class _RTDETRTransformerv2Config(PydanticConfig):
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


class _RTDETRTransformerv2TinyConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [384, 384, 384]


class _RTDETRTransformerv2SmallConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [384, 384, 384]


class _RTDETRTransformerv2BaseConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [384, 384, 384]


class _RTDETRTransformerv2LargeConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [384, 384, 384]


class _RTDETRTransformerv2ViTTConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [192, 192, 192]
    hidden_dim: int = 192
    num_layers: int = 4
    num_points: list[int] = [3, 6, 3]
    dim_feedforward: int = 512


class _RTDETRTransformerv2ViTTPlusConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [256, 256, 256]
    hidden_dim: int = 256
    num_layers: int = 4
    num_points: list[int] = [3, 6, 3]
    dim_feedforward: int = 512


class _RTDETRTransformerv2ViTSConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [224, 224, 224]
    hidden_dim: int = 224
    num_layers: int = 4
    num_points: list[int] = [3, 6, 3]
    dim_feedforward: int = 1792


class _RTDETRTransformerv2ViTBConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [768, 768, 768]
    hidden_dim: int = 768
    num_layers: int = 4
    num_points: list[int] = [3, 6, 3]
    dim_feedforward: int = 6144


class _RTDETRTransformerv2ViTLConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [1024, 1024, 1024]
    hidden_dim: int = 1024
    num_layers: int = 4
    num_points: list[int] = [3, 6, 3]
    dim_feedforward: int = 8192


class _DFINETransformerConfig(PydanticConfig):
    feat_channels: list[int] = [256, 256, 256]
    feat_strides: list[int] = [8, 16, 32]
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


class _DFINETransformerConvNextConfig(_DFINETransformerConfig):
    feat_channels: list[int] = [384, 384, 384]


class _DFINETransformerConvNextTinyConfig(_DFINETransformerConvNextConfig):
    num_layers: int = 3


class _DFINETransformerConvNextSmallConfig(_DFINETransformerConvNextConfig):
    num_layers: int = 3


class _DFINETransformerConvNextBaseConfig(_DFINETransformerConvNextConfig):
    num_layers: int = 4


class _DFINETransformerConvNextLargeConfig(_DFINETransformerConvNextConfig):
    reg_scale: float = 8.0


class _DFINETransformerViTTConfig(_DFINETransformerConfig):
    feat_channels: list[int] = [192, 192, 192]
    hidden_dim: int = 192
    num_layers: int = 4
    dim_feedforward: int = 512


class _DFINETransformerViTTPlusConfig(_DFINETransformerConfig):
    feat_channels: list[int] = [256, 256, 256]
    hidden_dim: int = 256
    num_layers: int = 4
    dim_feedforward: int = 512


class _DFINETransformerViTSConfig(_DFINETransformerConfig):
    feat_channels: list[int] = [224, 224, 224]
    hidden_dim: int = 224
    num_layers: int = 4
    dim_feedforward: int = 1792


class _DFINETransformerViTBConfig(_DFINETransformerConfig):
    feat_channels: list[int] = [768, 768, 768]
    hidden_dim: int = 768
    num_layers: int = 4
    dim_feedforward: int = 6144


class _DFINETransformerViTLConfig(_DFINETransformerConfig):
    feat_channels: list[int] = [1024, 1024, 1024]
    hidden_dim: int = 1024
    num_layers: int = 4
    dim_feedforward: int = 8192


class _RTDETRBackboneWrapperViTTConfig(PydanticConfig):
    interaction_indexes: list[int] = [3, 7, 11]
    finetune: bool = True
    conv_inplane: int = 16
    hidden_dim: int = 192


class _RTDETRBackboneWrapperViTTPlusConfig(PydanticConfig):
    interaction_indexes: list[int] = [3, 7, 11]
    finetune: bool = True
    conv_inplane: int = 16
    hidden_dim: int = 256


class _RTDETRBackboneWrapperViTSConfig(PydanticConfig):
    interaction_indexes: list[int] = [5, 8, 11]
    finetune: bool = True
    conv_inplane: int = 32
    hidden_dim: int = 224


class _RTDETRBackboneWrapperViTBConfig(PydanticConfig):
    interaction_indexes: list[int] = [5, 8, 11]
    finetune: bool = True
    conv_inplane: int = 64
    hidden_dim: int = 768


class _RTDETRBackboneWrapperViTLConfig(PydanticConfig):
    interaction_indexes: list[int] = [11, 17, 23]
    finetune: bool = True
    conv_inplane: int = 64
    hidden_dim: int = 1024


class _RTDETRPostProcessorConfig(PydanticConfig):
    num_top_queries: int = 300


class _DINOv3LTDETRConfig(PydanticConfig):
    decoder_name: _LTDETRDecoderName = "dfine"
    hybrid_encoder: _HybridEncoderConfig
    rtdetr_transformer: _RTDETRTransformerv2Config
    dfine_transformer: _DFINETransformerConfig
    rtdetr_postprocessor: _RTDETRPostProcessorConfig

    def resolve_auto(self, patch_size: int | None) -> None:
        self.rtdetr_transformer.resolve_auto(patch_size=patch_size)


class _DINOv3LTDETRLargeConfig(_DINOv3LTDETRConfig):
    hybrid_encoder: _HybridEncoderLargeConfig = Field(
        default_factory=_HybridEncoderLargeConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2LargeConfig = Field(
        default_factory=_RTDETRTransformerv2LargeConfig
    )
    dfine_transformer: _DFINETransformerConvNextLargeConfig = Field(
        default_factory=_DFINETransformerConvNextLargeConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )


class _DINOv3LTDETRBaseConfig(_DINOv3LTDETRConfig):
    hybrid_encoder: _HybridEncoderBaseConfig = Field(
        default_factory=_HybridEncoderBaseConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2BaseConfig = Field(
        default_factory=_RTDETRTransformerv2BaseConfig
    )
    dfine_transformer: _DFINETransformerConvNextBaseConfig = Field(
        default_factory=_DFINETransformerConvNextBaseConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )


class _DINOv3LTDETRSmallConfig(_DINOv3LTDETRConfig):
    hybrid_encoder: _HybridEncoderSmallConfig = Field(
        default_factory=_HybridEncoderSmallConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2SmallConfig = Field(
        default_factory=_RTDETRTransformerv2SmallConfig
    )
    dfine_transformer: _DFINETransformerConvNextSmallConfig = Field(
        default_factory=_DFINETransformerConvNextSmallConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )


class _DINOv3LTDETRTinyConfig(_DINOv3LTDETRConfig):
    hybrid_encoder: _HybridEncoderTinyConfig = Field(
        default_factory=_HybridEncoderTinyConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2TinyConfig = Field(
        default_factory=_RTDETRTransformerv2TinyConfig
    )
    dfine_transformer: _DFINETransformerConvNextTinyConfig = Field(
        default_factory=_DFINETransformerConvNextTinyConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )


class _DINOv3LTDETRViTTConfig(_DINOv3LTDETRConfig):
    hybrid_encoder: _HybridEncoderViTTConfig = Field(
        default_factory=_HybridEncoderViTTConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTTConfig = Field(
        default_factory=_RTDETRTransformerv2ViTTConfig
    )
    dfine_transformer: _DFINETransformerViTTConfig = Field(
        default_factory=_DFINETransformerViTTConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )
    backbone_wrapper: _RTDETRBackboneWrapperViTTConfig = Field(
        default_factory=_RTDETRBackboneWrapperViTTConfig
    )


class _DINOv3LTDETRViTTPlusConfig(_DINOv3LTDETRConfig):
    hybrid_encoder: _HybridEncoderViTTPlusConfig = Field(
        default_factory=_HybridEncoderViTTPlusConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTTPlusConfig = Field(
        default_factory=_RTDETRTransformerv2ViTTPlusConfig
    )
    dfine_transformer: _DFINETransformerViTTPlusConfig = Field(
        default_factory=_DFINETransformerViTTPlusConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )
    backbone_wrapper: _RTDETRBackboneWrapperViTTPlusConfig = Field(
        default_factory=_RTDETRBackboneWrapperViTTPlusConfig
    )


class _DINOv3LTDETRViTSConfig(_DINOv3LTDETRConfig):
    hybrid_encoder: _HybridEncoderViTSConfig = Field(
        default_factory=_HybridEncoderViTSConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTSConfig = Field(
        default_factory=_RTDETRTransformerv2ViTSConfig
    )
    dfine_transformer: _DFINETransformerViTSConfig = Field(
        default_factory=_DFINETransformerViTSConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )
    backbone_wrapper: _RTDETRBackboneWrapperViTSConfig = Field(
        default_factory=_RTDETRBackboneWrapperViTSConfig
    )


class _DINOv3LTDETRViTBConfig(_DINOv3LTDETRConfig):
    hybrid_encoder: _HybridEncoderViTBConfig = Field(
        default_factory=_HybridEncoderViTBConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTBConfig = Field(
        default_factory=_RTDETRTransformerv2ViTBConfig
    )
    dfine_transformer: _DFINETransformerViTBConfig = Field(
        default_factory=_DFINETransformerViTBConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )
    backbone_wrapper: _RTDETRBackboneWrapperViTBConfig = Field(
        default_factory=_RTDETRBackboneWrapperViTBConfig
    )


class _DINOv3LTDETRViTLConfig(_DINOv3LTDETRConfig):
    hybrid_encoder: _HybridEncoderViTLConfig = Field(
        default_factory=_HybridEncoderViTLConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTLConfig = Field(
        default_factory=_RTDETRTransformerv2ViTLConfig
    )
    dfine_transformer: _DFINETransformerViTLConfig = Field(
        default_factory=_DFINETransformerViTLConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )
    backbone_wrapper: _RTDETRBackboneWrapperViTLConfig = Field(
        default_factory=_RTDETRBackboneWrapperViTLConfig
    )


# Short aliases for EdgeCrafter (ECViT) LT-DETR object-detection models.
# ``ltdetrv2-{s,m,l,x}`` -> ``edgecrafter/<ecvit preset>-ltdetr``.
# These are resolved by ``_resolve_ltdetr_alias`` at the entry of
# ``_DINOv3LTDETRBase.parse_model_name`` so users can pass the short form
# directly to ``train_object_detection(model=...)`` and ``load_model(...)``.
# Order follows increasing backbone capacity (embed_dim / ffn).
_LTDETR_V2_ALIASES: dict[str, str] = {
    "ltdetrv2-s": "edgecrafter/ecvitt-ltdetr",
    "ltdetrv2-m": "edgecrafter/ecvittplus-ltdetr",
    "ltdetrv2-l": "edgecrafter/ecvits-ltdetr",
    "ltdetrv2-x": "edgecrafter/ecvitsplus-ltdetr",
}


def _resolve_ltdetr_alias(model_name: str) -> str:
    """Return the canonical LT-DETR model name for a short alias, or the input unchanged."""
    return _LTDETR_V2_ALIASES.get(model_name, model_name)


class _DINOv3LTDETRBase(TaskModel):
    model_suffix = "ltdetr"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        image_size: tuple[int, int],
        patch_size: int | None = None,
        image_normalize: dict[str, tuple[float, ...]] | None = None,
        backbone_freeze: bool = False,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        decoder_name: _LTDETRDecoderName = "dfine",
        load_weights: bool = True,
    ) -> None:
        """Create a DINOv3 LTDETR task model.

        Args:
            model_name:
                The model name. For example ``"vitt16-ltdetr"``.
            classes:
                A dict mapping class IDs to class names.
            image_size:
                The input image size.
            patch_size:
                Patch size used to initialize the DINOv3 backbone. This is stored in
                ``init_args`` so exported checkpoints can be reconstructed with the
                same backbone patch size.
            image_normalize:
                A dict containing normalization statistics with the keys ``"mean"``
                and ``"std"``.
            backbone_freeze:
                Whether to freeze the backbone during training.
            backbone_weights:
                Path to the DINOv3 backbone weights.
            backbone_args:
                Additional arguments to pass to the DINOv3 backbone.
            load_weights:
                If False, then no pretrained weights are loaded.
        """
        super().__init__(init_args=locals(), ignore_args={"load_weights"})
        parsed_name = self.parse_model_name(model_name=model_name)

        self.model_name = parsed_name["model_name"]
        self.image_size = image_size
        self.classes = classes
        self.backbone_freeze = backbone_freeze

        # Internally, the model processes classes as contiguous integers starting at 0.
        # This list maps the internal class id to the class id in `classes`.
        internal_class_to_class = list(self.classes.keys())

        # Efficient lookup for converting internal class IDs to class IDs.
        # Registered as buffer to be automatically moved to the correct device.
        self.internal_class_to_class: Tensor
        self.register_buffer(
            "internal_class_to_class",
            torch.tensor(internal_class_to_class, dtype=torch.long),
            persistent=False,  # No need to save it in the state dict.
        )
        self.included_classes: dict[int, str] = {
            internal_class_id: class_name
            for internal_class_id, class_name in enumerate(self.classes.values())
        }

        self.image_normalize = image_normalize

        # Resolve the backbone's expected input channel count. For the DINOv3
        # package we follow the same precedence as DINOV3_PACKAGE.get_model:
        # backbone_args["in_chans"] overrides image_normalize, which overrides
        # the DINOv3 default of 3. The EdgeCrafter (ECViT) package does not
        # support multi-channel input, so we always force 3 there.
        package_name = parsed_name["package_name"]
        if package_name == EDGE_CRAFTER_PACKAGE.name:
            self._expected_input_channels: int = 3
            # ECViT only supports patch_size=16 (the ECViT-NN uses a
            # ConvPyramidPatchEmbed that raises NotImplementedError otherwise).
            # We hard-code it here so the decoder's `config.resolve_auto`
            # below resolves to the correct `[8, 16, 32]` strides regardless
            # of whether the caller passed `patch_size=16` explicitly or
            # relied on the default `None`. The `backbone_model_args` dict
            # built next is discarded for EdgeCrafter (we pass
            # `model_args=None` to EDGE_CRAFTER_PACKAGE.get_model), so this
            # only affects the decoder strides.
            if patch_size is not None and patch_size != 16:
                raise ValueError(
                    f"ECViT (EdgeCrafter) backbones only support "
                    f"patch_size=16, but got patch_size={patch_size} for "
                    f"model {model_name!r}. Remove the `patch_size` argument "
                    "(or set it to 16) to use this model."
                )
            patch_size = 16
        elif backbone_args is not None and "in_chans" in backbone_args:
            self._expected_input_channels = backbone_args["in_chans"]
        elif self.image_normalize is not None:
            self._expected_input_channels = len(self.image_normalize["mean"])
        else:
            self._expected_input_channels = 3

        # NOTE(Guarin, 08/25): We don't set drop_path_rate=0 here because it is already
        # set by DINOv3.
        backbone_model_args: dict[str, Any] = {
            "patch_size": patch_size,
        }
        if backbone_args is not None:
            backbone_model_args.update(backbone_args)
        if backbone_weights is not None:
            backbone_model_args["weights"] = str(backbone_weights)

        get_model_kwargs = {}
        if self.image_normalize is not None:
            get_model_kwargs["num_input_channels"] = len(self.image_normalize["mean"])

        # Get the backbone. ECViT lives in its own package and is loaded via
        # EDGE_CRAFTER_PACKAGE.get_model (which uses the preset's pretrained URL
        # and ignores `patch_size`/`weights` in model_args).
        if package_name == EDGE_CRAFTER_PACKAGE.name:
            backbone: ConvNeXt | DinoVisionTransformer | ECViTModelWrapper = (
                EDGE_CRAFTER_PACKAGE.get_model(
                    model_name=parsed_name["backbone_name"],
                    model_args=None,
                    load_weights=load_weights,
                )
            )
        else:
            backbone = DINOV3_PACKAGE.get_model(
                model_name=parsed_name["backbone_name"],
                model_args=backbone_model_args,
                load_weights=load_weights,
                **get_model_kwargs,
            )
        assert isinstance(
            backbone, (ConvNeXt, DinoVisionTransformer, ECViTModelWrapper)
        )

        # Map preset name -> (config_cls, config_name_strip_suffixes). For
        # ECViT we strip no suffixes (the preset names are bare) and route
        # through the DINOv3 ViT-shaped configs that match the wrapper's
        # `proj_dim` (the per-level channel count the wrapper actually emits).
        config_mapping = {
            "vitt16": _DINOv3LTDETRViTTConfig,
            "vitt16plus": _DINOv3LTDETRViTTPlusConfig,
            "vits16": _DINOv3LTDETRViTSConfig,
            "vitb16": _DINOv3LTDETRViTBConfig,
            "vitl16": _DINOv3LTDETRViTLConfig,
            "convnext-tiny": _DINOv3LTDETRTinyConfig,
            "convnext-small": _DINOv3LTDETRSmallConfig,
            "convnext-base": _DINOv3LTDETRBaseConfig,
            "convnext-large": _DINOv3LTDETRLargeConfig,
            # ECViT presets (EdgeCrafter). Reuse the DINOv3 ViT-shaped configs
            # that match the wrapper's proj_dim:
            #   ecvitt         -> proj_dim 192 -> ViTT
            #   ecvittplus     -> proj_dim 256 -> ViTTPlus
            #   ecvits         -> proj_dim 256 -> ViTTPlus (embed_dim=384 is
            #                     projected down to 256 by ECViTModelWrapper)
            #   ecvitsplus     -> proj_dim 256 -> ViTTPlus
            "ecvitt": _DINOv3LTDETRViTTConfig,
            "ecvittplus": _DINOv3LTDETRViTTPlusConfig,
            "ecvits": _DINOv3LTDETRViTTPlusConfig,
            "ecvitsplus": _DINOv3LTDETRViTTPlusConfig,
        }
        config_name = parsed_name["backbone_name"].replace("-notpretrained", "")
        config_name = config_name.replace("-noreg", "")
        config_name = config_name.replace("-eupe", "")
        config_cls = config_mapping[config_name]
        config = config_cls()
        config.decoder_name = decoder_name

        config.resolve_auto(patch_size=patch_size)

        self.backbone: DINOv3STAs | DINOv3ConvNextWrapper | ECViTBackboneWrapper

        if isinstance(backbone, ECViTModelWrapper):
            # ECViT already fuses its own pyramid; no SpatialPriorModule
            # (use_sta=False). The wrapper exposes (P3, P4, P5) with channel
            # counts matching the encoder config's in_channels via `proj_dim`.
            self.backbone = ECViTBackboneWrapper(model_wrapper=backbone)
        elif isinstance(backbone, DinoVisionTransformer):
            # TODO(Guarin, 02/26): Improve how mask tokens are handled for fine-tuning.
            backbone.mask_token.requires_grad = False  # type: ignore

            # ViT models.
            vit_model_wrapper = DINOv3ViTModelWrapper(backbone)
            self.backbone = DINOv3STAs(
                model_wrapper=vit_model_wrapper,
                **config.backbone_wrapper.model_dump(),
            )

        else:
            # ConvNext models.
            assert isinstance(backbone, ConvNeXt)
            convnext_model_wrapper = DINOv3VConvNeXtModelWrapper(backbone)
            self.backbone = DINOv3ConvNextWrapper(model_wrapper=convnext_model_wrapper)

        self.encoder: HybridEncoder = HybridEncoder(
            **config.hybrid_encoder.model_dump()
        )

        self.decoder = self.build_decoder(config=config)
        self.postprocessor: Any = self.build_postprocessor(config=config)

        if self.backbone_freeze:
            self.freeze_backbone()

    @classmethod
    def list_model_names(cls) -> list[str]:
        # Concatenate the DINOv3 and EdgeCrafter (ECViT) backbone model names,
        # each suffixed with the LTDETR task suffix. Both packages share this
        # task model. Short LT-DETRv2 aliases (e.g. ``ltdetrv2-s``) are appended
        # so they show up in error messages and discoverability listings.
        names: list[str] = []
        names.extend(DINOV3_PACKAGE.list_model_names())
        names.extend(EDGE_CRAFTER_PACKAGE.list_model_names())
        names = [f"{name}-{cls.model_suffix}" for name in names]
        names.extend(_LTDETR_V2_ALIASES.keys())
        return names

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        try:
            cls.parse_model_name(model_name=model)
        except ValueError:
            return False
        else:
            return True

    @classmethod
    def parse_model_name(cls, model_name: str) -> dict[str, str]:
        # Resolve short LT-DETRv2 aliases (e.g. ``ltdetrv2-s``) to their
        # canonical ``edgecrafter/<preset>-ltdetr`` form before format
        # validation. Done first so the error message below reports the
        # resolved canonical name when applicable.
        model_name = _resolve_ltdetr_alias(model_name)

        def raise_invalid_name() -> NoReturn:
            raise ValueError(
                f"Model name '{model_name}' is not supported. Available "
                f"models are: {cls.list_model_names()}."
            )

        if not model_name.endswith(f"-{cls.model_suffix}"):
            raise_invalid_name()

        backbone_name = model_name[: -len(f"-{cls.model_suffix}")]

        try:
            package_name, backbone_name = package_helpers.parse_model_name(
                backbone_name
            )
        except ValueError:
            raise_invalid_name()

        # Accept both DINOv3 and EdgeCrafter (ECViT) packages.
        if package_name == DINOV3_PACKAGE.name:
            try:
                backbone_name = DINOV3_PACKAGE.parse_model_name(
                    model_name=backbone_name
                )
            except ValueError:
                raise_invalid_name()
            return {
                "package_name": DINOV3_PACKAGE.name,
                "model_name": (
                    f"{DINOV3_PACKAGE.name}/{backbone_name}-{cls.model_suffix}"
                ),
                "backbone_name": backbone_name,
            }
        if package_name == EDGE_CRAFTER_PACKAGE.name:
            try:
                backbone_name = EDGE_CRAFTER_PACKAGE.parse_model_name(
                    model_name=backbone_name
                )
            except ValueError:
                raise_invalid_name()
            return {
                "package_name": EDGE_CRAFTER_PACKAGE.name,
                "model_name": (
                    f"{EDGE_CRAFTER_PACKAGE.name}/{backbone_name}-{cls.model_suffix}"
                ),
                "backbone_name": backbone_name,
            }

        raise_invalid_name()

    def build_decoder(self, config: _DINOv3LTDETRConfig) -> nn.Module:
        raise NotImplementedError()

    def build_postprocessor(self, config: _DINOv3LTDETRConfig) -> nn.Module:
        raise NotImplementedError()

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    def load_train_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load the state dict from a training checkpoint.

        Loads the EMA weights if available, otherwise falls back to the model weights.
        """
        has_ema_weights = any(k.startswith("ema_model.model.") for k in state_dict)
        has_model_weights = any(k.startswith("model.") for k in state_dict)
        new_state_dict = {}
        if has_ema_weights:
            for name, param in state_dict.items():
                if name.startswith("ema_model.model."):
                    name = name[len("ema_model.model.") :]
                    new_state_dict[name] = param
        elif has_model_weights:
            for name, param in state_dict.items():
                if name.startswith("model."):
                    name = name[len("model.") :]
                    new_state_dict[name] = param
        return self.load_state_dict(new_state_dict, strict=strict, assign=assign)

    def deploy(self) -> Self:
        self.eval()
        self.postprocessor.deploy()  # type: ignore[no-untyped-call]
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()  # type: ignore[operator]
        return self

    def preprocess_image(
        self, image: PathLike | PILImage | Tensor
    ) -> tuple[Tensor, dict[str, Any]]:
        first_param = next(self.parameters())
        device, dtype = first_param.device, first_param.dtype

        x = file_helpers.as_image_tensor(image).to(device)
        image_h, image_w = x.shape[-2:]

        # Expand grayscale to the expected channel count so images can be stacked.
        # TODO(Nauryzbay, 05/26): Revisit grayscale handling — the implicit
        # 1-channel expansion is a convenience inherited from RGB-only models.
        expected_c = self._expected_input_channels
        if x.shape[-3] == 1 and expected_c > 1:
            x = x.expand(expected_c, -1, -1)
        elif x.shape[-3] != expected_c:
            raise ValueError(
                f"Image has {x.shape[-3]} channels but model expects {expected_c}."
            )

        x = transforms_functional.to_dtype(x, dtype=dtype, scale=True)
        x = transforms_functional.resize(x, self.image_size)
        return x, {"orig_h": image_h, "orig_w": image_w}

    def preprocess_batch(self, batch: Tensor) -> Tensor:
        if self.image_normalize is not None:
            batch = transforms_functional.normalize(
                batch,
                mean=list(self.image_normalize["mean"]),
                std=list(self.image_normalize["std"]),
            )
        return batch

    @torch.no_grad()
    def predict_batch(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
        threshold: float = 0.6,
    ) -> list[dict[str, Tensor]]:
        """Run inference on a batch of images and return per-image predictions.

        Args:
            images:
                Sequence of input images. Each can be a path, a PIL image, or a
                tensor of shape (C, H, W).
            threshold:
                Score threshold to filter low-confidence predictions. Predictions
                with scores <= threshold are discarded.

        Returns:
            A list with one prediction dict per input image.
        """
        self._track_inference()
        if self.training or not self.postprocessor.deploy_mode:
            self.deploy()
        tensors: list[Tensor] = []
        metadata: list[dict[str, Any]] = []
        for image in images:
            x, meta = self.preprocess_image(image)
            tensors.append(x)
            metadata.append(meta)
        batch = torch.stack(tensors, dim=0)
        batch = self.preprocess_batch(batch)
        raw = self.forward_backend(batch)
        return self.postprocess(raw, metadata, threshold=threshold)

    @torch.no_grad()
    def predict(
        self, image: PathLike | PILImage | Tensor, threshold: float = 0.6
    ) -> dict[str, Tensor]:
        """Run inference on a single image and return task-specific predictions.

        Args:
            image:
                Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).
            threshold:
                Score threshold to filter low-confidence predictions. Predictions with
                scores <= threshold are discarded.

        Returns:
            A task-specific prediction dictionary.
        """
        self._track_inference()
        if self.training or not self.postprocessor.deploy_mode:
            self.deploy()
        x, metadata = self.preprocess_image(image)
        batch = self.preprocess_batch(x.unsqueeze(0))
        raw = self.forward_backend(batch)
        return cast(
            dict[str, Tensor], self.postprocess(raw, [metadata], threshold=threshold)[0]
        )
