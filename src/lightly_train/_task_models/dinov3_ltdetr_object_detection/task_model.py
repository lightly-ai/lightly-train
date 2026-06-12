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
from copy import deepcopy
from typing import Any, Literal, NoReturn

import torch
from PIL.Image import Image as PILImage
from pydantic import Field
from torch import Tensor
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import Self

from lightly_train import _logging, _torch_testing
from lightly_train._commands import _warnings
from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._export import tensorrt_helpers
from lightly_train._export.onnx_helpers import (
    fix_topological_order,
    remove_redundant_casts,
)
from lightly_train._models import package_helpers
from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper
from lightly_train._models.dinov3.dinov3_package import DINOV3_PACKAGE
from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt
from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper
from lightly_train._models.ecvit.ecvit import ECViTWrapper
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
from lightly_train._task_models.object_detection_components import tiling_utils
from lightly_train._task_models.object_detection_components.dfine_decoder import (
    DFINETransformer,
)
from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
)
from lightly_train._task_models.object_detection_components.rtdetr_postprocessor import (
    RTDETRPostProcessor,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_decoder import (
    RTDETRTransformerv2,
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


class _DINOv3LTDETRObjectDetectionConfig(PydanticConfig):
    decoder_name: _LTDETRDecoderName = "rtdetrv2"
    hybrid_encoder: _HybridEncoderConfig
    rtdetr_transformer: _RTDETRTransformerv2Config
    dfine_transformer: _DFINETransformerConfig
    rtdetr_postprocessor: _RTDETRPostProcessorConfig

    def resolve_auto(self, patch_size: int | None) -> None:
        self.rtdetr_transformer.resolve_auto(patch_size=patch_size)


class _DINOv3LTDETRObjectDetectionLargeConfig(_DINOv3LTDETRObjectDetectionConfig):
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


class _DINOv3LTDETRObjectDetectionBaseConfig(_DINOv3LTDETRObjectDetectionConfig):
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


class _DINOv3LTDETRObjectDetectionSmallConfig(_DINOv3LTDETRObjectDetectionConfig):
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


class _DINOv3LTDETRObjectDetectionTinyConfig(_DINOv3LTDETRObjectDetectionConfig):
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


class _DINOv3LTDETRObjectDetectionViTTConfig(_DINOv3LTDETRObjectDetectionConfig):
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


class _DINOv3LTDETRObjectDetectionViTTPlusConfig(_DINOv3LTDETRObjectDetectionConfig):
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


class _DINOv3LTDETRObjectDetectionViTSConfig(_DINOv3LTDETRObjectDetectionConfig):
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


class _DINOv3LTDETRObjectDetectionViTBConfig(_DINOv3LTDETRObjectDetectionConfig):
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


class _DINOv3LTDETRObjectDetectionViTLConfig(_DINOv3LTDETRObjectDetectionConfig):
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


class DINOv3LTDETRObjectDetection(TaskModel):
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
        decoder_name: _LTDETRDecoderName = "rtdetrv2",
        load_weights: bool = True,
    ) -> None:
        """Create a DINOv3 LTDETR object detection model.

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
            backbone: ConvNeXt | DinoVisionTransformer | ECViTWrapper = (
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
        assert isinstance(backbone, (ConvNeXt, DinoVisionTransformer, ECViTWrapper))

        # Map preset name -> (config_cls, config_name_strip_suffixes). For
        # ECViT we strip no suffixes (the preset names are bare) and route
        # through the DINOv3 ViT-shaped configs that match the wrapper's
        # `proj_dim` (the per-level channel count the wrapper actually emits).
        config_mapping = {
            "vitt16": _DINOv3LTDETRObjectDetectionViTTConfig,
            "vitt16plus": _DINOv3LTDETRObjectDetectionViTTPlusConfig,
            "vits16": _DINOv3LTDETRObjectDetectionViTSConfig,
            "vitb16": _DINOv3LTDETRObjectDetectionViTBConfig,
            "vitl16": _DINOv3LTDETRObjectDetectionViTLConfig,
            "convnext-tiny": _DINOv3LTDETRObjectDetectionTinyConfig,
            "convnext-small": _DINOv3LTDETRObjectDetectionSmallConfig,
            "convnext-base": _DINOv3LTDETRObjectDetectionBaseConfig,
            "convnext-large": _DINOv3LTDETRObjectDetectionLargeConfig,
            # ECViT presets (EdgeCrafter). Reuse the DINOv3 ViT-shaped configs
            # that match the wrapper's proj_dim:
            #   ecvitt         -> proj_dim 192 -> ViTT
            #   ecvittplus     -> proj_dim 256 -> ViTTPlus
            #   ecvits         -> proj_dim 256 -> ViTTPlus (embed_dim=384 is
            #                     projected down to 256 by ECViTWrapper)
            #   ecvitsplus     -> proj_dim 256 -> ViTTPlus
            "ecvitt": _DINOv3LTDETRObjectDetectionViTTConfig,
            "ecvittplus": _DINOv3LTDETRObjectDetectionViTTPlusConfig,
            "ecvits": _DINOv3LTDETRObjectDetectionViTTPlusConfig,
            "ecvitsplus": _DINOv3LTDETRObjectDetectionViTTPlusConfig,
        }
        config_name = parsed_name["backbone_name"].replace("-notpretrained", "")
        config_name = config_name.replace("-noreg", "")
        config_name = config_name.replace("-eupe", "")
        config_cls = config_mapping[config_name]
        config = config_cls()
        config.decoder_name = decoder_name

        config.resolve_auto(patch_size=patch_size)

        self.backbone: DINOv3STAs | DINOv3ConvNextWrapper | ECViTBackboneWrapper

        if isinstance(backbone, ECViTWrapper):
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

        self.decoder = _build_decoder(
            config=config,
            decoder_name=config.decoder_name,
            num_classes=len(self.classes),
            image_size=self.image_size,
        )

        postprocessor_config = config.rtdetr_postprocessor.model_dump()
        postprocessor_config.update({"num_classes": len(self.classes)})
        self.postprocessor: RTDETRPostProcessor = RTDETRPostProcessor(
            **postprocessor_config
        )

        if self.backbone_freeze:
            self.freeze_backbone()

    @classmethod
    def list_model_names(cls) -> list[str]:
        # Concatenate the DINOv3 and EdgeCrafter (ECViT) backbone model names,
        # each suffixed with the LTDETR task suffix. Both packages share this
        # task model.
        names: list[str] = []
        names.extend(DINOV3_PACKAGE.list_model_names())
        names.extend(EDGE_CRAFTER_PACKAGE.list_model_names())
        return [f"{name}-{cls.model_suffix}" for name in names]

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

    def forward_backend(self, x: Tensor) -> Any:
        x = self.backbone(x)
        x = self.encoder(x)
        return self.decoder(x)

    def postprocess(  # type: ignore[override]
        self,
        raw_outputs: Any | dict[str, Tensor],
        metadata: Sequence[dict[str, Any]],
        threshold: float,
    ) -> list[dict[str, Tensor]]:
        if not isinstance(raw_outputs, dict):
            raise ValueError(
                f"Expected raw_outputs to be a dict, got {type(raw_outputs).__name__}."
            )
        device = next(self.parameters()).device
        # Postprocessor expects (W, H) per image.
        orig_target_size = torch.tensor(
            [[m["orig_w"], m["orig_h"]] for m in metadata],
            dtype=torch.int64,
            device=device,
        )
        postprocessor_out: tuple[Tensor, Tensor, Tensor] = self.postprocessor(
            raw_outputs, orig_target_size
        )
        out: list[dict[str, Tensor]] = []
        labels_batch, boxes_batch, scores_batch = postprocessor_out

        labels_batch = self.internal_class_to_class[labels_batch]
        for i in range(len(metadata)):
            keep = scores_batch[i] > threshold
            out.append(
                {
                    "labels": labels_batch[i][keep],
                    "bboxes": boxes_batch[i][keep],
                    "scores": scores_batch[i][keep],
                }
            )
        return out

    @torch.no_grad()
    def predict_batch(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
        threshold: float = 0.6,
    ) -> list[dict[str, Tensor]]:
        """Run inference on a batch of images and return per-image detections.

        Args:
            images:
                Sequence of input images. Each can be a path, a PIL image, or a
                tensor of shape (C, H, W).
            threshold:
                Score threshold to filter low-confidence predictions. Predictions
                with scores <= threshold are discarded.

        Returns:
            A list with one dict per input image. Each dict contains:
                - "labels": Tensor of shape (N,) with predicted class indices.
                - "bboxes": Tensor of shape (N, 4) with bounding boxes in
                  (x_min, y_min, x_max, y_max) in absolute pixel coordinates of the
                  original image.
                - "scores": Tensor of shape (N,) with confidence scores.
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
        """Run inference on a single image and return detected boxes, labels and scores.

        Args:
            image:
                Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).
            threshold:
                Score threshold to filter low-confidence predictions. Predictions with
                scores <= threshold are discarded.

        Returns:
            A dictionary with:
                - "labels": Tensor of shape (N,) with predicted class indices.
                - "bboxes": Tensor of shape (N, 4) with bounding boxes in
                  (x_min, y_min, x_max, y_max) in absolute pixel coordinates of the
                  original image.
                - "scores": Tensor of shape (N,) with confidence scores for each prediction.
        """
        self._track_inference()
        if self.training or not self.postprocessor.deploy_mode:
            self.deploy()
        x, metadata = self.preprocess_image(image)
        batch = self.preprocess_batch(x.unsqueeze(0))
        raw = self.forward_backend(batch)
        return self.postprocess(raw, [metadata], threshold=threshold)[0]

    @torch.no_grad()
    def predict_sahi(
        self,
        image: PathLike | PILImage | Tensor,
        threshold: float = 0.6,
        overlap: float = 0.2,
        nms_iou_threshold: float = 0.3,
        global_local_iou_threshold: float = 0.1,
    ) -> dict[str, Tensor]:
        """Run Slicing Aided Hyper Inference (SAHI) inference on the input image.

        The image is first converted to a tensor, then:

        - Tiled into overlapping crops of size `self.image_size`.
        - A resized full-image version is added as a "global" tile.
        - All tiles (global + local) are passed through the model in parallel.
        - Predictions are filtered by score and merged using NMS and a
          global/local consistency heuristic. NMS is only applied on tiles predictions.
          The heuristic discards tiles predictions that heavily overlaps with global
          predictions.

        Args:
            image:
                Input image. Can be a path, a PIL image, or a tensor of shape (C, H, W).
            threshold:
                Score threshold for filtering low-confidence predictions.
            overlap:
                Fractional overlap between tiles in [0, 1). 0.0 means no overlap.
            nms_iou_threshold:
                IoU threshold used for non-maximum suppression when merging
                predictions from tiles and global image. A lower nms_iou_threshold
                value yields less predictions.
            global_local_iou_threshold:
                Minimum IoU required to consider a tile prediction
                as matching a global prediction when combining them. A lower
                global_local_iou_threshold yields less predictions.

        Returns:
            A dictionary with:
                - "labels": Tensor of shape (N,) with predicted class indices.
                - "bboxes": Tensor of shape (N, 4) with bounding boxes in (x_min, y_min, x_max, y_max)
                  in the coordinates of the original image.
                - "scores": Tensor of shape (N,) with confidence scores for each prediction.
        """

        if self.training or not self.postprocessor.deploy_mode:
            self.deploy()

        device = next(self.parameters()).device
        x = file_helpers.as_image_tensor(image).to(device)

        # Tile the image.
        tiles, tiles_coordinates = tiling_utils.tile_image(x, overlap, self.image_size)

        # Prepare the full image tile
        h, w = x.shape[-2:]
        x = transforms_functional.resize(x, self.image_size)
        x = x.unsqueeze(0)
        tiles = torch.cat([x, tiles], dim=0)

        # Normalize the tiles and the image together.
        tiles = transforms_functional.to_dtype(tiles, dtype=torch.float32, scale=True)

        # Normalize the tiles.
        if self.image_normalize is not None:
            tiles = transforms_functional.normalize(
                tiles,
                mean=self.image_normalize["mean"],
                std=self.image_normalize["std"],
            )

        # Prepare the image/tiles sizes.
        orig_target_sizes = torch.tensor([self.image_size], device=device).repeat(
            len(tiles), 1
        )
        orig_target_sizes[0, 0] = h
        orig_target_sizes[0, 1] = w

        # Feed the tiles in parallel to the model.
        labels, boxes, scores = self(tiles, orig_target_size=orig_target_sizes)

        # Add coordinates of the tiles to the boxes.
        tiles_coordinates = (
            tiles_coordinates.repeat(1, 2).unsqueeze(1).expand(-1, boxes.shape[1], -1)
        )
        boxes[1:] += tiles_coordinates

        # Reorganize the predictions.
        boxes_global = boxes[0].view(-1, 4)
        boxes_tiles = boxes[1:].view(-1, 4)
        labels_global = labels[0].flatten()
        labels_tiles = labels[1:].flatten()
        scores_global = scores[0].flatten()
        scores_tiles = scores[1:].flatten()

        # Discard low-confidence predictions.
        keep_global = scores_global > threshold
        keep_tiles = scores_tiles > threshold

        # Combine global and tiles predictions.
        labels, boxes, scores = tiling_utils.combine_predictions_tiles_and_global(
            pred_global={
                "labels": labels_global[keep_global],
                "bboxes": boxes_global[keep_global],
                "scores": scores_global[keep_global],
            },
            pred_tiles={
                "labels": labels_tiles[keep_tiles],
                "bboxes": boxes_tiles[keep_tiles],
                "scores": scores_tiles[keep_tiles],
            },
            nms_iou_threshold=nms_iou_threshold,
            global_local_iou_threshold=global_local_iou_threshold,
        )

        return {
            "labels": labels,
            "bboxes": boxes,
            "scores": scores,
        }

    def forward(
        self, x: Tensor, orig_target_size: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Function used for ONNX export
        if orig_target_size is None:
            h, w = x.shape[-2:]
            orig_target_size_ = torch.tensor([[w, h]]).to(x.device)
        else:
            # Flip from (H, W) to (W, H).
            orig_target_size = orig_target_size[:, [1, 0]]

            # Move to device.
            orig_target_size_ = orig_target_size.to(device=x.device, dtype=torch.int64)

        # Forward the image through the model.
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)

        result: list[dict[str, Tensor]] | tuple[Tensor, Tensor, Tensor] = (
            self.postprocessor(x, orig_target_size_)
        )
        # Postprocessor must be in deploy mode at this point. It returns only tuples
        # during deploy mode.
        assert isinstance(result, tuple)
        labels, boxes, scores = result
        labels = self.internal_class_to_class[labels]
        return (labels, boxes, scores)

    def _forward_train(self, x: Tensor, targets):  # type: ignore[no-untyped-def]
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(feats=x, targets=targets)
        return x

    @torch.no_grad()
    def export_onnx(
        self,
        out: PathLike,
        *,
        precision: Literal["fp32", "fp16"] = "fp32",
        batch_size: int = 1,
        dynamic_batch_size: bool = True,
        opset_version: int | None = None,
        simplify: bool = True,
        verify: bool = True,
        format_args: dict[str, Any] | None = None,
        num_channels: int | None = None,
    ) -> None:
        """Exports the model to ONNX for inference.

        The export uses a dummy input of shape (batch_size, C, H, W) where C is
        inferred from the first model parameter and (H, W) come from
        `self.image_size`. If `dynamic_batch_size` is True, the ONNX graph will
        have a dynamic batch dimension for the input. The graph produces three
        outputs: labels, boxes, and scores.

        Optionally simplifies the exported model in-place using onnxslim and
        verifies numerical closeness against a float32 CPU reference via
        ONNX Runtime.

        Args:
            out:
                Path where the ONNX model will be written.
            precision:
                Precision for the ONNX model. Either "fp32", or "fp16".
            batch_size:
                Batch size for the ONNX input.
            dynamic_batch_size:
                If True, the ONNX graph will have a dynamic batch dimension for the
                input. If False, the batch dimension is fixed to `batch_size`.
            opset_version:
                ONNX opset version to target. If None, PyTorch's default opset is used.
            simplify:
                If True, run onnxslim to simplify and overwrite the exported model.
            verify:
                If True, validate the ONNX file and compare outputs to a float32 CPU
                reference forward pass.
            format_args:
                Optional extra keyword arguments forwarded to `torch.onnx.export`.
            num_channels:
                Number of input channels. If None, will be inferred.

        Returns:
            None. Writes the ONNX model to `out`.

        """
        # Set up logging.
        _warnings.filter_export_warnings()
        _logging.set_up_console_logging()

        # Set the model in eval and deploy mode.
        self.eval()

        if precision not in ("fp32", "fp16"):
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of 'fp32', 'fp16'."
            )

        # Always trace in fp32 to avoid dtype mismatches in the decoder's
        # autocast(enabled=False) blocks. fp16 conversion is applied
        # post-export via onnxruntime.transformers.
        self.to(torch.float32)
        self.deploy()
        model_device = next(self.parameters()).device

        # Try to infer num_channels if not provided.
        if num_channels is None:
            if self.image_normalize is not None:
                num_channels = len(self.image_normalize["mean"])
                logger.info(
                    f"Inferred num_channels={num_channels} from image_normalize."
                )
            else:
                # Try to find the number of channels from the first convolutional layer.
                for module in self.modules():
                    if isinstance(module, torch.nn.Conv2d):
                        num_channels = module.in_channels
                        logger.info(
                            f"Inferred num_channels={num_channels} from first Conv. layer."
                        )
                        break
                if num_channels is None:
                    logger.error(
                        "Could not infer num_channels. Please provide it explicitly."
                    )
                    raise ValueError(
                        "num_channels must be provided for ONNX export if it cannot be inferred."
                    )

        if dynamic_batch_size:
            batch_size = 2
        dynamic_axes = {"images": {0: "N"}} if dynamic_batch_size else None

        # Create dummy input using same device and dtype as the model.
        dummy_input = torch.randn(
            batch_size,
            num_channels,
            self.image_size[
                0
            ],  # TODO(Thomas, 12/25): Allow passing different image size.
            self.image_size[1],
            requires_grad=False,
            device=model_device,
            dtype=torch.float32,
        )

        # TODO(Thomas, 12/25): Add warm-up forward if needed.

        # Set the input/output names.
        input_names = ["images"]
        output_names = ["labels", "boxes", "scores"]

        # TODO(Nauryzbay, 05/2026): When refactoring forward() to use forward_backend(),
        # expose orig_target_size as a second ONNX input to rescale boxes to original
        # image coordinates inside the graph.
        torch.onnx.export(
            self,
            (dummy_input,),
            str(out),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamo=False,
            dynamic_axes=dynamic_axes,
            **(format_args or {}),
        )

        if precision == "fp16":
            # convert_float_to_float16 creates nodes with duplicate names. In order to avoid downstream issues
            # we require simplify to be True, as this correctly renames nodes.
            if not simplify:
                raise ValueError("fp16 precision requires simplify=True.")

            import onnx
            from onnxruntime.transformers import float16 as ort_float16

            model_onnx = onnx.load(str(out))
            # If the input to Softmax are too large the output of Softmax will be NaN values. Therefore we run
            #  the Softmax computation in fp32. The nodes before Softmax are always MatMul.
            # TODO (simon, 05/26) Ideally we would only block operators were a Matmul directly feeds into a Softmax.
            op_block_list = list(ort_float16.DEFAULT_OP_BLOCK_LIST) + [
                "Softmax",
                "MatMul",
            ]
            model_fp16 = ort_float16.convert_float_to_float16(
                model_onnx, op_block_list=op_block_list
            )
            # Using the op blocklist on a graph that looks like Softmax -> MatMul creates a graph that looks like
            #  Cast32 -> MatMul -> Cast16 -> Cast32 -> Softmax -> Cast16. Therefore, we need to remove the middle
            #  Cast16 -> Cast32.
            remove_redundant_casts(model_fp16)
            fix_topological_order(model_fp16)
            onnx.save(model_fp16, str(out))

        if simplify:
            import onnxslim  # type: ignore [import-not-found,import-untyped]

            onnxslim.slim(
                str(out),
                output_model=out,
            )

        if verify:
            logger.info("Verifying ONNX model")
            import onnx
            import onnxruntime as ort

            onnx.checker.check_model(out, full_check=True)

            providers = ort.get_available_providers()
            if precision == "fp16" and "CUDAExecutionProvider" not in providers:
                logger.warning(
                    "Skipping ONNX runtime verification for fp16 model because "
                    "CUDAExecutionProvider is not available in onnxruntime. "
                    "Install onnxruntime-gpu to enable full verification."
                )
            else:
                # Always run the reference input in float32 and on cpu for consistency.
                reference_model = deepcopy(self).cpu().to(torch.float32).eval()
                reference_model.deploy()
                reference_outputs = reference_model(
                    dummy_input.cpu().to(torch.float32),
                )

                # Get outputs from the ONNX model. Load from bytes to avoid
                # ORT errors about missing external data when weights are inline.
                with open(out, "rb") as f:
                    session = ort.InferenceSession(f.read())
                onnx_input = dummy_input.cpu()
                if precision == "fp16":
                    onnx_input = onnx_input.half()
                input_feed = {
                    "images": onnx_input.numpy(),
                }
                outputs_onnx = session.run(output_names=None, input_feed=input_feed)
                outputs_onnx = tuple(torch.from_numpy(y) for y in outputs_onnx)

                # Verify that the outputs from both models are close.
                if len(outputs_onnx) != len(reference_outputs):
                    raise AssertionError(
                        f"Number of onnx outputs should be {len(reference_outputs)} but is {len(outputs_onnx)}"
                    )
                for output_onnx, output_model, output_name in zip(
                    outputs_onnx, reference_outputs, output_names
                ):

                    def msg(s: str) -> str:
                        return f'ONNX validation failed for output "{output_name}": {s}'

                    # Due to the presence of top-k operations in the model, the outputs may be
                    # in different order but still valid. To account for this, we sum
                    # over the query dimension before comparing.
                    output_model = output_model.sum(dim=1)
                    if output_onnx.is_floating_point:
                        # Convert to fp32 to avoid overflow issues when summing in fp16.
                        output_onnx = output_onnx.float()
                    output_onnx = output_onnx.sum(dim=1)

                    if output_model.is_floating_point:
                        # Absolute and relative tolerances are a bit arbitrary and taken from here:
                        # https://github.com/pytorch/pytorch/blob/main/torch/onnx/_internal/exporter/_core.py#L1611-L1618
                        torch.testing.assert_close(
                            output_onnx,
                            output_model,
                            msg=msg,
                            equal_nan=True,
                            check_device=False,
                            check_dtype=False,
                            check_layout=False,
                            atol=5e-3,
                            rtol=1e-1,
                        )
                    else:
                        _torch_testing.assert_most_equal(
                            output_onnx,
                            output_model,
                            msg=msg,
                        )

        logger.info(f"Successfully exported ONNX model to '{out}'")

    def export_tensorrt(
        self,
        out: PathLike,
        *,
        precision: Literal["fp32", "fp16"] = "fp32",
        onnx_args: dict[str, Any] | None = None,
        max_batchsize: int = 1,
        opt_batchsize: int = 1,
        min_batchsize: int = 1,
        verbose: bool = False,
    ) -> None:
        """Build a TensorRT engine from an ONNX model.

        .. note::
            TensorRT is not part of LightlyTrain’s dependencies and must be installed separately.
            Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup.
            See the [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html) for more details.
            On CUDA 12.x systems you can often install the Python package via `pip install tensorrt-cu12`.

        This loads the ONNX file, parses it with TensorRT, infers the static input
        shape (C, H, W) from the `"images"` input, and creates an engine with a
        dynamic batch dimension in the range `[min_batchsize, opt_batchsize, max_batchsize]`.
        Spatial dimensions must be static in the ONNX model (dynamic H/W are not yet supported).

        The engine is serialized and written to `out`.

        Args:
            out:
                Path where the TensorRT engine will be saved.
            precision:
                Precision for ONNX export and TensorRT engine building. Either
                "fp32" or "fp16".
            onnx_args:
                Optional arguments to pass to `export_onnx` when exporting
                the ONNX model prior to building the TensorRT engine. If None,
                default arguments are used and the ONNX file is saved alongside
                the TensorRT engine with the same name but `.onnx` extension.
            max_batchsize:
                Maximum supported batch size.
            opt_batchsize:
                Batch size TensorRT optimizes for.
            min_batchsize:
                Minimum supported batch size.
            verbose:
                Enable verbose TensorRT logging.

        Raises:
            FileNotFoundError: If the ONNX file does not exist.
            RuntimeError: If the ONNX cannot be parsed or engine building fails.
            ValueError: If batch size constraints are invalid or H/W are dynamic.
        """
        model_dtype = next(self.parameters()).dtype

        onnx_args = dict(onnx_args) if onnx_args is not None else {}
        onnx_args.setdefault("precision", precision)

        tensorrt_helpers.export_tensorrt(
            export_onnx_fn=self.export_onnx,
            out=out,
            precision=precision,
            model_dtype=model_dtype,
            onnx_args=onnx_args,
            max_batchsize=max_batchsize,
            opt_batchsize=opt_batchsize,
            min_batchsize=min_batchsize,
            # We convert the fp32 attention scores already during ONNX export
            fp32_attention_scores=False,
            verbose=verbose,
        )


def _build_decoder(
    *,
    config: _DINOv3LTDETRObjectDetectionConfig,
    decoder_name: _LTDETRDecoderName,
    num_classes: int,
    image_size: tuple[int, int],
) -> RTDETRTransformerv2 | DFINETransformer:
    if decoder_name == "rtdetrv2":
        decoder_config = config.rtdetr_transformer.model_dump()
        decoder_config.update({"num_classes": num_classes})
        return RTDETRTransformerv2(  # type: ignore[no-untyped-call]
            **decoder_config,
            eval_spatial_size=image_size,
        )
    elif decoder_name == "dfine":
        decoder_config = config.dfine_transformer.model_dump()
        decoder_config.update({"num_classes": num_classes})
        return DFINETransformer(  # type: ignore[no-untyped-call]
            **decoder_config,
            eval_spatial_size=image_size,
        )
    else:
        raise ValueError(f"Unsupported LTDETR decoder: {decoder_name}")
