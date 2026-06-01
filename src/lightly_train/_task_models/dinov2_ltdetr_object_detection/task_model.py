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
from pathlib import Path
from typing import Any, Literal

import torch
from PIL.Image import Image as PILImage
from pydantic import Field
from torch import Tensor
from torch.nn import Module
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import Self

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models.dinov2_ltdetr_object_detection.dinov2_vit_wrapper import (
    DINOv2STAs,
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


class _HybridEncoderViTTestConfig(_HybridEncoderConfig):
    in_channels: list[int] = [224, 224, 224]
    feat_strides: list[int] = [7, 14, 28]
    hidden_dim: int = 224
    use_encoder_idx: list[int] = [1]
    num_encoder_layers: int = 1
    nhead: int = 1
    dim_feedforward: int = 224
    dropout: float = 0.0
    enc_act: str = "gelu"
    expansion: float = 1.0
    depth_mult: float = 1.0
    act: str = "silu"


class _HybridEncoderViTSConfig(_HybridEncoderConfig):
    in_channels: list[int] = [224, 224, 224]
    feat_strides: list[int] = [7, 14, 28]
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
    feat_strides: list[int] = [7, 14, 28]
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
    feat_strides: list[int] = [7, 14, 28]
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


class _HybridEncoderViTGConfig(_HybridEncoderConfig):
    in_channels: list[int] = [1536, 1536, 1536]
    feat_strides: list[int] = [7, 14, 28]
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


class _RTDETRTransformerv2Config(PydanticConfig):
    feat_channels: list[int]
    feat_strides: list[int]
    hidden_dim: int
    num_levels: int
    num_layers: int
    num_queries: int
    num_denoising: int
    label_noise_ratio: float
    box_noise_scale: float
    eval_idx: int
    num_points: list[int]
    query_select_method: str


class _RTDETRTransformerv2ViTSConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [224, 224, 224]
    feat_strides: list[int] = [7, 14, 28]
    hidden_dim: int = 224
    num_levels: int = 3
    num_layers: int = 4
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [3, 6, 3]
    query_select_method: str = "default"
    dim_feedforward: int = 1792


class _RTDETRTransformerv2ViTBConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [768, 768, 768]
    feat_strides: list[int] = [7, 14, 28]
    hidden_dim: int = 768
    num_levels: int = 3
    num_layers: int = 4
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [3, 6, 3]
    query_select_method: str = "default"
    dim_feedforward: int = 6144


class _RTDETRTransformerv2ViTLConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [1024, 1024, 1024]
    feat_strides: list[int] = [7, 14, 28]
    hidden_dim: int = 1024
    num_levels: int = 3
    num_layers: int = 4
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [3, 6, 3]
    query_select_method: str = "default"
    dim_feedforward: int = 8192


class _RTDETRTransformerv2ViTGConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [1536, 1536, 1536]
    feat_strides: list[int] = [7, 14, 28]
    hidden_dim: int = 1536
    num_levels: int = 3
    num_layers: int = 4
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [3, 6, 3]
    query_select_method: str = "default"
    dim_feedforward: int = 12288


class _DFINETransformerConfig(PydanticConfig):
    feat_channels: list[int]
    feat_strides: list[int]
    hidden_dim: int
    num_levels: int
    num_layers: int
    num_queries: int
    num_denoising: int
    label_noise_ratio: float
    box_noise_scale: float
    eval_idx: int
    num_points: list[int]
    query_select_method: str
    cross_attn_method: str
    dim_feedforward: int
    reg_max: int
    reg_scale: float
    layer_scale: float


class _DFINETransformerViTSConfig(_DFINETransformerConfig):
    feat_channels: list[int] = [224, 224, 224]
    feat_strides: list[int] = [7, 14, 28]
    hidden_dim: int = 224
    num_levels: int = 3
    num_layers: int = 4
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [3, 6, 3]
    query_select_method: str = "default"
    cross_attn_method: str = "default"
    dim_feedforward: int = 1792
    reg_max: int = 32
    reg_scale: float = 4.0
    layer_scale: float = 1.0


class _DFINETransformerViTBConfig(_DFINETransformerConfig):
    feat_channels: list[int] = [768, 768, 768]
    feat_strides: list[int] = [7, 14, 28]
    hidden_dim: int = 768
    num_levels: int = 3
    num_layers: int = 4
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [3, 6, 3]
    query_select_method: str = "default"
    cross_attn_method: str = "default"
    dim_feedforward: int = 6144
    reg_max: int = 32
    reg_scale: float = 4.0
    layer_scale: float = 1.0


class _DFINETransformerViTLConfig(_DFINETransformerConfig):
    feat_channels: list[int] = [1024, 1024, 1024]
    feat_strides: list[int] = [7, 14, 28]
    hidden_dim: int = 1024
    num_levels: int = 3
    num_layers: int = 4
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [3, 6, 3]
    query_select_method: str = "default"
    cross_attn_method: str = "default"
    dim_feedforward: int = 8192
    reg_max: int = 32
    reg_scale: float = 4.0
    layer_scale: float = 1.0


class _DFINETransformerViTGConfig(_DFINETransformerConfig):
    feat_channels: list[int] = [1536, 1536, 1536]
    feat_strides: list[int] = [7, 14, 28]
    hidden_dim: int = 1536
    num_levels: int = 3
    num_layers: int = 4
    num_queries: int = 300
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_idx: int = -1
    num_points: list[int] = [3, 6, 3]
    query_select_method: str = "default"
    cross_attn_method: str = "default"
    dim_feedforward: int = 12288
    reg_max: int = 32
    reg_scale: float = 4.0
    layer_scale: float = 1.0


class _BackboneWrapperViTSConfig(PydanticConfig):
    interaction_indexes: list[int] = [5, 8, 11]
    finetune: bool = True
    conv_inplane: int = 28
    hidden_dim: int = 224


class _BackboneWrapperViTBConfig(PydanticConfig):
    interaction_indexes: list[int] = [5, 8, 11]
    finetune: bool = True
    conv_inplane: int = 56
    hidden_dim: int = 768


class _BackboneWrapperViTLConfig(PydanticConfig):
    interaction_indexes: list[int] = [11, 17, 23]
    finetune: bool = True
    conv_inplane: int = 56
    hidden_dim: int = 1024


class _BackboneWrapperViTGConfig(PydanticConfig):
    interaction_indexes: list[int] = [19, 29, 39]
    finetune: bool = True
    conv_inplane: int = 64
    hidden_dim: int = 1536


class _RTDETRPostProcessorConfig(PydanticConfig):
    num_top_queries: int = 300


class _DINOv2LTDETRObjectDetectionConfig(PydanticConfig):
    decoder_name: _LTDETRDecoderName = "rtdetrv2"
    hybrid_encoder: _HybridEncoderConfig
    rtdetr_transformer: _RTDETRTransformerv2Config
    dfine_transformer: _DFINETransformerConfig
    rtdetr_postprocessor: _RTDETRPostProcessorConfig


class _DINOv2LTDETRObjectDetectionViTSConfig(_DINOv2LTDETRObjectDetectionConfig):
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
    backbone_wrapper: _BackboneWrapperViTSConfig = Field(
        default_factory=_BackboneWrapperViTSConfig
    )


class _DINOv2LTDETRObjectDetectionViTBConfig(_DINOv2LTDETRObjectDetectionConfig):
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
    backbone_wrapper: _BackboneWrapperViTBConfig = Field(
        default_factory=_BackboneWrapperViTBConfig
    )


class _DINOv2LTDETRObjectDetectionViTLConfig(_DINOv2LTDETRObjectDetectionConfig):
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
    backbone_wrapper: _BackboneWrapperViTLConfig = Field(
        default_factory=_BackboneWrapperViTLConfig
    )


class _DINOv2LTDETRObjectDetectionViTGConfig(_DINOv2LTDETRObjectDetectionConfig):
    hybrid_encoder: _HybridEncoderViTGConfig = Field(
        default_factory=_HybridEncoderViTGConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTGConfig = Field(
        default_factory=_RTDETRTransformerv2ViTGConfig
    )
    dfine_transformer: _DFINETransformerViTGConfig = Field(
        default_factory=_DFINETransformerViTGConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )
    backbone_wrapper: _BackboneWrapperViTGConfig = Field(
        default_factory=_BackboneWrapperViTGConfig
    )


class DINOv2LTDETRObjectDetection(TaskModel):
    model_suffix = "ltdetr"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        image_size: tuple[int, int],
        image_normalize: dict[str, tuple[float, ...]] | None = None,
        backbone_freeze: bool = False,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        decoder_name: _LTDETRDecoderName = "rtdetrv2",
        load_weights: bool = True,
    ) -> None:
        super().__init__(
            init_args=locals(), ignore_args={"backbone_weights", "load_weights"}
        )
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

        # Resolve the backbone's expected input channel count using the same
        # precedence as DINOV2_VIT_PACKAGE.get_model: backbone_args["in_chans"]
        # overrides image_normalize, which overrides the DINOv2 default of 3.
        if backbone_args is not None and "in_chans" in backbone_args:
            self._expected_input_channels: int = backbone_args["in_chans"]
        elif self.image_normalize is not None:
            self._expected_input_channels = len(self.image_normalize["mean"])
        else:
            self._expected_input_channels = 3

        # Instantiate the backbone.
        dinov2 = DINOV2_VIT_PACKAGE.get_model(
            model_name=parsed_name["backbone_name"],
            model_args=backbone_args,
            load_weights=load_weights,
        )

        # Optionally load the backbone weights.
        if load_weights and backbone_weights is not None:
            self.load_backbone_weights(dinov2, backbone_weights)

        # Get the configuration based on the model name.
        config_mapping = {
            "_vittest14": _DINOv2LTDETRObjectDetectionViTSConfig,
            "vits14": _DINOv2LTDETRObjectDetectionViTSConfig,
            "vitb14": _DINOv2LTDETRObjectDetectionViTBConfig,
            "vitl14": _DINOv2LTDETRObjectDetectionViTLConfig,
            "vitg14": _DINOv2LTDETRObjectDetectionViTGConfig,
        }
        config_name = parsed_name["backbone_name"].replace("-notpretrained", "")
        config_name = config_name.replace("-noreg", "")
        config_cls = config_mapping[config_name]
        config = config_cls()
        config.decoder_name = decoder_name

        # TODO(Guarin, 02/26): Improve how mask tokens are handled for fine-tuning.
        dinov2.mask_token.requires_grad = False  # type: ignore

        self.backbone: DINOv2STAs = DINOv2STAs(
            model=dinov2,
            # Disable STA for DINOv2 as it doesn't work well with patch size 14.
            use_sta=False,
            **config.backbone_wrapper.model_dump(),
        )

        self.encoder: HybridEncoder = HybridEncoder(  # type: ignore[no-untyped-call]
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
        self._deployed: bool = False

        if self.backbone_freeze:
            self.freeze_backbone()

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{name}-{cls.model_suffix}"
            for name in DINOV2_VIT_PACKAGE.list_model_names()
        ]

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
        def raise_invalid_name() -> None:
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

        if package_name != DINOV2_VIT_PACKAGE.name:
            raise_invalid_name()

        try:
            backbone_name = DINOV2_VIT_PACKAGE.parse_model_name(
                model_name=backbone_name
            )
        except ValueError:
            raise_invalid_name()

        return {
            "model_name": f"{DINOV2_VIT_PACKAGE.name}/{backbone_name}-{cls.model_suffix}",
            "backbone_name": backbone_name,
        }

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

    def load_backbone_weights(self, backbone: Module, path: PathLike) -> None:
        """
        Load backbone weights from a checkpoint file.

        Args:
            backbone: backbone to load the statedict in.
            path: path to a .pt file, e.g., exported_last.pt.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Backbone weights file not found: '{path}'")

        # Load the checkpoint.
        state_dict = torch.load(path, map_location="cpu", weights_only=False)

        # Load the state dict into the backbone.
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)

        # Log missing and unexpected keys.
        if missing or unexpected:
            if missing:
                logger.warning(f"Missing keys when loading backbone: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading backbone: {unexpected}")
        else:
            logger.info(f"Backbone weights loaded from '{path}'")

    def deploy(self) -> Self:
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()  # type: ignore[operator]
        self._deployed = True
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
        postprocessor_out: list[dict[str, Tensor]] = self.postprocessor(
            raw_outputs, orig_target_size
        )
        out: list[dict[str, Tensor]] = []
        for result in postprocessor_out:
            labels = self.internal_class_to_class[result["labels"]]  # type: ignore[index]
            keep = result["scores"] > threshold
            out.append(
                {
                    "labels": labels[keep],
                    "bboxes": result["boxes"][keep],
                    "scores": result["scores"][keep],
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
        if self.training or not self._deployed:
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
        if self.training or not self._deployed:
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
        - Predictions are filtered by score and merged using NMS and a global/local
          consistency heuristic. NMS is only applied on tiles predictions.
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
                IoU threshold used for non-maximum suppression when merging predictions
                from tiles and global image. A lower nms_iou_threshold value yields less
                predictions.
            global_local_iou_threshold:
                Minimum IoU required to consider a tile prediction as matching a global
                prediction when combining them. A lower global_local_iou_threshold
                yields less predictions.

        Returns:
            A dictionary with:
                - "labels": Tensor of shape (N,) with predicted class indices.
                - "bboxes": Tensor of shape (N, 4) with bounding boxes in
                    (x_min, y_min, x_max, y_max) in absolute pixel coordinates of the original image.
                - "scores": Tensor of shape (N,) with confidence scores for each prediction.
        """

        if self.training or not self._deployed:
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

        # Normalize the image.
        if self.image_normalize is not None:
            tiles = transforms_functional.normalize(
                tiles,
                mean=self.image_normalize["mean"],
                std=self.image_normalize["std"],
            )

        # Feed the tiles in parallel to the model.
        raw = self.forward_backend(tiles)

        # Build per-tile sizes in (W, H) format as required by the postprocessor.
        # All tiles use model image_size; the global image uses the original dimensions.
        tile_target_sizes = torch.tensor(
            [[self.image_size[1], self.image_size[0]]], device=device
        ).repeat(len(tiles), 1)
        tile_target_sizes[0, 0] = w  # global image W
        tile_target_sizes[0, 1] = h  # global image H

        postprocessor_out = self.postprocessor(raw, tile_target_sizes)
        labels = self.internal_class_to_class[  # type: ignore[index]
            torch.stack([r["labels"] for r in postprocessor_out])
        ]
        boxes = torch.stack([r["boxes"] for r in postprocessor_out])
        scores = torch.stack([r["scores"] for r in postprocessor_out])

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

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = self.backbone(x)
        x = self.encoder(x)
        return self.decoder(x)  # type: ignore[return-value]

    def _forward_train(self, x: Tensor, targets):  # type: ignore[no-untyped-def]
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(feats=x, targets=targets)
        return x


class DINOv2LTDETRDSPObjectDetection(DINOv2LTDETRObjectDetection):
    model_suffix = "ltdetr-dsp"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        image_size: tuple[int, int],
        image_normalize: dict[str, tuple[float, ...]] | None = None,
        backbone_freeze: bool = False,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        decoder_name: _LTDETRDecoderName = "rtdetrv2",
    ) -> None:
        super(DINOv2LTDETRObjectDetection, self).__init__(
            init_args=locals(), ignore_args={"backbone_weights"}
        )
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

        self.image_normalize = image_normalize

        if backbone_args is not None and "in_chans" in backbone_args:
            self._expected_input_channels: int = backbone_args["in_chans"]
        elif self.image_normalize is not None:
            self._expected_input_channels = len(self.image_normalize["mean"])
        else:
            self._expected_input_channels = 3

        dinov2 = DINOV2_VIT_PACKAGE.get_model(
            model_name=parsed_name["backbone_name"],
            model_args=backbone_args,
        )

        # Get the configuration based on the model name.
        config_mapping = {
            "vits14": _DINOv2LTDETRObjectDetectionViTSConfig,
            "vitb14": _DINOv2LTDETRObjectDetectionViTBConfig,
            "vitl14": _DINOv2LTDETRObjectDetectionViTLConfig,
            "vitg14": _DINOv2LTDETRObjectDetectionViTGConfig,
        }
        config_name = parsed_name["backbone_name"]
        config_cls = config_mapping[config_name]
        config = config_cls()
        config.decoder_name = decoder_name

        self.backbone: DINOv2STAs = DINOv2STAs(
            model=dinov2,
            # Disable STA for DINOv2 as it doesn't work well with patch size 14.
            use_sta=False,
            **config.backbone_wrapper.model_dump(),
        )

        self.encoder: HybridEncoder = HybridEncoder(  # type: ignore[no-untyped-call]
            **config.hybrid_encoder.model_dump()
        )

        self.decoder = _build_decoder(
            config=config,
            decoder_name=config.decoder_name,
            num_classes=len(self.classes),
            image_size=self.image_size,
            cross_attn_method="discrete",
        )

        postprocessor_config = config.rtdetr_postprocessor.model_dump()
        self.postprocessor: RTDETRPostProcessor = RTDETRPostProcessor(
            **postprocessor_config
        )


def _build_decoder(
    *,
    config: _DINOv2LTDETRObjectDetectionConfig,
    decoder_name: _LTDETRDecoderName,
    num_classes: int,
    image_size: tuple[int, int],
    cross_attn_method: str | None = None,
) -> RTDETRTransformerv2 | DFINETransformer:
    if decoder_name == "rtdetrv2":
        decoder_config = config.rtdetr_transformer.model_dump()
        if cross_attn_method is not None:
            decoder_config["cross_attn_method"] = cross_attn_method
        decoder_config.update({"num_classes": num_classes})
        return RTDETRTransformerv2(  # type: ignore[no-untyped-call]
            **decoder_config,
            eval_spatial_size=image_size,
        )
    elif decoder_name == "dfine":
        decoder_config = config.dfine_transformer.model_dump()
        if cross_attn_method is not None:
            decoder_config["cross_attn_method"] = cross_attn_method
        decoder_config.update({"num_classes": num_classes})
        return DFINETransformer(  # type: ignore[no-untyped-call]
            **decoder_config,
            eval_spatial_size=image_size,
        )
    else:
        raise ValueError(f"Unsupported LTDETR decoder: {decoder_name}")
