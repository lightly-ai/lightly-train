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
from typing import Any, Literal, cast

import torch
from PIL.Image import Image as PILImage
from pydantic import Field
from torch import Tensor, nn
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import Self

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models.ltdetr_object_detection.dino_vit_wrapper import (
    DINOSTAs as DINOv2STAs,
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


class _DINOv2LTDETRConfig(PydanticConfig):
    decoder_name: _LTDETRDecoderName = "rtdetrv2"
    hybrid_encoder: _HybridEncoderConfig
    rtdetr_transformer: _RTDETRTransformerv2Config
    dfine_transformer: _DFINETransformerConfig
    rtdetr_postprocessor: _RTDETRPostProcessorConfig


class _DINOv2LTDETRViTSConfig(_DINOv2LTDETRConfig):
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


class _DINOv2LTDETRViTBConfig(_DINOv2LTDETRConfig):
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


class _DINOv2LTDETRViTLConfig(_DINOv2LTDETRConfig):
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


class _DINOv2LTDETRViTGConfig(_DINOv2LTDETRConfig):
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


class _DINOv2LTDETRBase(TaskModel):
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
            "_vittest14": _DINOv2LTDETRViTSConfig,
            "vits14": _DINOv2LTDETRViTSConfig,
            "vitb14": _DINOv2LTDETRViTBConfig,
            "vitl14": _DINOv2LTDETRViTLConfig,
            "vitg14": _DINOv2LTDETRViTGConfig,
        }
        config_name = parsed_name["backbone_name"].replace("-notpretrained", "")
        config_name = config_name.replace("-noreg", "")
        config_cls = config_mapping[config_name]
        config = config_cls()
        config.decoder_name = decoder_name

        # TODO(Guarin, 02/26): Improve how mask tokens are handled for fine-tuning.
        dinov2.mask_token.requires_grad = False  # type: ignore

        model_wrapper = DINOv2ViTModelWrapper(dinov2)
        self.backbone: DINOv2STAs = DINOv2STAs(
            model_wrapper=model_wrapper,
            # Disable STA for DINOv2 as it doesn't work well with patch size 14.
            use_sta=False,
            **config.backbone_wrapper.model_dump(),
        )

        self.encoder: HybridEncoder = HybridEncoder(  # type: ignore[no-untyped-call]
            **config.hybrid_encoder.model_dump()
        )

        self.decoder = self.build_decoder(config=config)
        self.postprocessor: Any = self.build_postprocessor(config=config)

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

    def build_decoder(self, config: _DINOv2LTDETRConfig) -> nn.Module:
        raise NotImplementedError()

    def build_postprocessor(self, config: _DINOv2LTDETRConfig) -> nn.Module:
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

    def load_backbone_weights(self, backbone: nn.Module, path: PathLike) -> None:
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
