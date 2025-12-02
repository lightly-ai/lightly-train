#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Any

import torch
from PIL.Image import Image as PILImage
from pydantic import Field
from torch import Tensor
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import Self

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._models import package_helpers
from lightly_train._models.dinov3.dinov3_package import DINOV3_PACKAGE
from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt
from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.dinov3_convnext_wrapper import (
    DINOv3ConvNextWrapper,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.dinov3_vit_wrapper import (
    DINOv3STAs,
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


# TODO: Lionel(09/25) Make names more descriptive for ViT support.
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


class _RTDETRTransformerv2Config(PydanticConfig):
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
    num_points: list[int] = [4, 4, 4]


class _RTDETRTransformerv2LargeConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [384, 384, 384]


class _RTDETRTransformerv2BaseConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [384, 384, 384]


class _RTDETRTransformerv2SmallConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [384, 384, 384]


class _RTDETRTransformerv2TinyConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [384, 384, 384]


class _RTDETRTransformerv2ViTSConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [224, 224, 224]
    hidden_dim: int = 224
    num_layers: int = 4
    num_points: list[int] = [3, 6, 3]
    dim_feedforward: int = 1792


class _RTDETRTransformerv2ViTTPlusConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [256, 256, 256]
    hidden_dim: int = 256
    num_layers: int = 4
    num_points: list[int] = [3, 6, 3]
    dim_feedforward: int = 512


class _RTDETRTransformerv2ViTTConfig(_RTDETRTransformerv2Config):
    feat_channels: list[int] = [192, 192, 192]
    hidden_dim: int = 192
    num_layers: int = 4
    num_points: list[int] = [3, 6, 3]
    dim_feedforward: int = 512


class _RTDETRBackboneWrapperViTSConfig(PydanticConfig):
    interaction_indexes: list[int] = [5, 8, 11]
    finetune: bool = True
    conv_inplane: int = 32
    hidden_dim: int = 224


class _RTDETRBackboneWrapperViTTPlusConfig(PydanticConfig):
    interaction_indexes: list[int] = [3, 7, 11]
    finetune: bool = True
    conv_inplane: int = 16
    hidden_dim: int = 256


class _RTDETRBackboneWrapperViTTConfig(PydanticConfig):
    interaction_indexes: list[int] = [3, 7, 11]
    finetune: bool = True
    conv_inplane: int = 16
    hidden_dim: int = 192


class _RTDETRPostProcessorConfig(PydanticConfig):
    num_top_queries: int = 300


class _DINOv3LTDETRObjectDetectionConfig(PydanticConfig):
    hybrid_encoder: _HybridEncoderConfig
    rtdetr_transformer: _RTDETRTransformerv2Config
    rtdetr_postprocessor: _RTDETRPostProcessorConfig


class _DINOv3LTDETRObjectDetectionLargeConfig(_DINOv3LTDETRObjectDetectionConfig):
    hybrid_encoder: _HybridEncoderLargeConfig = Field(
        default_factory=_HybridEncoderLargeConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2LargeConfig = Field(
        default_factory=_RTDETRTransformerv2LargeConfig
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
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )


class _DINOv3LTDETRObjectDetectionViTSConfig(_DINOv3LTDETRObjectDetectionConfig):
    hybrid_encoder: _HybridEncoderViTSConfig = Field(
        default_factory=_HybridEncoderViTSConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTSConfig = Field(
        default_factory=_RTDETRTransformerv2ViTSConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )
    backbone_wrapper: _RTDETRBackboneWrapperViTSConfig = Field(
        default_factory=_RTDETRBackboneWrapperViTSConfig
    )


class _DINOv3LTDETRObjectDetectionViTTPlusConfig(_DINOv3LTDETRObjectDetectionConfig):
    hybrid_encoder: _HybridEncoderViTTPlusConfig = Field(
        default_factory=_HybridEncoderViTTPlusConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTTPlusConfig = Field(
        default_factory=_RTDETRTransformerv2ViTTPlusConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )
    backbone_wrapper: _RTDETRBackboneWrapperViTTPlusConfig = Field(
        default_factory=_RTDETRBackboneWrapperViTTPlusConfig
    )


class _DINOv3LTDETRObjectDetectionViTTConfig(_DINOv3LTDETRObjectDetectionConfig):
    hybrid_encoder: _HybridEncoderViTTConfig = Field(
        default_factory=_HybridEncoderViTTConfig
    )
    rtdetr_transformer: _RTDETRTransformerv2ViTTConfig = Field(
        default_factory=_RTDETRTransformerv2ViTTConfig
    )
    rtdetr_postprocessor: _RTDETRPostProcessorConfig = Field(
        default_factory=_RTDETRPostProcessorConfig
    )
    backbone_wrapper: _RTDETRBackboneWrapperViTTConfig = Field(
        default_factory=_RTDETRBackboneWrapperViTTConfig
    )


class DINOv3LTDETRObjectDetection(TaskModel):
    model_suffix = "ltdetr"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        image_size: tuple[int, int],
        image_normalize: dict[str, Any] | None = None,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> None:
        super().__init__(
            init_args=locals(), ignore_args={"backbone_weights", "load_weights"}
        )
        parsed_name = self.parse_model_name(model_name=model_name)

        self.model_name = parsed_name["model_name"]
        self.image_size = image_size
        self.classes = classes

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
        self.backbone_weights = backbone_weights
        if backbone_weights is not None:
            logger.warning(
                "The backbone_weights argument is currently ignored. "
                "Pretrained weights are not supported yet."
            )

        backbone_args = {} if backbone_args is None else backbone_args
        # TODO: Lionel(09/25) Relax constraint to random weights from the constructor.
        backbone_args.update({"pretrained": False})
        dinov3 = DINOV3_PACKAGE.get_model(
            parsed_name["backbone_name"],
            model_args=backbone_args,
            load_weights=load_weights,
        )
        assert isinstance(dinov3, (ConvNeXt, DinoVisionTransformer))

        config_mapping = {
            "vitt16": (_DINOv3LTDETRObjectDetectionViTTConfig, DINOv3STAs),
            "vitt16plus": (_DINOv3LTDETRObjectDetectionViTTPlusConfig, DINOv3STAs),
            "vits16": (_DINOv3LTDETRObjectDetectionViTSConfig, DINOv3STAs),
            "convnext-tiny": (
                _DINOv3LTDETRObjectDetectionTinyConfig,
                DINOv3ConvNextWrapper,
            ),
            "convnext-small": (
                _DINOv3LTDETRObjectDetectionSmallConfig,
                DINOv3ConvNextWrapper,
            ),
            "convnext-base": (
                _DINOv3LTDETRObjectDetectionBaseConfig,
                DINOv3ConvNextWrapper,
            ),
            "convnext-large": (
                _DINOv3LTDETRObjectDetectionLargeConfig,
                DINOv3ConvNextWrapper,
            ),
        }
        config_cls, wrapper_cls = config_mapping[parsed_name["backbone_name"]]
        config = config_cls()

        if hasattr(config, "backbone_wrapper"):
            # ViT models.
            self.backbone = wrapper_cls(
                model=dinov3, **config.backbone_wrapper.model_dump()
            )
        else:
            # ConvNext models.
            self.backbone = wrapper_cls(model=dinov3)

        self.encoder: HybridEncoder = HybridEncoder(
            **config.hybrid_encoder.model_dump()
        )

        decoder_config = config.rtdetr_transformer.model_dump()
        decoder_config.update({"num_classes": len(self.classes)})
        self.decoder: RTDETRTransformerv2 = RTDETRTransformerv2(  # type: ignore[no-untyped-call]
            **decoder_config,
            eval_spatial_size=self.image_size,  # From global config, otherwise anchors are not generated.
        )

        postprocessor_config = config.rtdetr_postprocessor.model_dump()
        postprocessor_config.update({"num_classes": len(self.classes)})
        self.postprocessor: RTDETRPostProcessor = RTDETRPostProcessor(
            **postprocessor_config
        )

    @classmethod
    def list_model_names(cls) -> list[str]:
        # TODO: Lionel(09/25) Add support for ViT models as well.
        return [
            f"{name}-{cls.model_suffix}"
            for name in DINOV3_PACKAGE.list_model_names()
            if "convnext" in name
        ]

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the EMA state dict from a training checkpoint."""
        new_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith("ema_model.model."):
                name = name[len("ema_model.model.") :]
                new_state_dict[name] = param
        self.load_state_dict(new_state_dict, strict=True)

    def deploy(self) -> Self:
        self.eval()
        self.postprocessor.deploy()  # type: ignore[no-untyped-call]
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()  # type: ignore[operator]
        return self

    @torch.no_grad()
    def predict(
        self, image: PathLike | PILImage | Tensor, threshold: float = 0.6
    ) -> dict[str, Tensor]:
        if self.training or not self.postprocessor.deploy_mode:
            self.deploy()

        device = next(self.parameters()).device
        x = file_helpers.as_image_tensor(image).to(device)

        h, w = x.shape[-2:]

        x = transforms_functional.to_dtype(x, dtype=torch.float32, scale=True)

        # Normalize the image.
        if self.image_normalize is not None:
            x = transforms_functional.normalize(
                x, mean=self.image_normalize["mean"], std=self.image_normalize["std"]
            )
        x = transforms_functional.resize(x, self.image_size)
        x = x.unsqueeze(0)

        labels, boxes, scores = self(x, orig_target_size=(h, w))
        keep = scores > threshold
        labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
        return {
            "labels": labels,
            "bboxes": boxes,
            "scores": scores,
        }

    def forward(
        self, x: Tensor, orig_target_size: tuple[int, int] | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Function used for ONNX export
        h, w = x.shape[-2:]
        if orig_target_size is None:
            orig_target_size_ = torch.tensor([w, h])[None].to(x.device)
        else:
            orig_target_size_ = torch.tensor(
                [orig_target_size[1], orig_target_size[0]]
            )[None].to(x.device)
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

        if package_name != DINOV3_PACKAGE.name:
            raise_invalid_name()

        try:
            backbone_name = DINOV3_PACKAGE.parse_model_name(model_name=backbone_name)
        except ValueError:
            raise_invalid_name()

        return {
            "model_name": f"{DINOV3_PACKAGE.name}/{backbone_name}-{cls.model_suffix}",
            "backbone_name": backbone_name,
        }

    @classmethod
    def is_supported_model(cls, model: str) -> bool:
        try:
            cls.parse_model_name(model_name=model)
        except ValueError:
            return False
        else:
            return True

    def _forward_train(self, x: Tensor, targets):  # type: ignore[no-untyped-def]
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(feats=x, targets=targets)
        return x
