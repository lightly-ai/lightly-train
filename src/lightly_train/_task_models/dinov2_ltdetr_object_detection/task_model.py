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
from torch import Tensor
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import Self

from lightly_train._data import file_helpers
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models.dinov2_ltdetr_object_detection.dinov2_vit_wrapper import (
    DINOv2ViTWrapper,
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


class DINOv2LTDETRObjectDetection(TaskModel):
    model_suffix = "ltdetr"

    def __init__(
        self,
        *,
        model_name: str,
        image_size: tuple[int, int],
        classes: dict[int, str] | None,
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

        # TODO: Lionel(09/25) Those will currently be ignored.
        self.image_normalize = image_normalize
        if image_normalize is not None:
            logger.warning(
                "The image_normalize argument is currently ignored. "
                "Images are only divided by 255."
            )
        self.backbone_weights = backbone_weights
        if backbone_weights is not None:
            logger.warning(
                "The backbone_weights argument is currently ignored. "
                "Pretrained weights are not supported yet."
            )

        dinov2 = DINOV2_VIT_PACKAGE.get_model(
            model_name=parsed_name["backbone_name"],
            model_args=backbone_args,
            load_weights=load_weights,
        )
        self.backbone: DINOv2ViTWrapper = DINOv2ViTWrapper(
            model=dinov2,
            keep_indices=[5, 8, 11],
        )
        # TODO(Lionel, 07/25): Improve how mask tokens are handled for fine-tuning.
        # Should we drop them from the model? We disable grads here for DDP to work
        # without find_unused_parameters=True.
        self.backbone.backbone.mask_token.requires_grad = False

        self.encoder: HybridEncoder = HybridEncoder(  # type: ignore[no-untyped-call]
            in_channels=[384, 384, 384],
            feat_strides=[14, 14, 14],
            hidden_dim=384,
            use_encoder_idx=[2],
            num_encoder_layers=1,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            enc_act="gelu",
            expansion=1.0,
            depth_mult=1,
            act="silu",
            upsample=False,
        )

        self.decoder: RTDETRTransformerv2 = RTDETRTransformerv2(  # type: ignore[no-untyped-call]
            feat_channels=[384, 384, 384],
            feat_strides=[14, 14, 14],
            hidden_dim=256,
            num_levels=3,
            num_layers=6,
            num_queries=300,
            num_denoising=100,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            eval_idx=-1,
            num_points=[4, 4, 4],
            query_select_method="default",
            # TODO Lionel (09/25): Remove when anchors are not in checkpoints anymore.
            eval_spatial_size=self.image_size,  # From global config, otherwise anchors are not generated.
        )

        self.postprocessor: RTDETRPostProcessor = RTDETRPostProcessor(
            num_top_queries=300,
        )

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

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{name}-{cls.model_suffix}"
            for name in DINOV2_VIT_PACKAGE.list_model_names()
        ]

    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the EMA state dict from a training checkpoint."""
        new_state_dict = {}
        for name, param in state_dict.items():
            if name.startswith("ema_model.model."):
                name = name[len("ema_model.model.") :]
                new_state_dict[name] = param
        self.load_state_dict(new_state_dict, strict=True)

    @torch.no_grad()
    def predict(
        self, image: PathLike | PILImage | Tensor, threshold: float = 0.6
    ) -> dict[str, Tensor]:
        self.postprocessor = self.postprocessor.deploy()  # type: ignore[no-untyped-call]
        self = self.deploy()  # type: ignore[no-untyped-call]

        device = next(self.parameters()).device
        x = file_helpers.as_image_tensor(image).to(device)

        h, w = x.shape[-2:]

        x = transforms_functional.to_dtype(x, dtype=torch.float32)
        x = transforms_functional.resize(x, self.image_size)
        # TODO: Lionel (09/25) Change to Normalize transform using saved params.
        x = x / 255.0
        x = x.unsqueeze(0)

        labels, boxes, scores = self(x, orig_target_size=(h, w))
        keep = scores > threshold
        labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
        return {
            "labels": labels.squeeze(0),
            "bboxes": boxes.squeeze(0),
            "scores": scores.squeeze(0),
        }

    def deploy(self) -> Self:
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()  # type: ignore[operator]
        return self

    def _forward_train(self, x: Tensor, targets):  # type: ignore[no-untyped-def]
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(feats=x, targets=targets)
        return x

    def forward(
        self, x: Tensor, orig_target_size: tuple[int, int] | None = None
    ) -> list[Tensor]:
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
        x_: list[Tensor] = self.postprocessor(x, orig_target_size_)
        return x_


class DINOv2LTDETRDSPObjectDetection(DINOv2LTDETRObjectDetection):
    model_suffix = "ltdetr-dsp"

    def __init__(
        self,
        *,
        model_name: str,
        image_size: tuple[int, int],
        classes: dict[int, str] | None,
        image_normalize: dict[str, Any] | None = None,
        backbone_weights: PathLike | None = None,
        backbone_args: dict[str, Any] | None = None,
    ) -> None:
        super(DINOv2LTDETRObjectDetection, self).__init__(
            init_args=locals(), ignore_args={"backbone_weights"}
        )
        parsed_name = self.parse_model_name(model_name=model_name)

        self.model_name = parsed_name["model_name"]
        self.image_size = image_size
        self.classes = classes
        # TODO: Lionel(09/25) this will currently be ignored, since we just divide by 255.
        self.image_normalize = image_normalize
        if image_normalize is not None:
            logger.warning(
                "The image_normalize argument is currently ignored. "
                "Images are only divided by 255."
            )

        dinov2 = DINOV2_VIT_PACKAGE.get_model(
            model_name=parsed_name["backbone_name"],
            model_args=backbone_args,
        )
        self.backbone: DINOv2ViTWrapper = DINOv2ViTWrapper(
            model=dinov2,
            keep_indices=[5, 8, 11],
        )

        self.encoder: HybridEncoder = HybridEncoder(  # type: ignore[no-untyped-call]
            in_channels=[384, 384, 384],
            feat_strides=[14, 14, 14],
            hidden_dim=384,
            use_encoder_idx=[2],
            num_encoder_layers=1,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.0,
            enc_act="gelu",
            expansion=1.0,
            depth_mult=1,
            act="silu",
            upsample=False,
        )

        self.decoder: RTDETRTransformerv2 = RTDETRTransformerv2(  # type: ignore[no-untyped-call]
            feat_channels=[384, 384, 384],
            feat_strides=[14, 14, 14],
            hidden_dim=256,
            num_levels=3,
            cross_attn_method="discrete",
            num_layers=6,
            num_queries=300,
            num_denoising=100,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            eval_idx=-1,
            num_points=[4, 4, 4],
            query_select_method="default",
            # TODO Lionel (09/25): Remove when anchors are not in checkpoints anymore.
            eval_spatial_size=(
                644,
                644,
            ),  # From global config, otherwise anchors are not generated.
        )

        self.postprocessor: RTDETRPostProcessor = RTDETRPostProcessor(
            num_top_queries=300,
        )
