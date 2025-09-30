#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train._data import file_helpers
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models.dinov2_ltdetr_object_detection.dinov2_vit_wrapper import (
    DINOv2ViTWrapper,
)
from lightly_train._task_models.object_detection_components.detr_postprocessor import (
    DetDETRPostProcessor,
)
from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_decoder import (
    RTDETRTransformerv2,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike


class DINOv2LTDetrDSPObjectDetectionTaskModel(TaskModel):
    model_suffix = "ltdetr-dsp"

    def __init__(
        self,
        *,
        model_name: str,
        classes: dict[int, str],
        class_ignore_index: int | None,
        backbone_freeze: bool,
        image_size: tuple[int, int],
        image_normalize: dict[str, tuple[float, ...]],
        backbone_weights: PathLike | None,
        backbone_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(init_args=locals(), ignore_args={"backbone_weights"})
        parsed_name = self.parse_model_name(model_name=model_name)

        self.model_name = parsed_name["model_name"]
        self.classes = classes
        self.class_ignore_index = class_ignore_index
        # TODO: Lionel(09/25) implement ignore index handling.
        if class_ignore_index is not None:
            raise NotImplementedError()
        self.backbone_freeze = backbone_freeze
        self.image_size = image_size
        # TODO: Lionel(09/25) this will currently be ignored, since we just divide by 255.
        self.image_normalize = image_normalize

        # TODO: Lionel(09/25) check drop_path in LTDetr.

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
            eval_spatial_size=(644, 644),
        )
        self.decoder.training = False

        self.postprocessor: DetDETRPostProcessor = DetDETRPostProcessor(
            num_top_queries=300,
        )

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

    @torch.no_grad()
    def predict(self, image: PathLike | PILImage | Tensor) -> dict[str, Tensor]:
        for stage in [self.backbone, self.encoder, self.decoder, self.postprocessor]:
            stage.deploy()

        if self.training:
            self.eval()

        device = next(self.parameters()).device
        x = file_helpers.as_image_tensor(image).to(device)
        image_h, image_w = x.shape[:-2]

        x = transforms_functional.to_dtype(x, dtype=torch.float32, scale=True)
        # TODO: Lionel (09/25) Change to Normalize transform using saved params.
        x = x / 255.0
        x = x.unsqueeze(0)

        labels, boxes, scores = self(x)
        return {
            "labels": labels,
            "bboxes": boxes,
            "scores": scores,
        }

    def forward(self, x: Tensor) -> list[Tensor]:
        # Function used for ONNX export
        orig_target_size = x.shape[1:3]
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x_: list[Tensor] = self.postprocessor(x)
        return x_

    def _forward_bboxes_logits(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass that returns bounding boxes and class logits. Intended for inference."""
        raise NotImplementedError()
