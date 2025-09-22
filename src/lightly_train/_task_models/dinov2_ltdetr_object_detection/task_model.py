#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import PIL.Image.Image as PILImage
import torch
from torch import Tensor

from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    HybridEncoder,
)
from lightly_train._task_models.object_detection_components.rtdetr_transformerv2 import (
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
        backbone_args: dict | None = None,
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

        self.backbone: DinoVisionTransformer = DINOV2_VIT_PACKAGE.get_model(
            model_name=parsed_name["backbone_name"],
            model_args=backbone_args,
        )

        self.encoder: HybridEncoder = None
        self.decoder: RTDETRTransformerv2 = None

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
    def predict(self, image: PathLike | PILImage | Tensor) -> dict:
        raise NotImplementedError()

    def forward(self, x: Tensor) -> dict:
        # Function used for ONNX export
        raise NotImplementedError()

    def _forward_bboxes_logits(self, x: Tensor) -> dict:
        """Forward pass that returns bounding boxes and class logits. Intended for inference."""
        raise NotImplementedError()
