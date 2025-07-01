#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch import Module, Tensor

from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._task_models.task_model import TaskModel


class DINOv2SemanticSegmentation(TaskModel):
    def __init__(self) -> None:
        super().__init__()
        self.backbone: DinoVisionTransformer
        self.head: Module

    def forward(self, x: Tensor) -> Tensor:
        # Forward pass for inference
        return x
