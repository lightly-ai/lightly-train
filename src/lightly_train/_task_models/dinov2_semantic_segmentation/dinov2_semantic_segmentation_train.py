#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

from lightning_fabric import Fabric
from torch import Module
from torch.optim.optimizer import Optimizer

from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation import (
    DINOv2SemanticSegmentation,
)
from lightly_train._task_models.task_train_model import (
    TaskTrainModel,
    TaskTrainModelArgs,
)
from lightly_train.types import MaskSemanticSegmentationBatch


class DINOv2SemanticSegmentationArgs(TaskTrainModelArgs):
    pass


class DINOv2SemanticSegmentationTrain(TaskTrainModel):
    def __init__(self, args: DINOv2SemanticSegmentationArgs) -> None:
        super().__init__()
        self.model: DINOv2SemanticSegmentation
        self.criterion: Module
        self.metric: Module

    def train_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> dict[str, Any]:
        return {}

    def val_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> dict[str, Any]:
        # Return dictionary with loss and metrics for logging.
        return {}

    def get_optimizer(self) -> Optimizer:
        raise NotImplementedError()
