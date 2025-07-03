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
from torch.nn import Module
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation import (
    DINOv2SemanticSegmentation,
)
from lightly_train._task_models.task_train_model import (
    TaskTrainModel,
    TaskTrainModelArgs,
)
from lightly_train.types import MaskSemanticSegmentationBatch


class DINOv2SemanticSegmentationTrainArgs(TaskTrainModelArgs):
    pass


class DINOv2SemanticSegmentationTrain(TaskTrainModel):
    def __init__(self, args: DINOv2SemanticSegmentationTrainArgs) -> None:
        super().__init__()
        self.model = DINOv2SemanticSegmentation(
            # TODO(Guarin, 10/23): Make configurable and pass all args.
            # We probably don't want to instantiate the model here. Either we pass it
            # from the outside or we use a setup function (might be useful for FSDP).
            model_name="vitb14",
            num_classes=2,
        )
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
        return AdamW(self.parameters())
