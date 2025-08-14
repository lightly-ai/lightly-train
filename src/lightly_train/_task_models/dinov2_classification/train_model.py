#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from lightning_fabric import Fabric
from torch import Tensor, nn

from lightly_train._data.classification_dataset import ClassificationDataArgs
from lightly_train._task_models.dinov2_classification.task_model import (
    DINOv2Classification,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)


class ClassificationBatch(Mapping[str, Tensor]):
    pass


class DINOv2ClassificationTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 64
    default_steps: ClassVar[int] = 10_000
    backbone_weights: str | None = None
    num_classes: int = 1000
    drop_path_rate: float = 0.0


class DINOv2ClassificationTrainModel(TrainModel):
    def __init__(
        self,
        *,
        model_name: str,
        model_args: DINOv2ClassificationTrainArgs,
        data_args: ClassificationDataArgs,
    ) -> None:
        super().__init__()
        self.model = DINOv2Classification(
            model_name=model_name,
            num_classes=model_args.num_classes,
            backbone_weights=model_args.backbone_weights,
            drop_path_rate=model_args.drop_path_rate,
        )
        self.criterion = nn.CrossEntropyLoss()

    def get_task_model(self) -> DINOv2Classification:
        # Return type must match base class
        return self.model  # type: ignore[return-value]

    def training_step(
        self, fabric: Fabric, batch: ClassificationBatch, step: int
    ) -> TaskStepResult:
        images = batch["image"]
        targets = batch["target"]
        logits = self.model(images)
        loss = self.criterion(logits, targets)
        return TaskStepResult(
            loss=loss,
            log_dict={"train_loss": loss.detach()},
        )

    def validation_step(
        self, fabric: Fabric, batch: ClassificationBatch
    ) -> TaskStepResult:
        images = batch["image"]
        targets = batch["target"]
        logits = self.model(images)
        loss = self.criterion(logits, targets)
        return TaskStepResult(
            loss=loss,
            log_dict={"val_loss": loss.detach()},
        )
