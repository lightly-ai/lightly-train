#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightning_fabric import Fabric
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import MeanMetric
from torchmetrics.segmentation import MeanIoU

from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
)
from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation import (
    DINOv2SemanticSegmentation,
)
from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation_ce_loss import (
    DINOv2SemanticSegmentationCrossEntropyLoss,
)
from lightly_train._task_models.task_train_model import (
    TaskStepResult,
    TaskTrainModel,
    TaskTrainModelArgs,
)
from lightly_train.types import MaskSemanticSegmentationBatch


class DINOv2SemanticSegmentationTrainArgs(TaskTrainModelArgs):
    pass


class DINOv2SemanticSegmentationTrain(TaskTrainModel):
    def __init__(
        self,
        task_args: DINOv2SemanticSegmentationTrainArgs,
        model_name: str,
        data_args: MaskSemanticSegmentationDataArgs,
    ) -> None:
        super().__init__()
        self.model = DINOv2SemanticSegmentation(
            # TODO(Guarin, 10/25): Make configurable and pass all args.
            # We probably don't want to instantiate the model here. Either we pass it
            # from the outside or we use a setup function (might be useful for FSDP).
            model_name=model_name,
            num_classes=len(data_args.classes),
        )
        self.criterion = DINOv2SemanticSegmentationCrossEntropyLoss()
        self.val_loss = MeanMetric()

        # MeanIoU assumes that background is class 0.
        self.train_miou = MeanIoU(
            num_classes=max(data_args.classes) + 1,
            include_background=True,
            per_class=False,
            input_format="index",
        )
        self.val_miou = self.train_miou.clone()

    def training_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> TaskStepResult:
        images = batch["images"]
        masks = batch["masks"]
        pred_masks, logits = self.model(images)
        loss = self.criterion(logits, masks)
        self.train_miou.update(pred_masks, masks)
        return TaskStepResult(
            loss=loss,
            log_dict={
                "train_loss": loss,
                "train_metric/miou": self.train_miou,
            },
        )

    def validation_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> TaskStepResult:
        images = batch["images"]
        masks = batch["masks"]
        pred_masks, logits = self.model(images)
        loss = self.criterion(logits, masks)
        self.val_loss.update(loss, weight=images.shape[0])
        self.val_miou.update(pred_masks, masks)
        return TaskStepResult(
            loss=loss,
            log_dict={
                "val_loss": self.val_loss,
                "val_metric/miou": self.val_miou,
            },
        )

    def get_optimizer(self) -> Optimizer:
        return AdamW(self.parameters())
