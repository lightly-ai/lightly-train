#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, cast

from lightning_fabric import Fabric
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import JaccardIndex, MeanMetric

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
from lightly_train.types import MaskSemanticSegmentationBatch, PathLike


class DINOv2SemanticSegmentationTrainArgs(TaskTrainModelArgs):
    backbone_weights: PathLike | None = None
    freeze_backbone: bool = False
    drop_path_rate: float = 0.0
    ignore_index: int = -100


class DINOv2SemanticSegmentationTrain(TaskTrainModel):
    def __init__(
        self,
        task_args: DINOv2SemanticSegmentationTrainArgs,
        model_name: str,
        data_args: MaskSemanticSegmentationDataArgs,
    ) -> None:
        super().__init__()
        self.task_args = task_args

        self.model = DINOv2SemanticSegmentation(
            # TODO(Guarin, 10/25): Make configurable and pass all args.
            # We probably don't want to instantiate the model here. Either we pass it
            # from the outside or we use a setup function (might be useful for FSDP).
            model_name=model_name,
            num_classes=len(data_args.classes),
            backbone_weights=task_args.backbone_weights,
            freeze_backbone=task_args.freeze_backbone,
            model_args={
                "drop_path_rate": task_args.drop_path_rate,
            },
        )
        self.criterion = DINOv2SemanticSegmentationCrossEntropyLoss(
            task_args.ignore_index
        )
        self.val_loss = MeanMetric()

        # MeanIoU assumes that background is class 0.
        # TODO(Guarin, 07/25): Make params configurable.
        self.train_miou = JaccardIndex(  # type: ignore[arg-type]
            task=cast(Any, "multiclass"),
            num_classes=max(data_args.classes) + 1,
            ignore_index=task_args.ignore_index,
        )
        self.val_miou = self.train_miou.clone()

    def training_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> TaskStepResult:
        images = batch["image"]
        masks = batch["mask"].long()  # Long required for metrics.
        pred_masks, logits = self.model(images)
        loss = self.criterion(logits, masks)
        self.train_miou.update(pred_masks, masks)
        return TaskStepResult(
            loss=loss,
            log_dict={
                "train_loss": loss.detach(),
                "train_metric/miou": self.train_miou,
            },
        )

    def validation_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> TaskStepResult:
        images = batch["image"]
        masks = batch["mask"].long()  # Long required for metrics.
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
        # TODO(Guarin, 07/25): Handle weight decay for norm and bias parameters.
        return AdamW(self.parameters())

    def set_train_mode(self) -> None:
        self.train()
        if self.task_args.freeze_backbone:
            self.model.freeze_backbone()
