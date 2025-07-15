#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from lightning_fabric import Fabric
from torch import Tensor
from torch.nn import ModuleList
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import JaccardIndex, MeanMetric
from torchmetrics.classification import MulticlassJaccardIndex

from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
)
from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation import (
    DINOv2SemanticSegmentation,
)
from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation_mask_loss import (
    MaskClassificationLoss,
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
    num_queries: int = 100  # Default for ADE20K
    # Corresponds to L_2 in the paper and network.num_blocks in the EoMT code.
    # Defaults in paper: base=3, large=4, giant=5.
    num_joint_blocks: int = 4

    # TODO(Guarin, 07/25): Move this to data args? EoMT uses 255 instead.
    ignore_index: int = -100

    # Loss terms
    loss_num_points: int = 12544
    loss_oversample_ratio: float = 3.0
    loss_importance_sample_ratio: float = 0.75
    loss_no_object_coefficient: float = 0.1
    loss_mask_coefficient: float = 5.0
    loss_dice_coefficient: float = 5.0
    loss_class_coefficient: float = 2.0


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
            num_queries=task_args.num_queries,
            num_joint_blocks=task_args.num_joint_blocks,
            backbone_weights=task_args.backbone_weights,
            freeze_backbone=task_args.freeze_backbone,
            model_args={
                "drop_path_rate": task_args.drop_path_rate,
            },
        )
        # self.criterion = DINOv2SemanticSegmentationCrossEntropyLoss()
        self.criterion = MaskClassificationLoss(
            num_points=task_args.loss_num_points,
            oversample_ratio=task_args.loss_oversample_ratio,
            importance_sample_ratio=task_args.loss_importance_sample_ratio,
            mask_coefficient=task_args.loss_class_coefficient,
            dice_coefficient=task_args.loss_dice_coefficient,
            class_coefficient=task_args.loss_class_coefficient,
            num_labels=len(data_args.classes),
            no_object_coefficient=task_args.loss_no_object_coefficient,
        )
        self.val_loss = MeanMetric()

        # MeanIoU assumes that background is class 0.
        # TODO(Guarin, 07/25): Make params configurable.
        self.train_miou = JaccardIndex(
            task="multiclass",
            num_classes=max(data_args.classes) + 1,
            ignore_index=task_args.ignore_index,
        )
        self.val_miou = self.train_miou.clone()

        # Classwise MeanIoU for each joint block. Based on EoMT implementation.
        self.metrics = ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=max(data_args.classes) + 1,
                    validate_args=False,
                    # NOTE(Guarin, 07/25): EoMT uses 255 as ignore index.
                    ignore_index=task_args.ignore_index,
                    average=None,
                )
                for _ in range(task_args.num_joint_blocks + 1)
            ]
        )

    def training_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> TaskStepResult:
        images = batch["image"]
        masks = batch["mask"].long()  # Long required for metrics.
        B, C, H, W = images.shape

        targets = []
        for mask in masks:
            img_masks = []
            img_labels = []
            class_ids = mask.unique()
            for class_id in class_ids:
                img_masks.append(mask == class_id)
                img_labels.append(class_id)
            targets.append(
                {
                    "masks": torch.stack(img_masks),
                    "labels": images.new_tensor(img_labels, dtype=torch.long),
                }
            )
        mask_logits_per_layer, class_logits_per_layer = self.model.forward_train(images)

        # Loss
        num_blocks = len(self.model.backbone.blocks)
        losses = {}
        log_dict = {}
        for block_idx, block_mask_logits, block_class_logits in zip(
            # Add +1 to num_blocks for final output.
            range(num_blocks - self.task_args.num_joint_blocks, num_blocks + 1),
            mask_logits_per_layer,
            class_logits_per_layer,
            strict=True,
        ):
            block_losses = self.criterion(
                masks_queries_logits=block_mask_logits,
                class_queries_logits=block_class_logits,
                targets=targets,
            )
            block_suffix = f"_block{block_idx}" if block_idx < num_blocks else ""
            block_losses = {f"{k}{block_suffix}": v for k, v in block_losses.items()}
            losses.update(block_losses)
        loss = self.criterion.loss_total(losses_all_layers=block_losses)
        log_dict = {f"train_loss/{k}": v for k, v in losses.items()}

        # Metrics
        target_pixel_masks = self.to_per_pixel_targets_semantic(targets, ignore_idx=0)
        for block_idx, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, (H, W), mode="bilinear")
            logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            self.update_metrics_semantic(
                preds=logits, targets=target_pixel_masks, block_idx=block_idx
            )
        for pred, targ in zip(logits, target_pixel_masks):
            self.train_miou.update(pred[None, ...], targ[None, ...])

        metrics: dict[str, Any] = {
            "train_metric/miou": self.train_miou,
        }
        for block_idx, metric in zip(
            range(num_blocks - self.task_args.num_joint_blocks, num_blocks + 1),
            self.metrics,
        ):
            block_suffix = f"_block{block_idx}" if block_idx < num_blocks else ""
            # These metrics should match the original EoMT metrics.
            metrics[f"train_metric/miou{block_suffix}_cls"] = metric

        # self.train_miou.update(pred_pixel_masks, target_pixel_masks)
        return TaskStepResult(
            loss=loss,
            log_dict={
                "train_loss": loss.detach(),
                **log_dict,
                **metrics,
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

    def to_per_pixel_logits_semantic(self, mask_logits: Tensor, class_logits: Tensor):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @torch.compiler.disable
    def to_per_pixel_targets_semantic(
        self,
        targets: list[dict[str, Tensor]],
        ignore_idx: int,
    ):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    @torch.compiler.disable
    def update_metrics_semantic(
        self,
        preds: Tensor,
        targets: list[torch.Tensor],
        block_idx: int,
    ):
        for i in range(len(preds)):
            self.metrics[block_idx].update(preds[i][None, ...], targets[i][None, ...])

    def get_optimizer(self) -> Optimizer:
        # TODO(Guarin, 07/25): Handle weight decay for norm and bias parameters.
        return AdamW(self.parameters())

    def set_train_mode(self) -> None:
        self.train()
        if self.task_args.freeze_backbone:
            self.model.freeze_backbone()
