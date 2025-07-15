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
from torch.optim.lr_scheduler import LRScheduler
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
from lightly_train._task_models.dinov2_semantic_segmentation.dinov2_semantic_segmentation_scheduler import (
    TwoStageWarmupPolySchedule,
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

    # Attention mask annealing.
    # This follows EoMT ADE20K semantic segmentation ViT-L defaults.
    attn_mask_annealing_steps_start: list[int] = [6520, 13040, 19560, 26080]
    attn_mask_annealing_steps_end: list[int] = [13040, 19560, 26080, 32600]

    # Optim
    lr: float = 1e-4
    llrd: float = 0.8  # Layer decay
    weight_decay: float = 0.05
    lr_warmup_steps: tuple[int, int] = (500, 1000)
    poly_power: float = 0.9  # Used for lr and mask annealing.

    # Unused EoMT args:
    # - mask_thresh: Only used for panoptic segmentation.
    # - overlap_thresh: Only used for panoptic segmentation.


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
        self.train_metrics = ModuleList(
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
        self.val_metrics = ModuleList(
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
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch, step: int
    ) -> TaskStepResult:
        images = batch["image"]
        masks = batch["mask"].long()  # Long required for metrics.
        B, C, H, W = images.shape

        targets = self.get_targets(masks)
        mask_logits_per_layer, class_logits_per_layer = self.model.forward_train(images)

        # Loss
        num_blocks = len(self.model.backbone.blocks)
        losses = {}
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
        loss_dict = {f"train_loss/{k}": v for k, v in losses.items()}

        # Metrics
        target_pixel_masks = self.to_per_pixel_targets_semantic(targets, ignore_idx=0)
        for block_idx, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, (H, W), mode="bilinear")
            logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            self.update_metrics_semantic(
                metrics=self.train_metrics,
                preds=logits,
                targets=target_pixel_masks,
                block_idx=block_idx,
            )
        for pred, targ in zip(logits, target_pixel_masks):
            self.train_miou.update(pred[None, ...], targ[None, ...])

        metrics: dict[str, Any] = {
            "train_metric/miou": self.train_miou,
        }
        for block_idx, metric in zip(
            range(num_blocks - self.task_args.num_joint_blocks, num_blocks + 1),
            self.train_metrics,
        ):
            block_suffix = f"_block{block_idx}" if block_idx < num_blocks else ""
            # These metrics should match the original EoMT metrics.
            metrics[f"train_metric/miou{block_suffix}_cls"] = metric

        mask_prob_dict = {
            f"train_attn_mask_prob/block{block_idx + num_blocks - self.task_args.num_joint_blocks}": value
            for block_idx, value in enumerate(self.model.attn_mask_probs)
        }

        # Mask annealing.
        for i in range(len(self.model.attn_mask_probs)):
            self.model.attn_mask_probs[i] = self.mask_annealing(
                start_iter=self.task_args.attn_mask_annealing_steps_start[i],
                current_iter=step,
                final_iter=self.task_args.attn_mask_annealing_steps_end[i],
            )

        return TaskStepResult(
            loss=loss,
            log_dict={
                "train_loss": loss.detach(),
                **loss_dict,
                **metrics,
                **mask_prob_dict,
            },
        )

    def validation_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> TaskStepResult:
        images = batch["image"]
        masks = batch["mask"].long()  # Long required for metrics.
        B, C, H, W = images.shape

        targets = self.get_targets(masks)
        # TODO(Guarin, 07/25): Use a different forward method for validation?
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
        log_dict = {f"val_loss/{k}": v for k, v in losses.items()}

        # Metrics
        target_pixel_masks = self.to_per_pixel_targets_semantic(targets, ignore_idx=0)
        for block_idx, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, (H, W), mode="bilinear")
            logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            self.update_metrics_semantic(
                metrics=self.val_metrics,
                preds=logits,
                targets=target_pixel_masks,
                block_idx=block_idx,
            )
        for pred, targ in zip(logits, target_pixel_masks):
            self.val_miou.update(pred[None, ...], targ[None, ...])

        metrics: dict[str, Any] = {
            "val_metric/miou": self.val_miou,
        }
        for block_idx, metric in zip(
            range(num_blocks - self.task_args.num_joint_blocks, num_blocks + 1),
            self.val_metrics,
        ):
            block_suffix = f"_block{block_idx}" if block_idx < num_blocks else ""
            # These metrics should match the original EoMT metrics.
            metrics[f"val_metric/miou{block_suffix}_cls"] = metric

        return TaskStepResult(
            loss=loss,
            log_dict={
                "val_loss": loss.detach(),
                **log_dict,
                **metrics,
            },
        )

    def get_targets(self, masks: Tensor) -> list[dict[str, Tensor]]:
        # This follows logic from: https://github.com/tue-mps/eomt/blob/716cbd562366b9746804579b48b866da487d9485/datasets/ade20k_semantic.py#L47-L48
        targets = []
        for mask in masks:
            img_masks = []
            img_labels = []
            class_ids = mask.unique()
            # TODO(Guarin, 07/25): EoMT checks whether class id is in class mappings.
            for class_id in class_ids:
                img_masks.append(mask == class_id)
                img_labels.append(class_id)
            targets.append(
                {
                    "masks": torch.stack(img_masks),
                    "labels": mask.new_tensor(img_labels, dtype=torch.long),
                }
            )
        return targets

    def to_per_pixel_logits_semantic(
        self, mask_logits: Tensor, class_logits: Tensor
    ) -> Tensor:
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @torch.compiler.disable  # type: ignore[misc]
    def to_per_pixel_targets_semantic(
        self,
        targets: list[dict[str, Tensor]],
        ignore_idx: int,
    ) -> list[Tensor]:
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

    def mask_annealing(
        self,
        start_iter: int,
        current_iter: int,
        final_iter: int,
    ) -> Tensor:
        device = self.model.attn_mask_probs[0].device
        dtype = self.model.attn_mask_probs[0].dtype
        if current_iter < start_iter:
            return torch.ones(1, device=device, dtype=dtype)
        elif current_iter >= final_iter:
            return torch.zeros(1, device=device, dtype=dtype)
        else:
            progress = torch.tensor(
                (current_iter - start_iter) / (final_iter - start_iter),
                device=device,
                dtype=dtype,
            )
            return (1.0 - progress).pow(self.task_args.poly_power)  # type: ignore[no-any-return]

    @torch.compiler.disable  # type: ignore[misc]
    def update_metrics_semantic(
        self,
        preds: Tensor,
        targets: list[torch.Tensor],
        block_idx: int,
        metrics: ModuleList,
    ) -> None:
        for i in range(len(preds)):
            metrics[block_idx].update(preds[i][None, ...], targets[i][None, ...])

    def get_optimizer(self, total_steps: int) -> tuple[Optimizer, LRScheduler]:
        # TODO(Guarin, 07/25): It seems like EoMT doesn't exclude norm/bias params
        # from weight decay. We might want to change this.
        backbone_params = set(self.model.backbone.parameters())
        backbone_param_groups = []
        other_param_groups = []
        backbone_blocks = len(self.model.backbone.blocks)
        block_i = backbone_blocks

        for name, param in reversed(list(self.named_parameters())):
            lr = self.task_args.lr
            if param in backbone_params:
                name_list = name.split(".")
                is_block = False
                for i, key in enumerate(name_list):
                    if key == "blocks":
                        block_i = int(name_list[i + 1])
                        is_block = True
                if is_block or block_i == 0:
                    lr *= self.task_args.llrd ** (backbone_blocks - 1 - block_i)
                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": name}
                )
            else:
                other_param_groups.append(
                    {"params": [param], "lr": self.task_args.lr, "name": name}
                )

        # TODO(Guarin, 07/25): Added this to reduce number of logged lr/wd values.
        # Might want to revisit this. Maybe we can make it nicer based on block names?
        def group_param_groups(
            param_groups: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            grouped = []
            current_group: dict[str, Any] = {}
            last_group = None
            for group in param_groups:
                if not current_group:
                    current_group = group
                    grouped.append(current_group)
                elif group["lr"] != current_group["lr"]:
                    assert last_group is not None
                    current_group["name"] = (
                        f"{current_group['name']}-{last_group['name']}"
                    )
                    current_group = group
                    grouped.append(current_group)
                else:
                    current_group["params"].extend(group["params"])
                last_group = group
            return grouped

        grouped_backbone_param_groups = group_param_groups(backbone_param_groups)
        grouped_other_param_groups = group_param_groups(other_param_groups)

        param_groups = grouped_backbone_param_groups + grouped_other_param_groups
        optimizer = AdamW(param_groups, weight_decay=self.task_args.weight_decay)

        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(grouped_backbone_param_groups),
            warmup_steps=self.task_args.lr_warmup_steps,
            total_steps=total_steps,
            poly_power=self.task_args.poly_power,
        )
        return optimizer, scheduler

    def set_train_mode(self) -> None:
        self.train()
        if self.task_args.freeze_backbone:
            self.model.freeze_backbone()
