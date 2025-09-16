#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from lightning_fabric import Fabric
from torch import Tensor
from torch.nn import ModuleList
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from lightly_train._configs.validate import no_auto
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
)
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.scheduler import (
    TwoStageWarmupPolySchedule,
)
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.task_model import (
    DINOv3EoMTSemanticSegmentation,
)
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.transforms import (
    DINOv3EoMTSemanticSegmentationTrainTransform,
    DINOv3EoMTSemanticSegmentationValTransform,
    DINOv3EoMTSemanticSegmentationValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train.types import MaskSemanticSegmentationBatch, PathLike


class DINOv3EoMTSemanticSegmentationTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    # Default comes from ADE20K dataset:
    # 20210 images / batch size 16 * 31 epochs ~= 40k steps.
    default_steps: ClassVar[int] = 40_000

    backbone_weights: PathLike | None = None
    backbone_url: str = ""

    num_queries: int = 100  # Default for ADE20K
    # Corresponds to L_2 in the paper and network.num_blocks in the EoMT code.
    # Defaults in paper: base=3, large=4, giant=5.
    num_joint_blocks: int | Literal["auto"] = "auto"

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
    attn_mask_annealing_steps_start: list[int] | Literal["auto"] = "auto"
    attn_mask_annealing_steps_end: list[int] | Literal["auto"] = "auto"

    # Gradient clipping.
    gradient_clip_val: float = 0.01

    # Optim
    lr: float = 1e-4
    llrd: float = 0.8  # Layer decay
    weight_decay: float = 0.05
    lr_warmup_steps: tuple[int, int] = (500, 1000)
    poly_power: float = 0.9  # Used for lr and mask annealing.

    # Metrics
    metric_log_classwise: bool = True
    metric_log_debug: bool = False

    def resolve_auto(self, total_steps: int, model_name: str) -> None:
        if self.num_joint_blocks == "auto":
            match = re.match(r"dinov3/(?P<model_size>vit(s|l|b|g|h|7b)).*", model_name)
            if match is None:
                raise ValueError(
                    f"Unknown model name '{model_name}', "
                    "see https://docs.lightly.ai/train/stable/semantic_segmentation.html#model "
                    "for all supported models."
                )
            model_size = match.group("model_size")
            self.num_joint_blocks = {
                "vits": 3,
                "vitb": 3,
                "vitl": 4,
                "vitg": 5,
                "vith": 5,
                # TODO: Verify the number of blocks. EoMT has an experiment with a
                # model of comparable size.
                "vit7b": 5,
            }[model_size]

        # Infer the number of training phases from the number of joint blocks.
        num_training_phases = self.num_joint_blocks + 2

        # The phases all have the same duration.
        phases = [
            round(i * total_steps / num_training_phases)
            for i in range(num_training_phases + 1)
        ]

        # Set the start and stop of each phases.
        self.attn_mask_annealing_steps_start = phases[1:-2]
        self.attn_mask_annealing_steps_end = phases[2:-1]

        # Ensure the number of phases is correct.
        assert len(self.attn_mask_annealing_steps_start) == self.num_joint_blocks
        assert len(self.attn_mask_annealing_steps_end) == self.num_joint_blocks


class DINOv3EoMTSemanticSegmentationTrain(TrainModel):
    task = "semantic_segmentation"
    train_model_args_cls = DINOv3EoMTSemanticSegmentationTrainArgs
    task_model_cls = DINOv3EoMTSemanticSegmentation
    train_transform_cls = DINOv3EoMTSemanticSegmentationTrainTransform
    val_transform_cls = DINOv3EoMTSemanticSegmentationValTransform

    def __init__(
        self,
        *,
        model_name: str,
        model_args: DINOv3EoMTSemanticSegmentationTrainArgs,
        data_args: MaskSemanticSegmentationDataArgs,
        val_transform_args: DINOv3EoMTSemanticSegmentationValTransformArgs,
    ) -> None:
        super().__init__()
        # Lazy import because torchmetrics is an optional dependency.
        from torchmetrics import ClasswiseWrapper, JaccardIndex, MeanMetric
        from torchmetrics.classification import (  # type: ignore[attr-defined]
            MulticlassJaccardIndex,
        )

        # Lazy import because MaskClassificationLoss depends on optional transformers
        # dependeny.
        from lightly_train._task_models.dinov3_eomt_semantic_segmentation.mask_loss import (
            MaskClassificationLoss,
        )

        self.model_args = model_args
        num_joint_blocks = no_auto(self.model_args.num_joint_blocks)

        self.model = DINOv3EoMTSemanticSegmentation(
            model_name=model_name,
            classes=data_args.included_classes,
            class_ignore_index=(
                data_args.ignore_index if data_args.ignore_classes else None
            ),
            image_size=val_transform_args.image_size,
            image_normalize=val_transform_args.normalize.model_dump(),
            num_queries=model_args.num_queries,
            num_joint_blocks=num_joint_blocks,
            backbone_weights=model_args.backbone_weights,
            backbone_url=model_args.backbone_url,
        )

        self.criterion = MaskClassificationLoss(
            num_points=model_args.loss_num_points,
            oversample_ratio=model_args.loss_oversample_ratio,
            importance_sample_ratio=model_args.loss_importance_sample_ratio,
            mask_coefficient=model_args.loss_mask_coefficient,
            dice_coefficient=model_args.loss_dice_coefficient,
            class_coefficient=model_args.loss_class_coefficient,
            num_labels=data_args.num_included_classes,
            no_object_coefficient=model_args.loss_no_object_coefficient,
        )

        # Metrics
        self.val_loss = MeanMetric()

        # TODO(Guarin, 08/25): Speed up metric calculation by not calculating
        # mIoU and classwise IoU separately. mIoU can be derived from the classwise IoU.
        self.train_miou = JaccardIndex(
            task="multiclass",  # type: ignore[arg-type]
            num_classes=data_args.num_included_classes,
            ignore_index=data_args.ignore_index,
        )
        self.val_miou = self.train_miou.clone()

        # Classwise MeanIoU for each joint block. Based on EoMT implementation.
        class_labels = list(data_args.included_classes.values())
        self.train_classwise_iou = ModuleList(
            [
                # Type ignore because old torchmetrics versions (0.8) don't support the
                # `prefix` argument. We only use the old versions for SuperGradients
                # support.
                ClasswiseWrapper(  # type: ignore[call-arg]
                    MulticlassJaccardIndex(
                        num_classes=data_args.num_included_classes,
                        validate_args=False,
                        ignore_index=data_args.ignore_index,
                        average=None,
                    ),
                    prefix="_",
                    labels=class_labels,
                )
                for _ in range(num_joint_blocks + 1)
            ]
        )
        self.val_classwise_iou = ModuleList(
            [
                # Type ignore because old torchmetrics versions (0.8) don't support the
                # `prefix` argument. We only use the old versions for SuperGradients
                # support.
                ClasswiseWrapper(  # type: ignore[call-arg]
                    MulticlassJaccardIndex(
                        num_classes=data_args.num_included_classes,
                        validate_args=False,
                        ignore_index=data_args.ignore_index,
                        average=None,
                    ),
                    prefix="_",
                    labels=class_labels,
                )
                for _ in range(num_joint_blocks + 1)
            ]
        )

    def get_task_model(self) -> DINOv3EoMTSemanticSegmentation:
        return self.model

    def training_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch, step: int
    ) -> TaskStepResult:
        num_joint_blocks = no_auto(self.model_args.num_joint_blocks)
        images = batch["image"]
        assert isinstance(images, Tensor), "Images must be a single tensor for training"
        masks = batch["mask"]
        binary_masks = batch["binary_masks"]
        _, _, H, W = images.shape

        mask_logits_per_layer, class_logits_per_layer = self.model.forward_train(
            images, return_logits_per_layer=True
        )

        # Loss
        num_blocks = len(self.model.backbone.blocks)
        losses = {}
        for block_idx, block_mask_logits, block_class_logits in zip(
            # Add +1 to num_blocks for final output.
            range(num_blocks - num_joint_blocks, num_blocks + 1),
            mask_logits_per_layer,
            class_logits_per_layer,
        ):
            block_losses = self.criterion(
                masks_queries_logits=block_mask_logits,
                class_queries_logits=block_class_logits,
                targets=binary_masks,
            )
            block_suffix = f"_block{block_idx}" if block_idx < num_blocks else ""
            block_losses = {f"{k}{block_suffix}": v for k, v in block_losses.items()}
            losses.update(block_losses)
        loss = self.criterion.loss_total(losses_all_layers=losses)
        loss_log_dict = {
            f"train_loss/{k}": v
            for k, v in losses.items()
            if "block" not in k or self.model_args.metric_log_debug
        }

        # Metrics
        for block_idx, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, (H, W), mode="bilinear")
            logits = self.model.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = logits[:, :-1]  # Drop ignore class logits.
            self.update_metrics_semantic(
                metrics=self.train_classwise_iou,
                preds=logits,
                targets=masks,
                block_idx=block_idx,
            )
        for pred, targ in zip(logits, masks):
            self.train_miou.update(pred[None, ...], targ[None, ...])

        metrics: dict[str, Any] = {
            "train_metric/miou": self.train_miou,
        }
        if self.model_args.metric_log_classwise or self.model_args.metric_log_debug:
            for block_idx, metric in zip(
                range(num_blocks - num_joint_blocks, num_blocks + 1),
                self.train_classwise_iou,
            ):
                block_suffix = f"_block{block_idx}" if block_idx < num_blocks else ""
                if not block_suffix or self.model_args.metric_log_debug:
                    metrics[f"train_metric_classwise/miou{block_suffix}"] = metric

        mask_prob_dict = {}
        if self.model_args.metric_log_debug:
            mask_prob_dict = {
                f"attention_mask_probability/block{block_idx + num_blocks - num_joint_blocks}": value
                for block_idx, value in enumerate(self.model.attn_mask_probs)
            }

        # Mask annealing.
        for i in range(len(self.model.attn_mask_probs)):
            self.model.attn_mask_probs[i] = self.mask_annealing(
                start_iter=no_auto(self.model_args.attn_mask_annealing_steps_start)[i],
                current_iter=step,
                final_iter=no_auto(self.model_args.attn_mask_annealing_steps_end)[i],
            )

        return TaskStepResult(
            loss=loss,
            log_dict={
                "train_loss": loss.detach(),
                **loss_log_dict,
                **metrics,
                **mask_prob_dict,
            },
        )

    def validation_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> TaskStepResult:
        num_joint_blocks = no_auto(self.model_args.num_joint_blocks)
        images = batch["image"]
        masks = batch["mask"]
        binary_masks = batch["binary_masks"]
        image_sizes = [(image.shape[-2], image.shape[-1]) for image in images]

        # Tile the images.
        crops_list, origins = self.model.tile(images)  # type: ignore[arg-type]
        crops = torch.stack(crops_list)

        # Tile the binary masks for the loss
        binary_masks_labels = [m["labels"] for m in binary_masks]
        binary_masks_crops, _ = self.model.tile([m["masks"] for m in binary_masks])

        # Compute the target per crop.
        binary_masks_crops_dicts = []
        for origin, binary_masks_crop in zip(origins, binary_masks_crops):
            # Store the binary mask and label for the crop.
            binary_masks_crops_dicts.append(
                {
                    "masks": binary_masks_crop,
                    "labels": binary_masks_labels[origin[0]],
                }
            )

        mask_logits_per_layer, class_logits_per_layer = self.model.forward_train(
            crops, return_logits_per_layer=True
        )
        num_blocks = len(self.model.backbone.blocks)
        losses = {}
        for i, (block_idx, mask_logits, class_logits) in enumerate(
            zip(
                # Add +1 to num_blocks for final output.
                range(num_blocks - num_joint_blocks, num_blocks + 1),
                mask_logits_per_layer,
                class_logits_per_layer,
            )
        ):
            h, w = crops.shape[-2:]
            mask_logits = F.interpolate(mask_logits, (h, w), mode="bilinear")
            crop_logits = self.model.to_per_pixel_logits_semantic(
                mask_logits, class_logits
            )
            crop_logits = crop_logits[:, :-1]  # Drop ignore class logits.

            # Un-tile the predictions.
            logits = self.model.untile(
                crop_logits=crop_logits, origins=origins, image_sizes=image_sizes
            )

            # Update the metrics.
            self.update_metrics_semantic(
                metrics=self.val_classwise_iou,
                preds=logits,
                targets=masks,
                block_idx=i,
            )

            # Compute the loss
            block_losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=binary_masks_crops_dicts,
            )
            block_suffix = f"_block{block_idx}" if block_idx < num_blocks else ""
            block_losses = {f"{k}{block_suffix}": v for k, v in block_losses.items()}
            losses.update(block_losses)

        # Compute the total loss.
        loss = self.criterion.loss_total(losses_all_layers=losses)

        # Store the block-wise losses.
        log_dict = {
            f"val_loss/{k}": v
            for k, v in losses.items()
            if "block" not in k or self.model_args.metric_log_debug
        }

        # Update the targets and predictions of the last block.
        for pred, targ in zip(logits, masks):
            self.val_miou.update(pred[None, ...], targ[None, ...])

        metrics: dict[str, Any] = {
            "val_metric/miou": self.val_miou,
        }
        if self.model_args.metric_log_classwise or self.model_args.metric_log_debug:
            for block_idx, metric in zip(
                range(num_blocks - num_joint_blocks, num_blocks + 1),
                self.val_classwise_iou,
            ):
                block_suffix = f"_block{block_idx}" if block_idx < num_blocks else ""
                if not block_suffix or self.model_args.metric_log_debug:
                    metrics[f"val_metric_classwise/miou{block_suffix}"] = metric

        return TaskStepResult(
            loss=loss,
            log_dict={
                "val_loss": loss.detach(),
                **log_dict,
                **metrics,
            },
        )

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
            return (1.0 - progress).pow(self.model_args.poly_power)  # type: ignore[no-any-return]

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
            lr = self.model_args.lr
            if param in backbone_params:
                name_list = name.split(".")
                is_block = False
                for i, key in enumerate(name_list):
                    if key == "blocks":
                        block_i = int(name_list[i + 1])
                        is_block = True
                if is_block or block_i == 0:
                    lr *= self.model_args.llrd ** (backbone_blocks - 1 - block_i)
                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": name}
                )
            else:
                other_param_groups.append(
                    {"params": [param], "lr": self.model_args.lr, "name": name}
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
        optimizer = AdamW(param_groups, weight_decay=self.model_args.weight_decay)

        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(grouped_backbone_param_groups),
            warmup_steps=self.model_args.lr_warmup_steps,
            total_steps=total_steps,
            poly_power=self.model_args.poly_power,
        )
        return optimizer, scheduler

    def set_train_mode(self) -> None:
        self.train()

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        fabric.clip_gradients(
            module=self,
            optimizer=optimizer,
            max_norm=self.model_args.gradient_clip_val,
        )
