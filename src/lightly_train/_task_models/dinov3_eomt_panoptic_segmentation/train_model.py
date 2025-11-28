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
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from lightly_train._configs.validate import no_auto
from lightly_train._data.mask_panoptic_segmentation_dataset import (
    MaskPanopticSegmentationDataArgs,
)
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
from lightly_train._task_models.dinov3_eomt_panoptic_segmentation.scheduler import (
    TwoStageWarmupPolySchedule,
)
from lightly_train._task_models.dinov3_eomt_panoptic_segmentation.task_model import (
    DINOv3EoMTPanopticSegmentation,
)
from lightly_train._task_models.dinov3_eomt_panoptic_segmentation.transforms import (
    DINOv3EoMTPanopticSegmentationTrainTransform,
    DINOv3EoMTPanopticSegmentationTrainTransformArgs,
    DINOv3EoMTPanopticSegmentationValTransform,
    DINOv3EoMTPanopticSegmentationValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train.types import (
    MaskPanopticSegmentationBatch,
    PathLike,
)


class DINOv3EoMTPanopticSegmentationTaskSaveCheckpointArgs(TaskSaveCheckpointArgs):
    watch_metric: str = "val_metric/pq"
    mode: Literal["min", "max"] = "max"


class DINOv3EoMTPanopticSegmentationTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    # Default comes from COCO dataset:
    # 118287 images / batch size 16 * 12 epochs ~= 90k steps.
    default_steps: ClassVar[int] = 90_000

    save_checkpoint_args_cls: ClassVar[type[TaskSaveCheckpointArgs]] = (
        DINOv3EoMTPanopticSegmentationTaskSaveCheckpointArgs
    )

    # Model args
    backbone_weights: PathLike | None = None
    num_queries: int | Literal["auto"] = "auto"
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
    lr: float = 2e-4
    llrd: float = 0.8  # Layer-wise lr decay
    # Layer-wise lr decay for joint blocks (1.0 = no decay)
    # This is equivalent to llrd_l2_enabled=False in the original EoMT
    llrd_joint_blocks: float = 1.0
    weight_decay: float = 0.05
    lr_warmup_steps: tuple[int, int] = (2000, 3000)
    poly_power: float = 0.9  # Used for lr and mask annealing.

    # Evaluation thresholds
    # Note that the naming is slightly different than in EoMT. EoMT doesn't have a
    # threshold variable, instead it uses mask_threshold=0.8 in place of threshold and
    # hardcodes the threshold to 0.5.
    # See:
    # - https://github.com/tue-mps/eomt/blob/660778b9641c1bacbb5b0249ee3dcb684d9c94d9/training/lightning_module.py#L761-L762
    # - https://github.com/tue-mps/eomt/blob/660778b9641c1bacbb5b0249ee3dcb684d9c94d9/training/lightning_module.py#L778-L779
    threshold: float = 0.8
    mask_threshold: float = 0.5
    mask_overlap_threshold: float = 0.8

    # Metrics
    metric_log_classwise: bool = False
    metric_log_train: bool = False
    metric_log_debug: bool = False

    def resolve_auto(
        self,
        total_steps: int,
        model_name: str,
        model_init_args: dict[str, Any],
    ) -> None:
        if self.num_queries == "auto":
            num_queries = model_init_args.get("num_queries", 200)
            assert isinstance(num_queries, int)  # for mypy
            self.num_queries = num_queries

        if self.num_joint_blocks == "auto":
            if num_joint_blocks := model_init_args.get("num_joint_blocks"):
                assert isinstance(num_joint_blocks, int)  # for mypy
                self.num_joint_blocks = num_joint_blocks
            else:
                match = re.match(
                    r"dinov3/(?P<model_size>vit(s|l|b|g|h|7b)).*", model_name
                )
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


class DINOv3EoMTPanopticSegmentationTrain(TrainModel):
    task = "panoptic_segmentation"
    train_model_args_cls = DINOv3EoMTPanopticSegmentationTrainArgs
    task_model_cls = DINOv3EoMTPanopticSegmentation
    train_transform_cls = DINOv3EoMTPanopticSegmentationTrainTransform
    val_transform_cls = DINOv3EoMTPanopticSegmentationValTransform

    def __init__(
        self,
        *,
        model_name: str,
        model_args: DINOv3EoMTPanopticSegmentationTrainArgs,
        data_args: MaskPanopticSegmentationDataArgs,
        train_transform_args: DINOv3EoMTPanopticSegmentationTrainTransformArgs,
        val_transform_args: DINOv3EoMTPanopticSegmentationValTransformArgs,
        load_weights: bool,
    ) -> None:
        super().__init__()
        # Lazy import because torchmetrics is an optional dependency.
        from torchmetrics import MeanMetric
        from torchmetrics.detection import PanopticQuality

        # Lazy import because MaskClassificationLoss depends on optional transformers
        # dependency.
        from lightly_train._task_models.dinov3_eomt_panoptic_segmentation.mask_loss import (
            MaskClassificationLoss,
        )

        self.model_args = model_args
        num_queries = no_auto(self.model_args.num_queries)
        num_joint_blocks = no_auto(self.model_args.num_joint_blocks)
        image_size_train = no_auto(train_transform_args.image_size)
        image_size_val = no_auto(val_transform_args.image_size)
        image_size = (
            image_size_val if isinstance(image_size_val, tuple) else image_size_train
        )
        normalize = no_auto(val_transform_args.normalize)

        self.model = DINOv3EoMTPanopticSegmentation(
            model_name=model_name,
            stuff_classes=data_args.stuff_classes,
            thing_classes=data_args.thing_classes,
            image_size=image_size,
            image_normalize=normalize.model_dump(),
            num_queries=num_queries,
            num_joint_blocks=num_joint_blocks,
            backbone_weights=model_args.backbone_weights,
            load_weights=load_weights,
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

        # NOTE: This must match the implementations in the task model and dataset!
        num_stuff_classes = len(data_args.stuff_classes)
        num_thing_classes = len(data_args.thing_classes)
        ignore_class_id = num_stuff_classes + num_thing_classes
        # We treat here ignore_class_id as a stuff class for PQ computation.
        internal_stuff_ids = list(range(num_stuff_classes)) + [ignore_class_id]
        internal_thing_ids = list(
            range(num_stuff_classes, num_stuff_classes + num_thing_classes)
        )

        self.train_pq = PanopticQuality(
            stuffs=internal_stuff_ids,
            things=internal_thing_ids,
            return_sq_and_rq=True,
            return_per_class=True,
        )
        self.val_pq = self.train_pq.clone()

        self.train_pq_debug = self.train_pq.clone()
        self.val_pq_debug = self.val_pq.clone()

    def get_task_model(self) -> DINOv3EoMTPanopticSegmentation:
        return self.model

    def training_step(
        self, fabric: Fabric, batch: MaskPanopticSegmentationBatch, step: int
    ) -> TaskStepResult:
        # NOTE: Crowd regions are dropped in the dataset during training.
        # Neither the training loss nor the training metrics take them into account.
        num_joint_blocks = no_auto(self.model_args.num_joint_blocks)
        images = batch["image"]
        assert isinstance(images, Tensor), "Images must be a single tensor for training"
        binary_masks = batch["binary_masks"]
        target_masks = batch["masks"]
        assert isinstance(target_masks, Tensor), (
            "Masks must be a single tensor for training"
        )
        _, _, H, W = images.shape

        mask_logits_per_layer, class_logits_per_layer = self.model.forward_train(
            images, return_logits_per_layer=True
        )

        # Loss
        num_blocks = len(self.model.backbone.blocks)  # type: ignore[arg-type]
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
        metrics: dict[str, Any] = {}
        if self.model_args.metric_log_train:
            with torch.no_grad():
                mask_logits = mask_logits_per_layer[-1].detach()
                class_logits = class_logits_per_layer[-1].detach()
                mask_logits = F.interpolate(mask_logits, (H, W), mode="bilinear")
                # (B, H, W, 2)
                masks, _, _ = self.model.get_masks_segment_ids_scores(
                    mask_logits=mask_logits,
                    class_logits=class_logits,
                    threshold=self.model_args.threshold,
                    mask_threshold=self.model_args.mask_threshold,
                    mask_overlap_threshold=self.model_args.mask_overlap_threshold,
                )
                update_metric_panoptic(
                    metric=self.train_pq_debug,
                    preds=masks.clone(),  # (B, H, W, 2)
                    targets=target_masks.clone(),  # (B, H, W, 2)
                    is_crowds=[m["iscrowd"].clone() for m in binary_masks], # (B, num_segments)
                )
                _mark_ignore_regions(
                    target_masks=target_masks,
                    void_color=self.train_pq.void_color,  # type: ignore
                )
                self.train_pq.update(preds=masks, target=target_masks)
                metrics["train_metric/pq"] = self.train_pq
                metrics["train_metric/pq_debug"] = self.train_pq_debug

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
        self, fabric: Fabric, batch: MaskPanopticSegmentationBatch
    ) -> TaskStepResult:
        # NOTE: Crow regions are included in the validation loss and metrics.
        num_joint_blocks = no_auto(self.model_args.num_joint_blocks)
        images = batch["image"]
        binary_masks = batch["binary_masks"]
        image_sizes = [(image.shape[-2], image.shape[-1]) for image in images]

        # Resize and pad images to self.model.image_size
        resized_images_list = []
        resized_binary_masks = []
        crop_sizes = []
        for image, binary_mask, target_masks in zip(
            images, binary_masks, batch["masks"]
        ):
            assert image.shape[-2:] == binary_mask["masks"].shape[-2:]  # type: ignore
            assert image.shape[-2:] == target_masks.shape[:2]  # type: ignore
            image, (crop_h, crop_w) = self.model.resize_and_pad(image)
            masks, _ = self.model.resize_and_pad(binary_mask["masks"])
            resized_images_list.append(image)
            crop_sizes.append((crop_h, crop_w))
            resized_binary_masks.append(
                {
                    "labels": binary_mask["labels"],
                    "masks": masks,
                }
            )
        resized_images = torch.stack(resized_images_list, dim=0)

        # Forward pass
        resized_mask_logits_per_layer, class_logits_per_layer = (
            self.model.forward_train(resized_images, return_logits_per_layer=True)
        )

        # Losses.
        num_blocks = len(self.model.backbone.blocks)  # type: ignore[arg-type]
        losses = {}
        for block_idx, resized_mask_logits, class_logits in zip(
            # Add +1 to num_blocks for final output.
            range(num_blocks - num_joint_blocks, num_blocks + 1),
            resized_mask_logits_per_layer,
            class_logits_per_layer,
        ):
            # Compute the loss
            block_losses = self.criterion(
                masks_queries_logits=resized_mask_logits,
                class_queries_logits=class_logits,
                targets=resized_binary_masks,
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

        # Metrics
        # Final layer only
        resized_mask_logits_last_layer = resized_mask_logits_per_layer[-1]
        class_logits_last_layer = class_logits_per_layer[-1]
        # Revert resize and pad for mask logits.
        for logits, class_logits, target_masks, target_binary_mask, (crop_h, crop_w), (
            image_h,
            image_w,
        ) in zip(
            resized_mask_logits_last_layer,
            class_logits_last_layer,
            batch["masks"],
            batch["binary_masks"],
            crop_sizes,
            image_sizes,
        ):
            logits = logits.unsqueeze(0)  # (1, Q, H', W')
            # Resize to same size as before passing through the model. This is usually
            # (1, Q, 640, 640) and depends on self.model.image_size.
            logits = F.interpolate(logits, resized_images.shape[-2:], mode="bilinear")
            # Revert resize and pad from self.model.resize_and_pad
            logits = logits[..., :crop_h, :crop_w]  # (1, Q, crop_h, crop_w)
            # (1, Q, H, W)
            logits = F.interpolate(logits, (image_h, image_w), mode="bilinear")
            # (H, W, 2)
            masks, _, _ = self.model.get_image_masks_segment_ids_scores(
                mask_logits=logits[0],
                class_logits=class_logits,
                threshold=self.model_args.threshold,
                mask_threshold=self.model_args.mask_threshold,
                mask_overlap_threshold=self.model_args.mask_overlap_threshold,
            )
            update_metric_panoptic(
                metric=self.val_pq_debug,
                preds=masks.unsqueeze(0).clone(),  # (1, H, W, 2)
                targets=target_masks.unsqueeze(0).clone(),  # (1, H, W, 2)
                is_crowds=target_binary_mask["iscrowd"]
                .unsqueeze(0)
                .clone(),  # (1, num_segments)
            )
            _mark_ignore_regions(
                target_masks=target_masks,
                void_color=self.val_pq.void_color,  # type: ignore
            )
            self.val_pq.update(
                preds=masks.unsqueeze(0),  # (1, H, W, 2)
                target=target_masks.unsqueeze(0),  # (1, H, W, 2)
            )

        metrics: dict[str, Any] = {
            "val_metric/pq": self.val_pq,
            "val_metric/pq_debug": self.val_pq_debug,
        }

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

    def get_optimizer(self, total_steps: int) -> tuple[Optimizer, LRScheduler]:
        # TODO(Guarin, 07/25): It seems like EoMT doesn't exclude norm/bias params
        # from weight decay. We might want to change this.
        backbone_params = set(self.model.backbone.parameters())
        backbone_param_groups = []
        other_param_groups = []
        backbone_blocks = len(self.model.backbone.blocks)  # type: ignore[arg-type]
        num_joint_blocks = no_auto(self.model_args.num_joint_blocks)
        block_i = backbone_blocks

        for name, param in reversed(list(self.named_parameters())):
            lr = self.model_args.lr
            if param in backbone_params:
                name_list = name.split(".")
                is_block = False
                is_joint_block = False
                is_backbone_norm = False
                for i, key in enumerate(name_list):
                    if key == "blocks":
                        block_i = int(name_list[i + 1])
                        is_block = True
                        is_joint_block = block_i >= (backbone_blocks - num_joint_blocks)
                        is_backbone_norm = "backbone.norm" in name
                        break

                if (is_block or block_i == 0) and not is_backbone_norm:
                    # Apply layer-wise lr decay except for backbone.norm layer.
                    llrd = (
                        self.model_args.llrd_joint_blocks
                        if is_joint_block
                        else self.model_args.llrd
                    )
                    lr *= llrd ** (backbone_blocks - 1 - block_i)

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


def _mark_ignore_regions(
    target_masks: Tensor,
    void_color: tuple[int, int],
) -> None:
    """Sets regions in target_masks that must be ignored in PQ computation to void color.

    Args:
        target_masks:
            (..., H, W, 2) tensor where the last dimension contains (label, segment_id).
        void_color:
            Color to set ignored regions to.
    """
    void_color_tensor = target_masks.new_tensor(void_color)
    # Masks that have no segments. EoMT filters those out when intitializing
    # the dataset but we don't want to load all data when initializing the dataset.
    # Instead we handle the empty targets here.
    # See: https://github.com/tue-mps/eomt/blob/660778b9641c1bacbb5b0249ee3dcb684d9c94d9/datasets/dataset.py#L135-L136
    is_empty = (target_masks[..., 1] == -1).all(dim=(-2, -1))
    target_masks[is_empty] = void_color_tensor
    # Pixels with label -1 are iscrowd regions (see dataset get_masks).
    # EoMT handles them by customizing the PQ computation. We avoid customizing the PQ
    # computation by setting them to the void color here.
    # See: https://github.com/tue-mps/eomt/blob/660778b9641c1bacbb5b0249ee3dcb684d9c94d9/training/lightning_module.py#L325-L326
    target_masks[target_masks[..., 0] == -1] = void_color_tensor


def update_metric_panoptic(
    metric,
    preds,
    targets,
    is_crowds,
):
    from torchmetrics.functional.detection._panoptic_quality_common import (
        _calculate_iou,
        _get_color_areas,
        _prepocess_inputs,
    )

    for i in range(len(preds)):
        if (targets[i, ..., 1] == -1).all():
            # Target without segments. This is not in EoMT because EoMT filters
            # empty targets in the dataset. We allow them and handle them here
            # instead.
            continue
        flatten_pred = _prepocess_inputs(
            metric.things,
            metric.stuffs,
            preds[i][None, ...],
            metric.void_color,
            metric.allow_unknown_preds_category,
        )[0]
        flatten_target = _prepocess_inputs(
            metric.things,
            metric.stuffs,
            targets[i][None, ...],
            metric.void_color,
            True,
        )[0]

        pred_areas = _get_color_areas(flatten_pred)
        target_areas = _get_color_areas(flatten_target)
        intersection_matrix = torch.transpose(
            torch.stack((flatten_pred, flatten_target), -1), -1, -2
        )
        intersection_areas = _get_color_areas(intersection_matrix)

        pred_segment_matched = set()
        target_segment_matched = set()
        for pred_color, target_color in intersection_areas:
            if is_crowds[i][target_color[1]]:
                continue
            if target_color == metric.void_color:
                continue
            if pred_color[0] != target_color[0]:
                continue
            iou = _calculate_iou(
                pred_color,
                target_color,
                pred_areas,
                target_areas,
                intersection_areas,
                metric.void_color,
            )
            continuous_id = metric.cat_id_to_continuous_id[target_color[0]]
            if iou > 0.5:
                pred_segment_matched.add(pred_color)
                target_segment_matched.add(target_color)
                metric.iou_sum[continuous_id] += iou
                metric.true_positives[continuous_id] += 1

        false_negative_colors = set(target_areas) - target_segment_matched
        false_positive_colors = set(pred_areas) - pred_segment_matched

        false_negative_colors.discard(metric.void_color)
        false_positive_colors.discard(metric.void_color)

        for target_color in list(false_negative_colors):
            void_target_area = intersection_areas.get(
                (metric.void_color, target_color), 0
            )
            if void_target_area / target_areas[target_color] > 0.5:
                false_negative_colors.discard(target_color)

        crowd_by_cat_id = {}
        for false_negative_color in false_negative_colors:
            if is_crowds[i][false_negative_color[1]]:
                crowd_by_cat_id[false_negative_color[0]] = false_negative_color[1]
                continue

            continuous_id = metric.cat_id_to_continuous_id[false_negative_color[0]]
            metric.false_negatives[continuous_id] += 1

        for pred_color in list(false_positive_colors):
            pred_void_crowd_area = intersection_areas.get(
                (pred_color, metric.void_color), 0
            )

            if pred_color[0] in crowd_by_cat_id:
                crowd_color = (pred_color[0], crowd_by_cat_id[pred_color[0]])
                pred_void_crowd_area += intersection_areas.get(
                    (pred_color, crowd_color), 0
                )

            if pred_void_crowd_area / pred_areas[pred_color] > 0.5:
                false_positive_colors.discard(pred_color)

        for false_positive_color in false_positive_colors:
            continuous_id = metric.cat_id_to_continuous_id[false_positive_color[0]]
            metric.false_positives[continuous_id] += 1
