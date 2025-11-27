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
from lightly_train._data.yolo_instance_segmentation_dataset import (
    YOLOInstanceSegmentationDataArgs,
)
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
from lightly_train._task_models import train_model_helpers
from lightly_train._task_models.dinov3_eomt_instance_segmentation.scheduler import (
    TwoStageWarmupPolySchedule,
)
from lightly_train._task_models.dinov3_eomt_instance_segmentation.task_model import (
    DINOv3EoMTInstanceSegmentation,
)
from lightly_train._task_models.dinov3_eomt_instance_segmentation.transforms import (
    DINOv3EoMTInstanceSegmentationTrainTransform,
    DINOv3EoMTInstanceSegmentationTrainTransformArgs,
    DINOv3EoMTInstanceSegmentationValTransform,
    DINOv3EoMTInstanceSegmentationValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train.types import InstanceSegmentationBatch, PathLike


class DINOv3EoMTInstanceSegmentationTaskSaveCheckpointArgs(TaskSaveCheckpointArgs):
    watch_metric: str = "val_metric/map"
    mode: Literal["min", "max"] = "max"


class DINOv3EoMTInstanceSegmentationTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    # Default comes from COCO dataset:
    # 118287 images / batch size 16 * 12 epochs ~= 90k steps.
    default_steps: ClassVar[int] = 90_000

    save_checkpoint_args_cls: ClassVar[type[TaskSaveCheckpointArgs]] = (
        DINOv3EoMTInstanceSegmentationTaskSaveCheckpointArgs
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

    # Metrics
    metric_topk_instances: int = 100
    metric_log_classwise: bool = True
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


class DINOv3EoMTInstanceSegmentationTrain(TrainModel):
    task = "instance_segmentation"
    train_model_args_cls = DINOv3EoMTInstanceSegmentationTrainArgs
    task_model_cls = DINOv3EoMTInstanceSegmentation
    train_transform_cls = DINOv3EoMTInstanceSegmentationTrainTransform
    val_transform_cls = DINOv3EoMTInstanceSegmentationValTransform

    def __init__(
        self,
        *,
        model_name: str,
        model_args: DINOv3EoMTInstanceSegmentationTrainArgs,
        data_args: YOLOInstanceSegmentationDataArgs,
        train_transform_args: DINOv3EoMTInstanceSegmentationTrainTransformArgs,
        val_transform_args: DINOv3EoMTInstanceSegmentationValTransformArgs,
        load_weights: bool,
    ) -> None:
        super().__init__()
        # Lazy import because torchmetrics is an optional dependency.
        from torchmetrics import MeanMetric

        # Type ignore because torchmetrics < 1.0 doesn't explicitly export MeanAveragePrecision
        from torchmetrics.detection import MeanAveragePrecision  # type: ignore

        # Lazy import because MaskClassificationLoss depends on optional transformers
        # dependency.
        from lightly_train._task_models.dinov3_eomt_instance_segmentation.mask_loss import (
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

        self.model = DINOv3EoMTInstanceSegmentation(
            model_name=model_name,
            classes=data_args.included_classes,
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

        # Type ignore because old torchmetrics have different formats for iou_type.
        self.train_map = MeanAveragePrecision(iou_type="segm")  # type: ignore[arg-type]
        self.train_map.warn_on_many_detections = False
        self.val_map = self.train_map.clone()

        if hasattr(self, "register_load_state_dict_pre_hook"):
            self.register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
                train_model_helpers.criterion_empty_weight_reinit_hook
            )
        else:
            # Backwards compatibility for PyTorch <= 2.4
            self._register_load_state_dict_pre_hook(  # type: ignore[no-untyped-call]
                train_model_helpers.criterion_empty_weight_reinit_hook, with_module=True
            )

    def get_task_model(self) -> DINOv3EoMTInstanceSegmentation:
        return self.model

    def training_step(
        self, fabric: Fabric, batch: InstanceSegmentationBatch, step: int
    ) -> TaskStepResult:
        num_joint_blocks = no_auto(self.model_args.num_joint_blocks)
        images = batch["image"]
        assert isinstance(images, Tensor), "Images must be a single tensor for training"
        binary_masks = batch["binary_masks"]
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
                mask_logits = mask_logits_per_layer[-1]
                class_logits = class_logits_per_layer[-1]
                mask_logits = F.interpolate(mask_logits, (H, W), mode="bilinear")
                # (B, Q), (B, Q, H, W), (B, Q)
                labels, masks, scores = self.model.get_labels_masks_scores(
                    mask_logits=mask_logits, class_logits=class_logits
                )
            self.train_map.update(
                preds=[
                    {
                        "labels": labels[i],
                        "masks": masks[i],
                        "scores": scores[i],
                    }
                    for i in range(len(labels))
                ],
                target=binary_masks,  # type: ignore[arg-type]
            )
            metrics["train_metric/map"] = self.train_map

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
        self, fabric: Fabric, batch: InstanceSegmentationBatch
    ) -> TaskStepResult:
        num_joint_blocks = no_auto(self.model_args.num_joint_blocks)
        images = batch["image"]
        binary_masks = batch["binary_masks"]
        image_sizes = [(image.shape[-2], image.shape[-1]) for image in images]

        # Resize and pad images to self.model.image_size
        resized_images_list = []
        resized_binary_masks = []
        crop_sizes = []
        for image, binary_mask in zip(images, binary_masks):
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
        predictions = []
        # Revert resize and pad for mask logits.
        for logits, class_logits, (crop_h, crop_w), (image_h, image_w) in zip(
            resized_mask_logits_last_layer,
            class_logits_last_layer,
            crop_sizes,
            image_sizes,
        ):
            logits = logits.unsqueeze(0)  # (1, Q, H', W')
            class_logits = class_logits.unsqueeze(0)  # (1, Q, num_classes)
            # Resize to same size as before passing through the model. This is usually
            # (1, Q, 640, 640) and depends on self.model.image_size.
            logits = F.interpolate(logits, resized_images.shape[-2:], mode="bilinear")
            # Revert resize and pad from self.model.resize_and_pad
            logits = logits[..., :crop_h, :crop_w]  # (1, Q, crop_h, crop_w)
            # (1, Q, H, W)
            logits = F.interpolate(logits, (image_h, image_w), mode="bilinear")
            # (1, Q), (1, Q, H, W), (1, Q)
            labels, masks, scores = self.model.get_labels_masks_scores(
                mask_logits=logits, class_logits=class_logits
            )
            predictions.append(
                {
                    "labels": labels[0],
                    "masks": masks[0],
                    "scores": scores[0],
                }
            )

        self.val_map.update(
            preds=predictions,
            target=binary_masks,
        )

        metrics: dict[str, Any] = {
            "val_metric/map": self.val_map,
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
