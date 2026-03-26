#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from typing import Any, ClassVar

import torch
from lightly.utils.scheduler import CosineWarmupScheduler
from lightning_fabric import Fabric
from pydantic import Field, model_validator
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

from lightly_train._configs.validate import no_auto
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
)
from lightly_train._metrics.multihead_task_metric import MultiheadTaskMetric
from lightly_train._metrics.semantic_segmentation.task_metric import (
    SemanticSegmentationTaskMetric,
    SemanticSegmentationTaskMetricArgs,
)
from lightly_train._optim import optimizer_helpers
from lightly_train._task_models.semantic_segmentation_multihead.task_model import (
    SemanticSegmentationMultihead,
)
from lightly_train._task_models.semantic_segmentation_multihead.transforms import (
    SemanticSegmentationMultiheadTrainTransform,
    SemanticSegmentationMultiheadTrainTransformArgs,
    SemanticSegmentationMultiheadValTransform,
    SemanticSegmentationMultiheadValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train._torch_compile import TorchCompileArgs
from lightly_train.types import MaskSemanticSegmentationBatch, PathLike


class SemanticSegmentationMultiheadTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 80_000

    # Backbone args
    backbone_weights: PathLike | None = None
    backbone_args: dict[str, Any] = Field(default_factory=dict)

    gradient_clip_val: float = 0.0

    # Optim
    lr: list[float] | float = [
        0.00001,
        0.00003,
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
    ]
    weight_decay: float = 0.0
    lr_warmup_steps: int = 0

    @model_validator(mode="after")
    def _convert_lr_to_list(self) -> SemanticSegmentationMultiheadTrainArgs:
        """Convert float lr to single-element list."""
        if isinstance(self.lr, float):
            self.lr = [self.lr]
        return self


class SemanticSegmentationMultiheadTrain(TrainModel):
    task = "semantic_segmentation_multihead"
    train_model_args_cls = SemanticSegmentationMultiheadTrainArgs
    task_metric_args_cls = SemanticSegmentationTaskMetricArgs
    task_model_cls = SemanticSegmentationMultihead
    train_transform_cls = SemanticSegmentationMultiheadTrainTransform
    val_transform_cls = SemanticSegmentationMultiheadValTransform
    torch_compile_args_cls = TorchCompileArgs

    def __init__(
        self,
        *,
        model_name: str,
        model_args: SemanticSegmentationMultiheadTrainArgs,
        data_args: MaskSemanticSegmentationDataArgs,
        train_transform_args: SemanticSegmentationMultiheadTrainTransformArgs,
        val_transform_args: SemanticSegmentationMultiheadValTransformArgs,
        load_weights: bool,
        metric_args: SemanticSegmentationTaskMetricArgs,
        gradient_accumulation_steps: int,
    ) -> None:
        super().__init__()

        image_size = no_auto(val_transform_args.image_size)
        normalize = no_auto(val_transform_args.normalize)

        self.model_args = model_args
        self.metric_args = metric_args

        # Convert lr to list and store for optimizer creation.
        self.lrs = model_args.lr if isinstance(model_args.lr, list) else [model_args.lr]

        # Generate head names from learning rates.
        head_names = [_format_head_name(lr) for lr in self.lrs]

        self.model = SemanticSegmentationMultihead(
            model=model_name,
            classes=data_args.included_classes,
            head_names=head_names,
            class_ignore_index=(
                data_args.ignore_index if data_args.ignore_classes else None
            ),
            image_size=image_size,
            image_normalize=normalize.model_dump(),
            backbone_weights=model_args.backbone_weights,
            backbone_args=model_args.backbone_args,
            load_weights=load_weights,
        )

        self.loss_fn = CrossEntropyLoss(ignore_index=data_args.ignore_index)

        # Create per-head metrics using MultiheadTaskMetric.
        # Loss tracking is embedded inside each head's TaskMetric via update_with_losses().
        # Always pass ignore_index: the dataset maps padded/unknown pixels to
        # data_args.ignore_index (-100) regardless of whether ignore_classes is set.
        ignore_index = data_args.ignore_index
        class_names = list(data_args.included_classes.values())
        train_head_metrics: dict[str, SemanticSegmentationTaskMetric] = {}
        val_head_metrics: dict[str, SemanticSegmentationTaskMetric] = {}
        for lr in self.lrs:
            head_name = _format_head_name(lr)
            train_head_metrics[head_name] = SemanticSegmentationTaskMetric(
                task_metric_args=metric_args,
                split="train",
                class_names=class_names,
                ignore_index=ignore_index,
                loss_names=["loss"],
                train_loss_running_mean_window=gradient_accumulation_steps,
            )
            val_head_metrics[head_name] = SemanticSegmentationTaskMetric(
                task_metric_args=metric_args,
                split="val",
                class_names=class_names,
                ignore_index=ignore_index,
                loss_names=["loss"],
            )
        self.train_metrics: MultiheadTaskMetric = MultiheadTaskMetric(
            head_metrics=train_head_metrics,
        )
        self.val_metrics: MultiheadTaskMetric = MultiheadTaskMetric(
            head_metrics=val_head_metrics,
        )

    def get_task_model(self) -> SemanticSegmentationMultihead:
        return self.model

    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
    ) -> tuple[Optimizer, LRScheduler]:
        # Create parameter groups for each head with independent learning rates.
        params: list[dict[str, Any]] = []

        for lr in self.lrs:
            head_name = _format_head_name(lr)
            head_module = self.model.class_heads[head_name]

            # Get parameters for this head, separated by weight decay.
            params_wd, params_no_wd = optimizer_helpers.get_weight_decay_parameters(
                [head_module]
            )

            # Filter out parameters with requires_grad=False.
            params_wd = [p for p in params_wd if p.requires_grad]
            params_no_wd = [p for p in params_no_wd if p.requires_grad]

            # Scale learning rate for this head.
            lr_scaled = lr * math.sqrt(
                global_batch_size / self.model_args.default_batch_size
            )

            # Create parameter groups for this head.
            if params_wd:
                params.append(
                    {
                        "name": f"{head_name}_params",
                        "params": params_wd,
                        "lr": lr_scaled,
                    }
                )
            if params_no_wd:
                params.append(
                    {
                        "name": f"{head_name}_no_weight_decay",
                        "params": params_no_wd,
                        "lr": lr_scaled,
                        "weight_decay": 0.0,
                    }
                )

        # Create optimizer with all parameter groups.
        optimizer = SGD(
            params=params,
            weight_decay=self.model_args.weight_decay,
        )

        # Create learning rate scheduler.
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=self.model_args.lr_warmup_steps,
            max_epochs=total_steps,
        )

        return optimizer, scheduler

    def training_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch, step: int
    ) -> TaskStepResult:
        images = batch["image"]
        assert isinstance(images, Tensor), "Images must be a single tensor for training"
        masks = batch["mask"]
        assert isinstance(masks, Tensor), "Masks must be a single tensor for training"

        # Forward through all heads.
        logits_dict = self.model.forward_train(images)

        # Compute loss and update metrics for each head.
        losses: list[Tensor] = []
        for head_name, logits in logits_dict.items():
            # Handle ignore class.
            if self.model.class_ignore_index is not None:
                logits = logits[:, :-1]  # Drop logits for the ignored class.

            # Compute loss.
            loss = self.loss_fn(logits, masks)
            losses.append(loss)

            # Update per-head quality metrics and loss (scalar, no accumulation).
            head_metrics: SemanticSegmentationTaskMetric = (
                self.train_metrics.head_metrics[  # type: ignore
                    head_name
                ]
            )
            head_metrics.update_with_predictions(logits, masks)
            head_metrics.update_with_losses({"loss": loss.detach()}, weight=len(images))

        # Sum losses for backprop.
        loss_sum = torch.stack(losses).sum()

        return TaskStepResult(loss=loss_sum, log_dict={}, metrics=self.train_metrics)

    def validation_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> TaskStepResult:
        images = batch["image"]
        masks = batch["mask"]
        image_sizes = [(image.shape[-2], image.shape[-1]) for image in images]

        # Tile the images.
        crops_list, origins = self.model.tile(images)
        crops = torch.stack(crops_list)

        # Forward through all heads.
        crop_logits_dict = self.model.forward_train(crops)

        # Process each head separately.
        losses: list[Tensor] = []
        for head_name, crop_logits in crop_logits_dict.items():
            # Handle ignore class.
            if self.model.class_ignore_index is not None:
                crop_logits = crop_logits[:, :-1]

            # Un-tile the predictions.
            logits = self.model.untile(
                crop_logits=crop_logits, origins=origins, image_sizes=image_sizes
            )

            # Compute loss and update metrics per image.
            head_metrics: SemanticSegmentationTaskMetric = (
                self.val_metrics.head_metrics[  # type: ignore
                    head_name
                ]
            )
            head_loss = torch.tensor(
                0.0, device=crop_logits.device, dtype=crop_logits.dtype
            )
            for image_logits, image_mask in zip(logits, masks):
                image_logits = image_logits.unsqueeze(0)  # Add batch dimension.
                image_mask = image_mask.unsqueeze(0)  # Add batch dimension.
                loss = self.loss_fn(image_logits, image_mask)
                head_loss += loss
                head_metrics.update_with_predictions(image_logits, image_mask)
            head_loss /= len(images)
            losses.append(head_loss)

            # Accumulate loss in the head's TaskMetric (weighted by batch size).
            head_metrics.update_with_losses(
                {"loss": head_loss.detach()}, weight=len(images)
            )

        # Use sum of losses for consistency with training_step.
        loss_sum = torch.stack(losses).sum()
        return TaskStepResult(loss=loss_sum, log_dict={}, metrics=self.val_metrics)

    def set_train_mode(self) -> None:
        self.train()
        # backbone is always frozen for multihead training
        self.model.freeze_backbone()

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        if self.model_args.gradient_clip_val > 0:
            fabric.clip_gradients(
                module=self,
                optimizer=optimizer,
                max_norm=self.model_args.gradient_clip_val,
                error_if_nonfinite=False,
            )


def _format_head_name(lr: float) -> str:
    """Format learning rate into a head name.

    Args:
        lr: Learning rate value.

    Returns:
        Formatted head name, e.g., "lr0_001" for lr=0.001.
    """
    # Convert to string and replace dot with underscore.
    lr_str = f"{lr:.10f}".rstrip("0").rstrip(".")
    lr_str = lr_str.replace(".", "_")
    return f"lr{lr_str}"
