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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

from lightly_train._configs.validate import no_auto
from lightly_train._data.image_classification_dataset import ImageClassificationDataArgs
from lightly_train._metrics.classification.task_metric import (
    ClassificationTaskMetric,
    ClassificationTaskMetricArgs,
    MulticlassClassificationTaskMetricArgs,
    MultilabelClassificationTaskMetricArgs,
)
from lightly_train._metrics.multihead_task_metric import MultiheadTaskMetric
from lightly_train._optim import optimizer_helpers
from lightly_train._task_models.image_classification_multihead.task_model import (
    ImageClassificationMultihead,
)
from lightly_train._task_models.image_classification_multihead.transforms import (
    ImageClassificationMultiheadTrainTransform,
    ImageClassificationMultiheadTrainTransformArgs,
    ImageClassificationMultiheadValTransform,
    ImageClassificationMultiheadValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train._torch_compile import TorchCompileArgs
from lightly_train.types import ImageClassificationBatch, PathLike


class ImageClassificationMultiheadTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 128
    default_steps: ClassVar[int] = 100_000

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

    # Loss
    label_smoothing: float = 0.0

    @model_validator(mode="after")
    def _convert_lr_to_list(self) -> ImageClassificationMultiheadTrainArgs:
        """Convert float lr to single-element list."""
        if isinstance(self.lr, float):
            self.lr = [self.lr]
        return self


class ImageClassificationMultiheadTrain(TrainModel):
    task = "image_classification_multihead"
    train_model_args_cls = ImageClassificationMultiheadTrainArgs
    # Needs type ignore because ClassificationTaskMetricArgs is not a pure type but
    # a union of MulticlassClassificationTaskMetricArgs and MultilabelClassificationTaskMetricArgs
    # This is properly handled in get_metric_args in train_task_helpers.py
    task_metric_args_cls = ClassificationTaskMetricArgs  # type: ignore[assignment]
    task_model_cls = ImageClassificationMultihead
    train_transform_cls = ImageClassificationMultiheadTrainTransform
    val_transform_cls = ImageClassificationMultiheadValTransform
    torch_compile_args_cls = TorchCompileArgs

    def __init__(
        self,
        *,
        model_name: str,
        model_args: ImageClassificationMultiheadTrainArgs,
        data_args: ImageClassificationDataArgs,
        train_transform_args: ImageClassificationMultiheadTrainTransformArgs,
        val_transform_args: ImageClassificationMultiheadValTransformArgs,
        load_weights: bool,
        metric_args: MulticlassClassificationTaskMetricArgs
        | MultilabelClassificationTaskMetricArgs,
    ) -> None:
        super().__init__()
        image_size = no_auto(val_transform_args.image_size)
        normalize = no_auto(val_transform_args.normalize)

        self.model_args = model_args
        self.classification_task = data_args.classification_task
        self.num_classes = data_args.num_included_classes

        # Convert lr to list and store for optimizer creation.
        self.lrs = model_args.lr if isinstance(model_args.lr, list) else [model_args.lr]

        # Generate head names from learning rates.
        head_names = [_format_head_name(lr) for lr in self.lrs]

        self.model = ImageClassificationMultihead(
            model=model_name,
            classes=data_args.included_classes,
            head_names=head_names,
            image_size=image_size,
            image_normalize=normalize.model_dump(),
            backbone_weights=model_args.backbone_weights,
            backbone_args=model_args.backbone_args,
            load_weights=load_weights,
        )

        self.criterion: Module
        if self.classification_task == "multiclass":
            self.criterion = CrossEntropyLoss(
                label_smoothing=model_args.label_smoothing
            )
        elif self.classification_task == "multilabel":
            self.criterion = BCEWithLogitsLoss()
        else:
            raise ValueError(
                f"Unsupported classification task: {self.classification_task}"
            )

        class_names = list(data_args.included_classes.values())

        train_head_metrics: dict[str, ClassificationTaskMetric] = {}
        for lr in self.lrs:
            head_name = _format_head_name(lr)
            train_head_metrics[head_name] = ClassificationTaskMetric(
                task_metric_args=metric_args,
                split="train",
                class_names=class_names,
                loss_names=["loss"],
            )
        self.train_metrics: MultiheadTaskMetric = MultiheadTaskMetric(
            head_metrics=train_head_metrics,
        )

        val_head_metrics: dict[str, ClassificationTaskMetric] = {}
        for lr in self.lrs:
            head_name = _format_head_name(lr)
            val_head_metrics[head_name] = ClassificationTaskMetric(
                task_metric_args=metric_args,
                split="val",
                class_names=class_names,
                loss_names=["loss"],
            )
        self.val_metrics: MultiheadTaskMetric = MultiheadTaskMetric(
            head_metrics=val_head_metrics,
        )

    def get_task_model(self) -> ImageClassificationMultihead:
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

            # Filter out parameters with requires_grad=False (shouldn't be any for heads).
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
                        "name": f"params_{head_name}",
                        "params": params_wd,
                        "lr": lr_scaled,
                    }
                )
            if params_no_wd:
                params.append(
                    {
                        "name": f"params_no_weight_decay_{head_name}",
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
        self,
        fabric: Fabric,
        batch: ImageClassificationBatch,
        step: int,
    ) -> TaskStepResult:
        images = batch["image"]
        classes = batch["classes"]
        logits_dict = self.model.forward_train(images)

        # Prepare targets based on classification task.
        if self.classification_task == "multiclass":
            targets = torch.concatenate(classes)
        elif self.classification_task == "multilabel":
            targets = _class_ids_to_multihot(
                class_ids=classes, num_classes=self.num_classes
            )
        else:
            raise ValueError(
                f"Unsupported classification task: {self.classification_task}"
            )

        losses: list[Tensor] = []
        for head_name, logits in logits_dict.items():
            loss = self.criterion(logits, targets)
            losses.append(loss)

            head_metrics: ClassificationTaskMetric = self.train_metrics.head_metrics[  # type: ignore
                head_name
            ]
            head_metrics.update_with_predictions(logits, targets.int())
            head_metrics.update_with_losses({"loss": loss.detach()}, weight=len(images))

        loss = torch.stack(losses).sum()
        return TaskStepResult(loss=loss, log_dict={}, metrics=self.train_metrics)

    def validation_step(
        self, fabric: Fabric, batch: ImageClassificationBatch
    ) -> TaskStepResult:
        images = batch["image"]
        classes = batch["classes"]
        logits_dict = self.model.forward_train(images)

        # Prepare targets based on classification task.
        if self.classification_task == "multiclass":
            targets = torch.concatenate(classes)
        elif self.classification_task == "multilabel":
            targets = _class_ids_to_multihot(
                class_ids=classes, num_classes=self.num_classes
            )
        else:
            raise ValueError(
                f"Unsupported classification task: {self.classification_task}"
            )

        losses: list[Tensor] = []
        for head_name, logits in logits_dict.items():
            loss = self.criterion(logits, targets)
            losses.append(loss)

            head_metrics: ClassificationTaskMetric = self.val_metrics.head_metrics[  # type: ignore
                head_name
            ]
            head_metrics.update_with_predictions(logits, targets.int())
            head_metrics.update_with_losses({"loss": loss.detach()}, weight=len(images))

        loss = torch.stack(losses).sum()
        return TaskStepResult(loss=loss, log_dict={}, metrics=self.val_metrics)

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


def _class_ids_to_multihot(class_ids: list[Tensor], num_classes: int) -> Tensor:
    """Convert list of class ID tensors to a single multi-hot encoded tensor."""
    row = torch.repeat_interleave(
        torch.arange(len(class_ids), device=class_ids[0].device),
        torch.tensor([t.numel() for t in class_ids], device=class_ids[0].device),
    )
    col = torch.cat(class_ids)
    y = torch.zeros(len(class_ids), num_classes, device=class_ids[0].device)
    y[row, col] = 1
    return y


def _format_head_name(lr: float) -> str:
    """Format learning rate for head name by removing trailing zeros.

    Args:
        lr: Learning rate value to format.

    Returns:
        Formatted head name like "lr0_001" for lr=0.001.
    """
    # Format the learning rate without trailing zeros and decimal point.
    lr_str = f"{lr:.10f}".rstrip("0").rstrip(".")
    # Replace dots with underscores to make it a valid module name.
    lr_str = lr_str.replace(".", "_")
    return f"lr{lr_str}"
