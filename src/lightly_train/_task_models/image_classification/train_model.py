#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import torch
from lightly.utils.scheduler import CosineWarmupScheduler
from lightning_fabric import Fabric
from pydantic import Field
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from lightly_train import _torch_compile
from lightly_train._configs.validate import no_auto
from lightly_train._data.image_classification_dataset import ImageClassificationDataArgs
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._metrics.classification.task_metric import (
    ClassificationTaskMetric,
    ClassificationTaskMetricArgs,
)
from lightly_train._optim import optimizer_helpers
from lightly_train._task_models.image_classification.task_model import (
    ImageClassification,
)
from lightly_train._task_models.image_classification.transforms import (
    ImageClassificationTrainTransform,
    ImageClassificationTrainTransformArgs,
    ImageClassificationValTransform,
    ImageClassificationValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train.types import (
    ImageClassificationBatch,
    PathLike,
)


class ImageClassificationTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 100_000

    # Backbone args
    backbone_freeze: bool = False
    backbone_weights: PathLike | None = None
    backbone_args: dict[str, Any] = Field(default_factory=dict)

    gradient_clip_val: float | Literal["auto"] = "auto"

    # Optim
    lr: float = 3e-4
    weight_decay: float | Literal["auto"] = "auto"
    lr_warmup_steps: int | Literal["auto"] = "auto"

    # Loss
    label_smoothing: float = 0.0

    def resolve_auto(
        self,
        total_steps: int,
        model_name: str,
        model_init_args: dict[str, Any],
        data_args: TaskDataArgs,
    ) -> None:
        if self.weight_decay == "auto":
            if self.backbone_freeze:
                self.weight_decay = 0.0
            else:
                self.weight_decay = 1e-4
        if self.lr_warmup_steps == "auto":
            if self.backbone_freeze:
                self.lr_warmup_steps = 0
            else:
                self.lr_warmup_steps = min(500, total_steps)
        if self.gradient_clip_val == "auto":
            if self.backbone_freeze:
                self.gradient_clip_val = 0.0
            else:
                self.gradient_clip_val = 3.0


class ImageClassificationTrain(TrainModel):
    task = "image_classification"
    train_model_args_cls = ImageClassificationTrainArgs
    # Needs type ignore because ClassificationTaskMetricArgs is not a pure type but
    # a union of MulticlassClassificationTaskMetricArgs and MultilabelClassificationTaskMetricArgs
    # This is properly handled in get_metric_args in train_task_helpers.py
    task_metric_args_cls = ClassificationTaskMetricArgs  # type: ignore[assignment]
    task_model_cls = ImageClassification
    train_transform_cls = ImageClassificationTrainTransform
    val_transform_cls = ImageClassificationValTransform

    def __init__(
        self,
        *,
        model_name: str,
        model_args: ImageClassificationTrainArgs,
        data_args: ImageClassificationDataArgs,
        train_transform_args: ImageClassificationTrainTransformArgs,
        val_transform_args: ImageClassificationValTransformArgs,
        load_weights: bool,
        metric_args: ClassificationTaskMetricArgs,
    ) -> None:
        # Import here because old torchmetrics versions (0.8.0) don't support the
        # metrics we use. But we need old torchmetrics support for SuperGradients.

        super().__init__()
        image_size = no_auto(val_transform_args.image_size)
        normalize = no_auto(val_transform_args.normalize)

        self.model_args = model_args
        self.model = ImageClassification(
            model=model_name,
            classes=data_args.included_classes,
            classification_task=data_args.classification_task,
            # TODO(Guarin, 02/26): Check drop path rate for DINO models.
            image_size=image_size,
            image_normalize=normalize.model_dump(),
            backbone_freeze=self.model_args.backbone_freeze,
            backbone_weights=model_args.backbone_weights,
            backbone_args=model_args.backbone_args,
            load_weights=load_weights,
        )

        self.criterion: Module
        if self.model.classification_task == "multiclass":
            self.criterion = CrossEntropyLoss(
                label_smoothing=model_args.label_smoothing
            )
        elif self.model.classification_task == "multilabel":
            self.criterion = BCEWithLogitsLoss()
        else:
            raise ValueError(
                f"Unsupported classification task: {self.model.classification_task}"
            )

        self.val_metrics = ClassificationTaskMetric(
            task_metric_args=metric_args,
            split="val",
            class_names=list(data_args.included_classes.values()),
            loss_names=["loss"],
        )
        self.train_metrics = ClassificationTaskMetric(
            task_metric_args=metric_args,
            split="train",
            class_names=list(data_args.included_classes.values()),
            loss_names=["loss"],
        )

    def get_task_model(self) -> ImageClassification:
        return self.model

    def forward(self, images: Tensor) -> Tensor:
        return self.model.forward_train(images)

    @_torch_compile.disable_compile
    def training_step(
        self, fabric: Fabric, batch: ImageClassificationBatch, step: int
    ) -> TaskStepResult:
        images = batch["image"]
        classes = batch["classes"]
        logits = self(images)
        if self.model.classification_task == "multiclass":
            targets = torch.concatenate(classes)
            loss = self.criterion(logits, targets)
        elif self.model.classification_task == "multilabel":
            targets = _class_ids_to_multihot(
                class_ids=classes, num_classes=len(self.model.classes)
            )
            loss = self.criterion(logits, targets)
            targets = targets.int()  # For metrics
        else:
            raise ValueError(
                f"Unsupported classification task: {self.model.classification_task}"
            )

        self.train_metrics.update(logits, targets)
        self.train_metrics.update_loss({"loss": loss.detach()}, weight=len(images))
        return TaskStepResult(loss=loss, log_dict={}, metrics=self.train_metrics)


    def validation_step(
        self, fabric: Fabric, batch: ImageClassificationBatch
    ) -> TaskStepResult:
        images = batch["image"]
        classes = batch["classes"]
        logits = self(images)
        if self.model.classification_task == "multiclass":
            targets = torch.concatenate(classes)
            loss = self.criterion(logits, targets)
        elif self.model.classification_task == "multilabel":
            targets = _class_ids_to_multihot(
                class_ids=classes, num_classes=len(self.model.classes)
            )
            loss = self.criterion(logits, targets)
            targets = targets.int()  # For metrics
        else:
            raise ValueError(
                f"Unsupported classification task: {self.model.classification_task}"
            )
        self.val_metrics.update(logits, targets)
        self.val_metrics.update_loss({"loss": loss.detach()}, weight=len(images))
        return TaskStepResult(loss=loss, log_dict={}, metrics=self.val_metrics)

    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
    ) -> tuple[Optimizer, LRScheduler]:
        params_wd, params_no_wd = optimizer_helpers.get_weight_decay_parameters([self])
        params_wd = [p for p in params_wd if p.requires_grad]
        params_no_wd = [p for p in params_no_wd if p.requires_grad]
        params: list[dict[str, Any]] = [
            {"name": "params", "params": params_wd},
            {
                "name": "no_weight_decay",
                "params": params_no_wd,
                "weight_decay": 0.0,
            },
        ]
        lr = self.model_args.lr * math.sqrt(
            global_batch_size / self.model_args.default_batch_size
        )
        optimizer = AdamW(
            params=params,
            lr=lr,
            weight_decay=no_auto(self.model_args.weight_decay),
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=no_auto(self.model_args.lr_warmup_steps),
            max_epochs=total_steps,
        )
        return optimizer, scheduler

    def set_train_mode(self) -> None:
        self.train()
        if self.model_args.backbone_freeze:
            self.model.freeze_backbone()

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        if no_auto(self.model_args.gradient_clip_val) > 0:
            fabric.clip_gradients(
                module=self,
                optimizer=optimizer,
                max_norm=no_auto(self.model_args.gradient_clip_val),
            )


def _class_ids_to_multihot(class_ids: list[Tensor], num_classes: int) -> Tensor:
    row = torch.repeat_interleave(
        torch.arange(len(class_ids), device=class_ids[0].device),
        torch.tensor([t.numel() for t in class_ids], device=class_ids[0].device),
    )
    col = torch.cat(class_ids)
    y = torch.zeros(len(class_ids), num_classes, device=class_ids[0].device)
    y[row, col] = 1
    return y
