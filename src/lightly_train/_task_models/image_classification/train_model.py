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

from lightly_train._configs.validate import no_auto
from lightly_train._data.image_classification_dataset import ImageClassificationDataArgs
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._optim import optimizer_helpers
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
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


class ClassificationTaskSaveCheckpointArgs(TaskSaveCheckpointArgs):
    watch_metric: str = "auto"
    mode: Literal["min", "max"] = "max"

    def resolve_auto(
        self,
        data_args: TaskDataArgs,
    ) -> None:
        assert isinstance(data_args, ImageClassificationDataArgs)
        if self.watch_metric == "auto":
            if data_args.classification_task == "multiclass":
                self.watch_metric = "val_metric/top1_acc"
            elif data_args.classification_task == "multilabel":
                self.watch_metric = "val_metric/f1"
            else:
                raise ValueError(
                    f"Unsupported classification task: {data_args.classification_task}"
                )


class ImageClassificationTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 100_000

    save_checkpoint_args_cls: ClassVar[type[TaskSaveCheckpointArgs]] = (
        ClassificationTaskSaveCheckpointArgs
    )

    # Backbone args
    backbone_freeze: bool = False
    backbone_weights: PathLike | None = None
    backbone_args: dict[str, Any] = Field(default_factory=dict)

    gradient_clip_val: float = 3.0

    # Optim
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lr_warmup_steps: int | Literal["auto"] = "auto"

    # Metrics
    metric_log_classwise: bool = False
    metric_log_debug: bool = False
    metric_topk: list[int] = Field(default_factory=lambda: [1, 5])

    # Loss
    label_smoothing: float = 0.0

    def resolve_auto(
        self,
        total_steps: int,
        model_name: str,
        model_init_args: dict[str, Any],
    ) -> None:
        if self.lr_warmup_steps == "auto":
            if self.backbone_freeze:
                self.lr_warmup_steps = 0
            else:
                self.lr_warmup_steps = min(500, total_steps)


class ImageClassificationTrain(TrainModel):
    task = "image_classification"
    train_model_args_cls = ImageClassificationTrainArgs
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
    ) -> None:
        # Import here because old torchmetrics versions (0.8.0) don't support the
        # metrics we use. But we need old torchmetrics support for SuperGradients.
        from torchmetrics import MeanMetric, Metric, MetricCollection
        from torchmetrics.classification import (  # type: ignore[attr-defined]
            MulticlassAccuracy,
            MulticlassF1Score,
            MulticlassPrecision,
            MulticlassRecall,
            MultilabelAccuracy,
            MultilabelAUROC,
            MultilabelAveragePrecision,
            MultilabelF1Score,
            MultilabelHammingDistance,
        )

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

        # Metrics
        self.max_topk = min(
            max(model_args.metric_topk) if model_args.metric_topk else 1,
            len(data_args.included_classes),
        )
        self.val_loss = MeanMetric()

        metrics: dict[str, Metric | MetricCollection] = {}
        if self.model.classification_task == "multiclass":
            metrics.update(
                {
                    f"top{k}_acc": MulticlassAccuracy(
                        num_classes=data_args.num_included_classes, top_k=k
                    )
                    for k in model_args.metric_topk
                    if k <= self.max_topk
                }
            )
            metrics.update(
                {
                    "precision": MulticlassPrecision(
                        num_classes=data_args.num_included_classes,
                        average="macro",
                    ),
                    "recall": MulticlassRecall(
                        num_classes=data_args.num_included_classes,
                        average="macro",
                    ),
                    "f1": MulticlassF1Score(
                        num_classes=data_args.num_included_classes, average="macro"
                    ),
                }
            )
        elif self.model.classification_task == "multilabel":
            metrics.update(
                {
                    "hamming_distance": MultilabelHammingDistance(
                        num_labels=data_args.num_included_classes
                    ),
                    "f1": MultilabelF1Score(num_labels=data_args.num_included_classes),
                    "accuracy": MultilabelAccuracy(
                        num_labels=data_args.num_included_classes
                    ),
                    "auroc": MultilabelAUROC(num_labels=data_args.num_included_classes),
                    "average_precision": MultilabelAveragePrecision(
                        num_labels=data_args.num_included_classes
                    ),
                }
            )
        self.val_metrics = MetricCollection(metrics, prefix="val_metric/")

    def get_task_model(self) -> ImageClassification:
        return self.model

    def training_step(
        self, fabric: Fabric, batch: ImageClassificationBatch, step: int
    ) -> TaskStepResult:
        images = batch["image"]
        classes = batch["classes"]
        logits = self.model.forward_train(images)
        if self.model.classification_task == "multiclass":
            targets = torch.concatenate(classes)
            loss = self.criterion(logits, targets)
        elif self.model.classification_task == "multilabel":
            targets = _class_ids_to_multihot(
                class_ids=classes, num_classes=len(self.model.classes)
            )
            loss = self.criterion(logits, targets)
        else:
            raise ValueError(
                f"Unsupported classification task: {self.model.classification_task}"
            )
        log_dict = {
            "train_loss": loss.detach(),
        }
        return TaskStepResult(loss=loss, log_dict=log_dict)

    def validation_step(
        self, fabric: Fabric, batch: ImageClassificationBatch
    ) -> TaskStepResult:
        images = batch["image"]
        classes = batch["classes"]
        logits = self.model.forward_train(images)
        if self.model.classification_task == "multiclass":
            targets = torch.concatenate(classes)
            loss = self.criterion(logits, targets)
            self.val_metrics.update(logits, targets)
        elif self.model.classification_task == "multilabel":
            targets = _class_ids_to_multihot(
                class_ids=classes, num_classes=len(self.model.classes)
            )
            loss = self.criterion(logits, targets)
            self.val_metrics.update(logits, targets.int())
        else:
            raise ValueError(
                f"Unsupported classification task: {self.model.classification_task}"
            )
        self.val_loss.update(loss, weight=len(images))
        log_dict = {
            "val_loss": loss.detach(),
            **dict(self.val_metrics.items()),
        }
        return TaskStepResult(loss=loss, log_dict=log_dict)

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
            weight_decay=self.model_args.weight_decay,
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
        if self.model_args.gradient_clip_val > 0:
            fabric.clip_gradients(
                module=self,
                optimizer=optimizer,
                max_norm=self.model_args.gradient_clip_val,
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
