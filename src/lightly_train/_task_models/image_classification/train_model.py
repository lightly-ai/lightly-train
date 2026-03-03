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
                self.watch_metric = "val_metric/top1_acc_micro"
            elif data_args.classification_task == "multilabel":
                self.watch_metric = "val_metric/f1_micro"
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

    gradient_clip_val: float | Literal["auto"] = "auto"

    # Optim
    lr: float = 3e-4
    weight_decay: float | Literal["auto"] = "auto"
    lr_warmup_steps: int | Literal["auto"] = "auto"

    # Metrics
    metrics: dict[str, dict[str, Any]] | Literal["auto"] = "auto"
    metrics_classwise: dict[str, dict[str, Any]] | None = None
    metric_log_classwise: bool = False
    metric_log_debug: bool = False

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
        if self.metrics == "auto":
            assert isinstance(data_args, ImageClassificationDataArgs)
            if data_args.classification_task == "multiclass":
                self.metrics = {
                    "accuracy": {"topk": [1, 5], "average": ["micro"]},
                    "f1": {"average": ["micro"]},
                    "precision": {"average": ["micro"]},
                    "recall": {"average": ["micro"]},
                }
            elif data_args.classification_task == "multilabel":
                self.metrics = {
                    "hamming_distance": {"threshold": 0.5, "average": ["micro"]},
                    "accuracy": {"threshold": 0.5, "average": ["micro"]},
                    "f1": {"threshold": 0.5, "average": ["micro"]},
                    "auroc": {"thresholds": None, "average": ["micro"]},
                    "average_precision": {"thresholds": None, "average": ["micro"]},
                }
            else:
                raise ValueError(
                    f"Unsupported classification task: {data_args.classification_task}"
                )


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
        from torchmetrics import ClasswiseWrapper, MeanMetric, MetricCollection

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
        self.val_loss = MeanMetric()

        # Create metrics from configuration
        from torchmetrics import Metric

        metrics: dict[str, Metric] = {}
        for metric_name, metric_config in no_auto(model_args.metrics).items():
            metrics.update(
                _create_metric(
                    metric_name=metric_name,
                    metric_config=metric_config,
                    num_classes=data_args.num_included_classes,
                    classification_task=self.model.classification_task,
                )
            )
        self.val_metrics = MetricCollection(metrics, prefix="val_metric/")  # type: ignore

        # Create classwise metrics if enabled
        self.val_metrics_classwise: MetricCollection | None
        if model_args.metric_log_classwise:
            classwise_metrics: dict[str, Metric] = {}
            # If metrics_classwise is None, use filtered metrics from main metrics
            if model_args.metrics_classwise is None:
                metrics_classwise_config = _filter_classwise_metrics(
                    no_auto(model_args.metrics),
                    classification_task=self.model.classification_task,
                )
            else:
                metrics_classwise_config = model_args.metrics_classwise

            class_labels = list(data_args.included_classes.values())
            for metric_name, metric_config in metrics_classwise_config.items():
                base_metrics = _create_metric(
                    metric_name=metric_name,
                    metric_config=metric_config,
                    num_classes=data_args.num_included_classes,
                    classification_task=self.model.classification_task,
                    classwise=True,
                )
                for key, base_metric in base_metrics.items():
                    # Type ignore because old torchmetrics versions (0.8) don't support
                    # the `prefix` argument. We only use the old versions for
                    # SuperGradients support.
                    classwise_metrics[key] = ClasswiseWrapper(  # type: ignore[call-arg]
                        base_metric,
                        prefix="_",
                        labels=class_labels,
                    )
            self.val_metrics_classwise = MetricCollection(
                classwise_metrics,  # type: ignore
                prefix="val_metric_classwise/",
            )
        else:
            self.val_metrics_classwise = None

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
            if self.val_metrics_classwise is not None:
                self.val_metrics_classwise.update(logits, targets)
        elif self.model.classification_task == "multilabel":
            targets = _class_ids_to_multihot(
                class_ids=classes, num_classes=len(self.model.classes)
            )
            loss = self.criterion(logits, targets)
            self.val_metrics.update(logits, targets.int())
            if self.val_metrics_classwise is not None:
                self.val_metrics_classwise.update(logits, targets.int())
        else:
            raise ValueError(
                f"Unsupported classification task: {self.model.classification_task}"
            )
        self.val_loss.update(loss, weight=len(images))
        log_dict = {
            "val_loss": loss.detach(),
            **dict(self.val_metrics.items()),
        }
        if self.val_metrics_classwise is not None:
            log_dict.update(dict(self.val_metrics_classwise.items()))
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


def _create_metric(
    metric_name: str,
    metric_config: dict[str, Any],
    num_classes: int,
    classification_task: Literal["multiclass", "multilabel"],
    classwise: bool = False,
) -> dict[str, Any]:
    """Create metrics from configuration.

    Args:
        metric_name: Name of the metric (e.g. "accuracy", "f1").
        metric_config: Configuration dictionary for the metric.
        num_classes: Number of classes.
        classification_task: Classification task type.
        classwise: Whether to create classwise metrics.

    Returns:
        Dictionary mapping metric names to metric instances.
    """
    from torchmetrics import Metric
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

    metrics: dict[str, Metric] = {}
    average_list = metric_config.get("average", ["micro"])

    if classification_task == "multiclass":
        if metric_name == "accuracy":
            topk_list = metric_config.get("topk", [1])
            for k in topk_list:
                if k > num_classes:
                    continue
                for average in average_list:
                    key = f"top{k}_acc_{average}"
                    metrics[key] = MulticlassAccuracy(
                        num_classes=num_classes,
                        top_k=k,
                        average="none" if classwise else average,
                    )
        elif metric_name == "f1":
            for average in average_list:
                key = f"f1_{average}"
                metrics[key] = MulticlassF1Score(
                    num_classes=num_classes,
                    average="none" if classwise else average,
                )
        elif metric_name == "precision":
            for average in average_list:
                key = f"precision_{average}"
                metrics[key] = MulticlassPrecision(
                    num_classes=num_classes,
                    average="none" if classwise else average,
                )
        elif metric_name == "recall":
            for average in average_list:
                key = f"recall_{average}"
                metrics[key] = MulticlassRecall(
                    num_classes=num_classes,
                    average="none" if classwise else average,
                )
        else:
            raise ValueError(
                f"Unsupported metric '{metric_name}' for {classification_task}"
            )
    elif classification_task == "multilabel":
        if metric_name == "hamming_distance":
            threshold = metric_config.get("threshold", 0.5)
            for average in average_list:
                key = f"hamming_distance_{average}"
                metrics[key] = MultilabelHammingDistance(
                    num_labels=num_classes,
                    threshold=threshold,
                    average="none" if classwise else average,
                )
        elif metric_name == "accuracy":
            threshold = metric_config.get("threshold", 0.5)
            for average in average_list:
                key = f"accuracy_{average}"
                metrics[key] = MultilabelAccuracy(
                    num_labels=num_classes,
                    threshold=threshold,
                    average="none" if classwise else average,
                )
        elif metric_name == "f1":
            threshold = metric_config.get("threshold", 0.5)
            for average in average_list:
                key = f"f1_{average}"
                metrics[key] = MultilabelF1Score(
                    num_labels=num_classes,
                    threshold=threshold,
                    average="none" if classwise else average,
                )
        elif metric_name == "auroc":
            thresholds = metric_config.get("thresholds", None)
            for average in average_list:
                key = f"auroc_{average}"
                metrics[key] = MultilabelAUROC(
                    num_labels=num_classes,
                    thresholds=thresholds,
                    average="none" if classwise else average,
                )
        elif metric_name == "average_precision":
            thresholds = metric_config.get("thresholds", None)
            for average in average_list:
                key = f"average_precision_{average}"
                metrics[key] = MultilabelAveragePrecision(
                    num_labels=num_classes,
                    thresholds=thresholds,
                    average="none" if classwise else average,
                )
        else:
            raise ValueError(
                f"Unsupported metric '{metric_name}' for {classification_task}"
            )
    else:
        raise ValueError(f"Unsupported classification task: {classification_task}")

    return metrics


def _filter_classwise_metrics(
    metrics: dict[str, dict[str, Any]],
    classification_task: Literal["multiclass", "multilabel"],
) -> dict[str, dict[str, Any]]:
    """Filter metrics that make sense for classwise computation.

    Args:
        metrics: Metrics configuration dictionary.
        classification_task: Classification task type.

    Returns:
        Filtered metrics configuration dictionary.
    """
    if classification_task == "multiclass":
        # Exclude topk accuracy for classwise
        return {
            k: {key: val for key, val in v.items() if key != "topk"}
            for k, v in metrics.items()
            if k != "accuracy"  # Exclude accuracy entirely for multiclass
        }
    elif classification_task == "multilabel":
        # For multilabel, all metrics make sense classwise
        return metrics.copy()
    else:
        raise ValueError(f"Unsupported classification task: {classification_task}")


def _class_ids_to_multihot(class_ids: list[Tensor], num_classes: int) -> Tensor:
    row = torch.repeat_interleave(
        torch.arange(len(class_ids), device=class_ids[0].device),
        torch.tensor([t.numel() for t in class_ids], device=class_ids[0].device),
    )
    col = torch.cat(class_ids)
    y = torch.zeros(len(class_ids), num_classes, device=class_ids[0].device)
    y[row, col] = 1
    return y
