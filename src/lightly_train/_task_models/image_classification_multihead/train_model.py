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
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._optim import optimizer_helpers
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
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
from lightly_train.types import ImageClassificationBatch, PathLike


def _format_head_name(lr: float) -> str:
    """Format learning rate for head name by removing trailing zeros.

    Args:
        lr: Learning rate value to format.

    Returns:
        Formatted head name like "head_lr0_001" for lr=0.001.
    """
    # Format the learning rate without trailing zeros and decimal point.
    lr_str = f"{lr:.10f}".rstrip("0").rstrip(".")
    # Replace dots with underscores to make it a valid module name.
    lr_str = lr_str.replace(".", "_")
    return f"head_lr{lr_str}"


class ImageClassificationMultiheadSaveCheckpointArgs(TaskSaveCheckpointArgs):
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


class ImageClassificationMultiheadTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 128
    default_steps: ClassVar[int] = 100_000

    save_checkpoint_args_cls: ClassVar[type[TaskSaveCheckpointArgs]] = (
        ImageClassificationMultiheadSaveCheckpointArgs
    )

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

    # Metrics
    metrics: dict[str, dict[str, Any]] | Literal["auto"] = "auto"
    metrics_classwise: dict[str, dict[str, Any]] | None = None
    metric_log_classwise: bool = False

    # Loss
    label_smoothing: float = 0.0

    @model_validator(mode="after")
    def _convert_lr_to_list(self) -> ImageClassificationMultiheadTrainArgs:
        """Convert float lr to single-element list."""
        if isinstance(self.lr, float):
            self.lr = [self.lr]
        return self

    def resolve_auto(
        self,
        total_steps: int,
        model_name: str,
        model_init_args: dict[str, Any],
        data_args: TaskDataArgs,
    ) -> None:
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


class ImageClassificationMultiheadTrain(TrainModel):
    task = "image_classification_multihead"
    train_model_args_cls = ImageClassificationMultiheadTrainArgs
    task_model_cls = ImageClassificationMultihead
    train_transform_cls = ImageClassificationMultiheadTrainTransform
    val_transform_cls = ImageClassificationMultiheadValTransform

    def __init__(
        self,
        *,
        model_name: str,
        model_args: ImageClassificationMultiheadTrainArgs,
        data_args: ImageClassificationDataArgs,
        train_transform_args: ImageClassificationMultiheadTrainTransformArgs,
        val_transform_args: ImageClassificationMultiheadValTransformArgs,
        load_weights: bool,
    ) -> None:
        # Import here because old torchmetrics versions (0.8.0) don't support the
        # metrics we use. But we need old torchmetrics support for SuperGradients.
        from torchmetrics import ClasswiseWrapper, MeanMetric, MetricCollection

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

        # Validation loss tracking: overall and per-head.
        self.val_loss = MeanMetric()
        self.val_loss_per_head: dict[str, MeanMetric] = {}
        for lr in self.lrs:
            head_name = _format_head_name(lr)
            self.val_loss_per_head[head_name] = MeanMetric()

        # Create metrics from configuration.
        from torchmetrics import Metric

        base_metrics: dict[str, Metric] = {}
        for metric_name, metric_config in no_auto(model_args.metrics).items():
            base_metrics.update(
                _create_metric(
                    metric_name=metric_name,
                    metric_config=metric_config,
                    num_classes=data_args.num_included_classes,
                    classification_task=self.classification_task,
                )
            )

        # Create per-head metric collections with suffix _head_lr{value}.
        self.val_metrics_per_head: dict[str, MetricCollection] = {}
        for lr in self.lrs:
            head_name = _format_head_name(lr)
            # Add suffix to each metric key.
            head_metrics: dict[str, Metric] = {}
            for key, metric in base_metrics.items():
                suffixed_key = f"{key}_{head_name}"
                head_metrics[suffixed_key] = metric.clone()
            # Type ignore because old torchmetrics versions (0.8) don't have the
            # correct type annotations for MetricCollection.
            self.val_metrics_per_head[head_name] = MetricCollection(
                head_metrics,  # type: ignore[arg-type]
                prefix="val_metric_head/",
            )

        # Create classwise metrics if enabled.
        self.val_metrics_classwise: MetricCollection | None
        self.val_metrics_classwise_per_head: dict[str, MetricCollection] | None
        if model_args.metric_log_classwise:
            classwise_metrics: dict[str, Metric] = {}
            # If metrics_classwise is None, use filtered metrics from main metrics.
            if model_args.metrics_classwise is None:
                metrics_classwise_config = _filter_classwise_metrics(
                    no_auto(model_args.metrics),
                    classification_task=self.classification_task,
                )
            else:
                metrics_classwise_config = model_args.metrics_classwise

            class_labels = list(data_args.included_classes.values())
            for metric_name, metric_config in metrics_classwise_config.items():
                base_metrics_classwise = _create_metric(
                    metric_name=metric_name,
                    metric_config=metric_config,
                    num_classes=data_args.num_included_classes,
                    classification_task=self.classification_task,
                    classwise=True,
                )
                for key, base_metric in base_metrics_classwise.items():
                    # Type ignore because old torchmetrics versions (0.8) don't support
                    # the `prefix` argument. We only use the old versions for
                    # SuperGradients support.
                    classwise_metrics[key] = ClasswiseWrapper(  # type: ignore[call-arg]
                        base_metric,
                        prefix="_",
                        labels=class_labels,
                    )

            # Create per-head classwise metrics.
            self.val_metrics_classwise_per_head = {}
            for lr in self.lrs:
                head_name = _format_head_name(lr)
                head_classwise_metrics: dict[str, Metric] = {}
                for key, metric in classwise_metrics.items():
                    suffixed_key = f"{key}_{head_name}"
                    head_classwise_metrics[suffixed_key] = metric.clone()
                # Type ignore because old torchmetrics versions (0.8) don't have the
                # correct type annotations for MetricCollection.
                self.val_metrics_classwise_per_head[head_name] = MetricCollection(
                    head_classwise_metrics,  # type: ignore[arg-type]
                    prefix="val_metric_classwise_head/",
                )
        else:
            self.val_metrics_classwise_per_head = None

    def get_task_model(self) -> ImageClassificationMultihead:
        return self.model

    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
    ) -> tuple[Optimizer, LRScheduler]:
        # Create parameter groups for each head with independent learning rates.
        params: list[dict[str, Any]] = []

        for idx, lr in enumerate(self.lrs):
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
        self,
        fabric: Fabric,
        batch: ImageClassificationBatch,
        step: int,
    ) -> TaskStepResult:
        import torch

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

        # Compute loss for each head.
        losses: list[Tensor] = []
        log_dict: dict[str, Any] = {}
        for head_name, logits in logits_dict.items():
            loss = self.criterion(logits, targets)
            losses.append(loss)
            log_dict[f"train_loss_{head_name}"] = loss.detach()

        # Sum losses for backprop.
        loss_sum = torch.stack(losses).sum()
        # Mean loss for logging.
        loss_mean = torch.stack(losses).mean()
        log_dict["train_loss"] = loss_mean.detach()

        return TaskStepResult(loss=loss_sum, log_dict=log_dict)

    def validation_step(
        self, fabric: Fabric, batch: ImageClassificationBatch
    ) -> TaskStepResult:
        import torch

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
            targets = targets.int()
        else:
            raise ValueError(
                f"Unsupported classification task: {self.classification_task}"
            )

        # Process each head: compute loss and update metrics.
        losses: list[Tensor] = []
        for head_name, logits in logits_dict.items():
            loss = self.criterion(logits, targets)
            losses.append(loss)

            # Update per-head loss tracker.
            self.val_loss_per_head[head_name].update(loss, weight=len(images))

            # Update per-head metrics.
            if self.classification_task == "multiclass":
                self.val_metrics_per_head[head_name].update(logits, targets)
            else:  # multilabel
                self.val_metrics_per_head[head_name].update(logits, targets)

            # Update per-head classwise metrics if enabled.
            if self.val_metrics_classwise_per_head is not None:
                self.val_metrics_classwise_per_head[head_name].update(logits, targets)

        # Update overall loss tracker (mean of all heads).
        loss_mean = torch.stack(losses).mean()
        self.val_loss.update(loss_mean, weight=len(images))

        # Build comprehensive log_dict.
        log_dict: dict[str, Any] = {}

        # Mean loss across all heads.
        log_dict["val_loss"] = loss_mean.detach()

        # Per-head losses.
        for head_name in logits_dict.keys():
            log_dict[f"val_loss_{head_name}"] = self.val_loss_per_head[
                head_name
            ].compute()

        # Per-head metrics.
        for head_name in logits_dict.keys():
            log_dict.update(dict(self.val_metrics_per_head[head_name].items()))

        # Classwise metrics if enabled.
        if self.val_metrics_classwise_per_head is not None:
            for head_name in logits_dict.keys():
                log_dict.update(
                    dict(self.val_metrics_classwise_per_head[head_name].items())
                )

        # Use sum of losses for consistency with training_step.
        loss_sum = torch.stack(losses).sum()
        return TaskStepResult(loss=loss_sum, log_dict=log_dict)

    def set_train_mode(self) -> None:
        self.train()
        self.model.freeze_backbone()

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        if self.model_args.gradient_clip_val > 0:
            fabric.clip_gradients(
                module=self,
                optimizer=optimizer,
                max_norm=self.model_args.gradient_clip_val,
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
        # Exclude topk accuracy for classwise.
        return {
            k: {key: val for key, val in v.items() if key != "topk"}
            for k, v in metrics.items()
            if k != "accuracy"  # Exclude accuracy entirely for multiclass.
        }
    elif classification_task == "multilabel":
        # For multilabel, all metrics make sense classwise.
        return metrics.copy()
    else:
        raise ValueError(f"Unsupported classification task: {classification_task}")


def _class_ids_to_multihot(class_ids: list[Tensor], num_classes: int) -> Tensor:
    import torch

    row = torch.repeat_interleave(
        torch.arange(len(class_ids), device=class_ids[0].device),
        torch.tensor([t.numel() for t in class_ids], device=class_ids[0].device),
    )
    col = torch.cat(class_ids)
    y = torch.zeros(len(class_ids), num_classes, device=class_ids[0].device)
    y[row, col] = 1
    return y
