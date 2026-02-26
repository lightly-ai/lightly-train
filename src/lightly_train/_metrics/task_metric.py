#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch.nn import Module
from torchmetrics import Metric

from lightly_train._configs.config import PydanticConfig
from lightly_train._metrics.metric_args import MetricArgs

# ## Metrics Structure Overview
#
# The metrics system has three layers:
#
# 1. **MetricArgs** (Pydantic): Configuration for individual metrics
#    - Example: MulticlassAccuracyArgs(topk=[1, 5], average=["micro", "macro"])
#
# 2. **TaskMetricArgs** (Pydantic): Configuration for all metrics in a task
#    - Example: MulticlassClassificationTaskMetricArgs(
#         accuracy=..., f1=..., precision=...
#     )
#    - Groups related MetricArgs for a task. This is what the user configures
#
# 3. **TaskMetric** : Class that stores actual Metric instances
#    - Example: ClassificationTaskMetric(metric_args, class_names=[...], prefix="val_metric/")
#    - Stores actual torchmetrics instances (MetricCollection, MeanMetric, etc.)
#    - Provides .update(), .update_loss(), and .reset() for updating metrics during training/validation
#    - Provides .compute() for logging
#    - All metric logic for a task must be contained in this class. The training loop
#      only interacts with TaskMetric objects, not individual Metric instances.
#


@dataclass
class MetricComputeResult:
    """Result of computing all metrics.

    Attributes:
        metrics:
            Dictionary mapping metric names to computed float values.
            For multihead metrics, contains per-head keys (e.g.,
            "val_metric_head/miou_lr0_001") and best-head top-level keys
            (e.g., "val_metric/miou").
        watch_metric:
            The key of the metric used for model selection
            (e.g., "val_metric/miou"). Used by the training loop for
            checkpointing. None if the watch metric is not found in the
            computed metrics (e.g., a train metric watching a val key).
        watch_metric_value:
            The value of the watch_metric metric.
            None if watch_metric is None.
        watch_metric_mode:
            Whether to maximise ("max") or minimise ("min") the watch metric.
            None if watch_metric is None.
        best_head_name:
            Name of the best head (e.g., "lr0_01").
            None for single-head metrics.
        best_head_metrics:
            Metrics of the best head with top-level keys
            (no head suffix). None for single-head metrics.
    """

    metrics: dict[str, float]
    watch_metric: str | None
    watch_metric_value: float | None
    watch_metric_mode: Literal["min", "max"] | None
    best_head_name: str | None
    best_head_metrics: dict[str, float] | None


class TaskMetricArgs(PydanticConfig):
    """Base class for task-specific metrics collection configurations."""

    # Metric to watch for best model selection. E.g. "val_metric/f1_macro".
    watch_metric: str


class TaskMetric(Module):
    """Stores all metrics for a task and provides unified update/compute interface.

    This is the base class for all task-specific metrics (e.g., ClassificationTaskMetric)
    and is the object returned from the training and validation steps for logging.
    """

    def __init__(self, task_metric_args: TaskMetricArgs) -> None:
        super().__init__()
        self.task_metric_args = task_metric_args

    def compute(self) -> MetricComputeResult:
        """Compute all metrics and return values for logging."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset all metrics."""
        for module in self.modules():
            if isinstance(module, Metric):
                module.reset()


def get_watch_metric_mode(
    task_metric_args: TaskMetricArgs,
    loss_names: list[str],
    watch_metric_name: str,
) -> Literal["min", "max"]:
    """Determine whether watch_metric_name should be maximised or minimised.

    Note 1: The matching is not exact key matching. Instead, we check if the
    watch_metric_name contains a known metric or loss name as a substring. The reason
    for this is because there are many possible metric name formats. For example,
    "val_metric/miou", "val_metric_classwise/iou_car", "val_metric_head/miou_lr0_01",
    or "val_metric_head_classwise/iou_car_lr0_01" could all be valid keys for a mIoU
    metric. As the watch mode doesn't depend on the exact naming format we keep it
    flexible.

    Note 2: Another reason for the flexible matching is that it is really hard to keep
    track of which metric names belong to which metric instances. The reason for this is
    that torchmetrics Metric instances can return many metric names from a single
    compute() call. Furthemore, wrappers like MetricCollection and ClasswiseWrapper
    do not keep track of which metric name came from which Metric instance.

    Args:
        task_metric_args: TaskMetricArgs instance to inspect.
        loss_names: Bare loss names tracked (e.g. ["loss", "loss_vfl"]).
        watch_metric_name: Full metric key to match
            (e.g. "val_metric/miou", "val_loss").

    Returns:
        "min" or "max".

    Raises:
        ValueError: If watch_metric_name couldn't be matched to a metric or loss name.

    """
    # Build bare-name -> mode from all MetricArgs fields
    name_to_mode: dict[str, Literal["min", "max"]] = {}
    for field_name in task_metric_args.__class__.model_fields:
        field_value = getattr(task_metric_args, field_name)
        if isinstance(field_value, MetricArgs):
            for name in field_value.get_metric_names():
                name_to_mode[name] = field_value.watch_mode

    # Match with metric names
    metric_parts = watch_metric_name.split("/")
    if len(metric_parts) > 1:
        # Metric suffix is usualy something like "top1_acc_micro" or for classwise
        # metrics it would be "iou_car".
        metric_suffix = metric_parts[1]
        for metric_name, mode in name_to_mode.items():
            if metric_suffix.startswith(metric_name):
                return mode

    # Match with loss names
    for loss_name in loss_names:
        if loss_name in watch_metric_name:
            return "min"

    raise ValueError(
        f"watch_metric_name {watch_metric_name!r} not found in metric names "
        f"{sorted(name_to_mode)} or loss names {loss_names}."
    )
