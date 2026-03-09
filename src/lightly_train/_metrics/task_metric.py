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
from torchmetrics import Metric as TorchmetricsMetric

from lightly_train._configs.config import PydanticConfig
from lightly_train._metrics.metric_args import MetricArgs

# ## Metrics Structure Overview
#
# The metrics system has three layers:
#
# 1. **MetricArgs** (Pydantic)
#   Stores configuration for one type of torchmetrics Metric. For example, AccuracyArgs
#   for torchmetrics.Accuracy, F1Args for torchmetrics.F1 etc. The sole purpose of
#   these classes is validation of user input through Pydantic and instantiation of
#   the actual torchmetrics Metric instances.
#
# 2. **TaskMetricArgs** (Pydantic)
#   Stores configuration for all metrics of a task. For example, ClassificationTaskMetricArgs
#   stores the MetricArgs for all classification metrics (e.g. AccuracyArgs, F1Args, etc.)
#   This is what the user interacts with when configuring `metric_args` in the train
#   commands.
#
# 3. **TaskMetric** (PyTorch Module)
#   Is initialized from TaskMetricArgs and stores all torchmetrics Metric instances
#   for a task and provides a unified interface for keeping track of metrics during
#   training and validation.
#
#   TaskMetric is updated with the latest predictions and losses in training_step and
#   validation_step of TrainModel. The TaskMetric instance is then returned from these
#   steps and the training loop calls compute_aggregated_values() to compute the aggregated
#   metric values over multiple training/validation steps for logging and best model
#   selection. Finally, the training loop calls reset() to reset the metrics after each
#   logging step.
#
#
# The metrics system follows the same structure as the transforms with the following
# equivalences:
# - MetricArgs      <-> TransformArgs
# - TaskMetricArgs  <-> TaskTransformArgs
# - TaskMetric      <-> TaskTransform
#
# Transforms have one additional layer which are <Model>TaskTransformArgs which store
# model specific configurations. This is not needed for metrics as they do not need
# model specific settings. For example, an ObjectDetectionTaskMetric works for all
# object detection models.
#
#
# ## Loss "Metrics"
#
# Losses are also tracked in the metrics system for the purpose of logging and
# best model selection. There is essentially no difference between a loss metric and a
# regular metric, both have a name and an associated float value. For this reason the
# TaskMetric classes also track the losses in addition to the regular metrics.
# This simplifies the interfaces and keeps all metric handling in one place.
#
# Note that the actual loss value for backpropagation is handled separately from the
# metrics system as it requires gradients. Everything in the metrics system is detached.
#


class TaskMetricArgs(PydanticConfig):
    """Base class for task-specific metrics configuration.

    Add any MetricArgs fields for the task here.
    """

    # Metric to watch for best model selection. E.g. "val_metric/f1_macro".
    watch_metric: str


class TaskMetric(Module):
    """Stores all torchmetric instances for a task and provides a unified interface for
    updating and computing metrics during training and validation.

    This is the base class for all task-specific metrics (e.g., ClassificationTaskMetric)
    and is the object returned from the training and validation steps.
    """

    def __init__(self, task_metric_args: TaskMetricArgs) -> None:
        super().__init__()
        self.task_metric_args = task_metric_args

    def compute_aggregated_values(self) -> AggregatedMetricValues:
        """Aggregate all metric values since the last .reset() and return them as an
        AggregatedMetricValues."""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets all metrics."""
        for module in self.modules():
            if isinstance(module, TorchmetricsMetric):
                module.reset()


@dataclass
class AggregatedMetricValues:
    """Result of computing all metrics.

    Attributes:
        metric_values:
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
        best_head_metric_values:
            Metric values of the best head with top-level keys
            (no head suffix). None for single-head metrics.
    """

    metric_values: dict[str, float]
    watch_metric: str | None
    watch_metric_value: float | None
    watch_metric_mode: Literal["min", "max"] | None
    best_head_name: str | None
    best_head_metric_values: dict[str, float] | None


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
