#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from lightly_train._metrics.instance_segmentation.mean_average_precision_args import (
    InstanceSegmentationMeanAveragePrecisionArgs,
)
from lightly_train._metrics.loss_metrics import LossMetrics
from lightly_train._metrics.metric_args import (
    translate_watch_metric,
)
from lightly_train._metrics.task_metric import (
    MetricComputeResult,
    TaskMetric,
    TaskMetricArgs,
)

# Explicit mapping of base metric names to display name suffixes
BASE_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "map": "mAP@0.5:0.95",
    "map_50": "mAP@0.5",
    "map_75": "mAP@0.75",
    "map_small": "mAP (small)",
    "map_medium": "mAP (medium)",
    "map_large": "mAP (large)",
}


class InstanceSegmentationTaskMetricArgs(TaskMetricArgs):
    loss_names: ClassVar[list[str]] = ["loss", "loss_vfl", "loss_bbox", "loss_giou"]

    watch_metric: str = "val_metric/map"

    mean_average_precision: InstanceSegmentationMeanAveragePrecisionArgs | None = Field(
        default_factory=InstanceSegmentationMeanAveragePrecisionArgs
    )


class InstanceSegmentationTaskMetric(TaskMetric):
    """Container for all metrics for instance segmentation tasks.

    Inherits from TaskMetric which inherits from nn.Module.
    All metrics stored as attributes are automatically detected as child modules
    and handled by Lightning Fabric for device transfer.
    """

    def __init__(
        self,
        *,
        task_metric_args: InstanceSegmentationTaskMetricArgs,
        split: str,
        class_names: list[str],
        log_classwise: bool,
        classwise_metric_args: InstanceSegmentationTaskMetricArgs | None,
    ) -> None:
        """Initialize instance segmentation metrics container.

        Args:
            metric_args: Metrics configuration
            split: Split name (e.g., "val", "train")
            class_names: Class names for all metrics
            log_classwise: Whether to log classwise metrics
        """
        super().__init__(task_metric_args=task_metric_args)
        self.split = split
        self.prefix = f"{split}_metric/"
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.log_classwise = log_classwise
        self._best_metric_key = translate_watch_metric(
            task_metric_args.watch_metric, split
        )

        self.metrics = task_metric_args.build_metric_collection(
            prefix=f"{self.split}_metric/",
        )
        self.metrics_classwise = task_metric_args.build_classwise_metric_collection(
            log_classwise=log_classwise,
            prefix=f"{self.split}_metric_classwise/",
            classwise_metrics_args=classwise_metric_args,
            class_names=class_names,
        )
        self.loss_metrics = LossMetrics(
            split=split, loss_names=task_metric_args.loss_names
        )

    def update(
        self,
        preds: list[dict[str, Any]],
        target: list[dict[str, Any]],
    ) -> None:
        """Update all metrics with inputs.

        Args:
            preds: List of prediction dictionaries with keys "masks", "scores", "labels"
            target: List of target dictionaries with keys "masks", "labels"
        """
        self.metrics.update(preds, target)
        if self.metrics_classwise is not None:
            self.metrics_classwise.update(preds, target)

    def update_loss(self, loss_dict: dict[str, float], weight: int) -> None:
        self.loss_metrics.update(loss_dict=loss_dict, weight=weight)

    def compute(self) -> MetricComputeResult:
        """Compute all metrics and return combined results."""
        result: dict[str, float] = self.loss_metrics.compute()
        result.update(self.metrics.compute())
        if self.metrics_classwise is not None:
            result.update(self.metrics_classwise.compute())
        best_val = result.get(self._best_metric_key)
        return MetricComputeResult(
            metrics=result,
            best_metric_key=self._best_metric_key if best_val is not None else None,
            best_metric_value=float(best_val) if best_val is not None else None,
            best_head_name=None,
            best_head_metrics=None,
        )
