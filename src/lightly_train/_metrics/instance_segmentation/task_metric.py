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
from torch import Tensor
from torchmetrics import MetricCollection

from lightly_train._metrics.loss_metrics import LossMetrics
from lightly_train._metrics.mean_average_precision import (
    MeanAveragePrecisionArgs,
)
from lightly_train._metrics.metric_args import (
    translate_watch_metric,
)
from lightly_train._metrics.task_metric import (
    MetricComputeResult,
    TaskMetric,
    TaskMetricArgs,
)


class InstanceSegmentationTaskMetricArgs(TaskMetricArgs):
    loss_names: ClassVar[list[str]] = ["loss"]

    watch_metric: str = "val_metric/map"

    map: MeanAveragePrecisionArgs | None = Field(
        default_factory=MeanAveragePrecisionArgs
    )


class InstanceSegmentationTaskMetric(TaskMetric):
    """Container for all metrics for instance segmentation tasks."""

    def __init__(
        self,
        *,
        task_metric_args: InstanceSegmentationTaskMetricArgs,
        split: str,
        class_names: list[str],
        log_classwise: bool,
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
        self.class_names = class_names
        self.log_classwise = log_classwise
        self._best_metric_key = translate_watch_metric(
            task_metric_args.watch_metric, split
        )

        metrics = {}
        if task_metric_args.map is not None:
            metrics.update(
                task_metric_args.map.get_metrics(
                    classwise=log_classwise,
                    prefix=f"{split}_metric",
                    class_names=class_names,
                    iou_type="segm",
                    box_format="xyxy",
                )
            )
        self.metrics = MetricCollection(metrics)  # type: ignore
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

    def update_loss(self, loss_dict: dict[str, Tensor], weight: int) -> None:
        self.loss_metrics.update(loss_dict=loss_dict, weight=weight)

    def compute(self) -> MetricComputeResult:
        """Compute all metrics and return combined results."""
        result = self.loss_metrics.compute()
        result.update(self.metrics.compute())
        result = {name: float(value) for name, value in result.items()}
        best_val = result.get(self._best_metric_key)
        return MetricComputeResult(
            metrics=result,
            best_metric_key=self._best_metric_key if best_val is not None else None,
            best_metric_value=float(best_val) if best_val is not None else None,
            best_head_name=None,
            best_head_metrics=None,
        )
