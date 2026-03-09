#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field
from torch import Tensor
from torchmetrics import MetricCollection as TorchmetricsMetricCollection

from lightly_train._metrics.loss_metric_collection import LossMetricCollection
from lightly_train._metrics.mean_average_precision import (
    MeanAveragePrecisionArgs,
)
from lightly_train._metrics.task_metric import (
    MetricComputeResult,
    TaskMetric,
    TaskMetricArgs,
    get_watch_metric_mode,
)


class InstanceSegmentationTaskMetricArgs(TaskMetricArgs):
    watch_metric: str = "val_metric/map"
    classwise: bool = False
    train: bool = False
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
        class_names: Sequence[str],
        loss_names: Sequence[str],
        init_metrics: bool | None = None,
    ) -> None:
        """Initialize instance segmentation metrics container.

        Args:
            metric_args: Metrics configuration
            split: Split name (e.g., "val", "train")
            class_names: Class names for all metrics
            loss_names: Names of losses to track
            init_metrics:
                Whether to initialize metrics. If None, uses task_metric_args.train
                for the train split and True for other splits.
        """
        super().__init__(task_metric_args=task_metric_args)
        self.split = split
        self.class_names = class_names
        self.watch_metric = task_metric_args.watch_metric
        self.watch_metric_mode = get_watch_metric_mode(
            task_metric_args, list(loss_names), task_metric_args.watch_metric
        )

        if init_metrics is None:
            init_metrics = task_metric_args.train if split == "train" else True

        metrics = {}
        if init_metrics and task_metric_args.map is not None:
            metrics.update(
                task_metric_args.map.get_metrics(
                    classwise=task_metric_args.classwise,
                    prefix=f"{split}_metric",
                    class_names=class_names,
                    iou_type="segm",
                    box_format="xyxy",
                )
            )
        self.metrics = TorchmetricsMetricCollection(metrics)  # type: ignore
        self.loss_metrics = LossMetricCollection(split=split, loss_names=loss_names)

    def update_with_predictions(
        self,
        preds: Sequence[Mapping[str, Any]],
        target: Sequence[Mapping[str, Any]],
    ) -> None:
        """Update all metrics with inputs.

        Args:
            preds: List of prediction dictionaries with keys "masks", "scores", "labels"
            target: List of target dictionaries with keys "masks", "labels"
        """
        self.metrics.update(preds, target)

    def update_loss(self, loss_dict: Mapping[str, Tensor], weight: int) -> None:
        self.loss_metrics.update(loss_dict=loss_dict, weight=weight)

    def compute(self) -> MetricComputeResult:
        """Compute all metrics and return combined results."""
        result = self.loss_metrics.compute()
        result.update(self.metrics.compute())
        result = {name: float(value) for name, value in result.items()}
        best_val = result.get(self.watch_metric)
        return MetricComputeResult(
            metrics=result,
            watch_metric=self.watch_metric if best_val is not None else None,
            watch_metric_value=float(best_val) if best_val is not None else None,
            watch_metric_mode=self.watch_metric_mode if best_val is not None else None,
            best_head_name=None,
            best_head_metrics=None,
        )
