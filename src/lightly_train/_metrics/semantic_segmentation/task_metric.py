#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import Field
from torch import Tensor
from torchmetrics import MetricCollection as TorchmetricsMetricCollection

from lightly_train._metrics.classwise_metric_collection import (
    ClasswiseMetricCollection,
)
from lightly_train._metrics.loss_metric_collection import LossMetricCollection
from lightly_train._metrics.semantic_segmentation.jaccard_index import (
    JaccardIndexArgs,
)
from lightly_train._metrics.task_metric import (
    MetricComputeResult,
    TaskMetric,
    TaskMetricArgs,
    get_watch_metric_mode,
)


class SemanticSegmentationTaskMetricArgs(TaskMetricArgs):
    watch_metric: str = "val_metric/miou"
    classwise: bool = False
    train: bool = False
    miou: JaccardIndexArgs | None = Field(default_factory=JaccardIndexArgs)


class SemanticSegmentationTaskMetric(TaskMetric):
    """Container for all metrics for semantic segmentation tasks."""

    def __init__(
        self,
        *,
        task_metric_args: SemanticSegmentationTaskMetricArgs,
        split: str,
        class_names: Sequence[str],
        ignore_index: int | None,
        loss_names: Sequence[str],
        init_metrics: bool | None = None,
    ) -> None:
        """Initialize semantic segmentation metrics container.

        Args:
            metric_args: Metrics configuration
            split: Split name (e.g., "val", "train")
            num_classes: Number of classes
            ignore_index: Class index to ignore in computation
            loss_names: Names of losses to track
            init_metrics:
                Whether to initialize metrics. If None, uses task_metric_args.train
                for the train split and True for other splits.
        """
        super().__init__(task_metric_args=task_metric_args)
        self.split = split
        self.ignore_index = ignore_index
        self.watch_metric = task_metric_args.watch_metric
        self.watch_metric_mode = get_watch_metric_mode(
            task_metric_args, list(loss_names), task_metric_args.watch_metric
        )

        if init_metrics is None:
            init_metrics = task_metric_args.train if split == "train" else True

        metrics = {}
        if init_metrics and task_metric_args.miou is not None:
            metrics.update(
                task_metric_args.miou.get_metrics(
                    classwise=False,
                    num_classes=len(class_names),
                    ignore_index=ignore_index,
                )
            )
        self.metrics = TorchmetricsMetricCollection(metrics, prefix=f"{split}_metric/")  # type: ignore

        metrics_classwise = {}
        if (
            init_metrics
            and task_metric_args.classwise
            and task_metric_args.miou is not None
        ):
            metrics_classwise = task_metric_args.miou.get_metrics(
                classwise=True,
                num_classes=len(class_names),
                ignore_index=ignore_index,
            )
        self.metrics_classwise = ClasswiseMetricCollection(
            metrics_classwise,
            class_names=class_names,
            prefix=f"{split}_metric_classwise/",
            classwise_prefix="iou",
        )
        self.loss_metrics = LossMetricCollection(split=split, loss_names=loss_names)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update all metrics

        Args:
            preds: Prediction tensor of shape (B, H, W) with class indices
            target: Target tensor of shape (B, H, W) with class indices
        """
        self.metrics.update(preds, target)  # type: ignore[operator]
        self.metrics_classwise.update(preds, target)  # type: ignore[operator]

    def update_loss(self, loss_dict: Mapping[str, Tensor], weight: int) -> None:
        self.loss_metrics.update(loss_dict=loss_dict, weight=weight)  # type: ignore[operator]

    def compute(self) -> MetricComputeResult:
        result = self.loss_metrics.compute()  # type: ignore[operator]
        result.update(self.metrics.compute())
        result.update(self.metrics_classwise.compute())
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
