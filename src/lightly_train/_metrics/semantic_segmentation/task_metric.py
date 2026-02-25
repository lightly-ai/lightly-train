#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Union

from pydantic import Field
from torch import Tensor

from lightly_train._metrics.loss_metrics import LossMetrics
from lightly_train._metrics.metric_args import (
    translate_watch_metric,
)
from lightly_train._metrics.semantic_segmentation.jaccard_index_args import (
    JaccardIndexArgs,
)
from lightly_train._metrics.task_metric import (
    MetricComputeResult,
    TaskMetric,
    TaskMetricArgs,
)

# Explicit mapping of base metric names to display name suffixes
BASE_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "miou": "mIoU",
    "iou": "IoU",
}


class SemanticSegmentationTaskMetricArgs(TaskMetricArgs):
    loss_names: ClassVar[list[str]] = ["loss"]

    watch_metric: str = "val_metric/miou"

    jaccard_index: JaccardIndexArgs | None = Field(default_factory=JaccardIndexArgs)


class SemanticSegmentationTaskMetric(TaskMetric):
    """Container for all metrics for semantic segmentation tasks.

    Inherits from TaskMetric which inherits from nn.Module.
    All metrics stored as attributes are automatically detected as child modules
    and handled by Lightning Fabric for device transfer.
    """

    def __init__(
        self,
        *,
        task_metric_args: SemanticSegmentationTaskMetricArgs,
        split: str,
        num_classes: int,
        class_names: Sequence[str],
        ignore_index: int | None,
        log_classwise: bool,
        classwise_metric_args: SemanticSegmentationTaskMetricArgs | None,
    ) -> None:
        """Initialize semantic segmentation metrics container.

        Args:
            metric_args: Metrics configuration
            split: Split name (e.g., "val", "train")
            num_classes: Number of classes
            ignore_index: Class index to ignore in computation
            log_classwise: Whether to log classwise metrics
        """
        super().__init__(task_metric_args=task_metric_args)
        self.split = split
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.log_classwise = log_classwise
        self._best_metric_key = translate_watch_metric(
            task_metric_args.watch_metric, split
        )

        self.metrics = task_metric_args.build_metric_collection(
            prefix=f"{self.split}_metric/",
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.metrics_classwise = task_metric_args.build_classwise_metric_collection(
            log_classwise=log_classwise,
            prefix=f"{self.split}_metric_classwise/",
            classwise_metrics_args=classwise_metric_args,
            class_names=class_names,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        self.loss_metrics = LossMetrics(
            split=split, loss_names=task_metric_args.loss_names
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update all metrics

        Args:
            preds: Prediction tensor of shape (B, H, W) with class indices
            target: Target tensor of shape (B, H, W) with class indices
        """
        self.metrics.update(preds, target)  # type: ignore[operator]
        if self.metrics_classwise is not None:
            self.metrics_classwise.update(preds, target)  # type: ignore[operator]

    def update_loss(
        self,
        loss_dict: dict[str, Union[float, Tensor]],
        weight: int,
    ) -> None:
        """Accumulate loss values.

        Args:
            loss_dict: Mapping from loss name (e.g., "loss", "loss_vfl") to value.
                       Only names present in metric_args.loss_names are tracked.
            weight: Sample weight for accumulation (typically batch size).
        """
        self.loss_metrics.update_loss(loss_dict=loss_dict, weight=weight)  # type: ignore[operator]

    def compute(self) -> MetricComputeResult:
        """Compute all metrics and return combined results."""
        result = self.loss_metrics.compute()  # type: ignore[operator]
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
