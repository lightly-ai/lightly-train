#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from pydantic import Field
from torch import Tensor

from lightly_train._metrics.loss_metrics import LossMetrics
from lightly_train._metrics.metric_args import (
    translate_watch_metric,
)
from lightly_train._metrics.panoptic_segmentation.panoptic_quality_args import (
    PanopticQualityArgs,
)
from lightly_train._metrics.task_metric import (
    MetricComputeResult,
    TaskMetric,
    TaskMetricArgs,
)


class PanopticSegmentationTaskMetricArgs(TaskMetricArgs):
    loss_names: ClassVar[list[str]] = ["loss", "loss_vfl", "loss_bbox", "loss_giou"]

    watch_metric: str = "val_metric/pq"

    panoptic_quality: PanopticQualityArgs | None = Field(
        default_factory=PanopticQualityArgs
    )


class PanopticSegmentationTaskMetric(TaskMetric):
    """Container for all metrics for panoptic segmentation tasks.

    Inherits from TaskMetric which inherits from nn.Module.
    All metrics stored as attributes are automatically detected as child modules
    and handled by Lightning Fabric for device transfer.
    """

    def __init__(
        self,
        *,
        task_metric_args: PanopticSegmentationTaskMetricArgs,
        split: str,
        things: Sequence[int],
        stuffs: Sequence[int],
        class_names: Sequence[str],
        log_classwise: bool,
        classwise_metric_args: PanopticSegmentationTaskMetricArgs | None,
    ) -> None:
        """Initialize panoptic segmentation metrics container.

        Args:
            metric_args: Metrics configuration
            split: Split name (e.g., "val", "train")
            things: List of thing class IDs
            stuffs: List of stuff class IDs
            class_names: List of all class names (things + stuffs)
        """
        super().__init__(task_metric_args=task_metric_args)
        self.split = split
        self.prefix = f"{split}_metric/"
        self.things = things
        self.stuffs = stuffs
        self._best_metric_key = translate_watch_metric(
            task_metric_args.watch_metric, split
        )

        self.metrics = task_metric_args.build_metric_collection(
            prefix=f"{self.split}_metric/",
            things=things,
            stuffs=stuffs,
        )
        self.metrics_classwise = task_metric_args.build_classwise_metric_collection(
            log_classwise=log_classwise,
            prefix=f"{self.split}_metric_classwise/",
            classwise_metrics_args=classwise_metric_args,
            class_names=class_names,
            things=things,
            stuffs=stuffs,
        )
        self.loss_metrics = LossMetrics(
            split=split, loss_names=task_metric_args.loss_names
        )

    def update(
        self,
        preds: Tensor,
        target: Tensor,
    ) -> None:
        """Update all metrics with inputs.

        Args:
            preds: Prediction tensor of shape (B, H, W, 2) where last dim is (class_id, instance_id)
            target: Target tensor of shape (B, H, W, 2) where last dim is (class_id, instance_id)
        """
        self.metrics.update(preds, target)  # type: ignore[operator]

    def compute(self) -> MetricComputeResult:
        """Compute all metrics and return combined results."""
        result = self.loss_metrics.compute()

        # Compute metrics
        for key, metric in self.metrics.items():
            metric_result = metric.compute()  # type: ignore[operator]
            # PanopticQuality with return_sq_and_rq=True returns a tensor of shape (3,)
            # where the values are [PQ, SQ, RQ]
            if (  # type: ignore[operator]
                isinstance(metric_result, Tensor)
                and metric_result.ndim == 1  # type: ignore[operator]
                and len(metric_result) == 3  # type: ignore[operator]
            ):
                result[f"{self.prefix}pq"] = float(metric_result[0])  # type: ignore[operator]
                result[f"{self.prefix}sq"] = float(metric_result[1])  # type: ignore[operator]
                result[f"{self.prefix}rq"] = float(metric_result[2])  # type: ignore[operator]
            else:
                result[f"{self.prefix}{key}"] = float(metric_result)

        if self.metrics_classwise is not None:
            result.update(self.metrics_classwise.compute())

        best_value = result.get(self._best_metric_key)
        return MetricComputeResult(
            metrics=result,
            best_metric_key=self._best_metric_key if best_value is not None else None,
            best_metric_value=float(best_value) if best_value is not None else None,
            best_head_name=None,
            best_head_metrics=None,
        )
