#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from torchmetrics import Metric
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from lightly_train._metrics.metric_args import MetricArgs


class InstanceSegmentationMeanAveragePrecisionArgs(MetricArgs):
    """Mean Average Precision metric for instance segmentation."""

    def get_metrics(
        self,
        *,
        classwise: bool = False,
        num_classes: int,
    ) -> dict[str, Metric]:
        """Create MeanAveragePrecision metric instance for instance segmentation.

        Args:
            classwise: If True, compute per-class metrics
            num_classes: Number of classes (unused for detection metrics)

        Returns:
            Dictionary with single "map" key containing the metric instance
        """
        metrics: dict[str, Metric] = {}

        map_metric = MeanAveragePrecision(
            iou_type="segm",  # type: ignore[arg-type]
            class_metrics=classwise,
        )
        map_metric.warn_on_many_detections = False  # type: ignore[attr-defined]
        metrics["map"] = map_metric

        return metrics

    def supports_classwise(self) -> bool:
        """MeanAveragePrecision supports classwise computation."""
        return True
