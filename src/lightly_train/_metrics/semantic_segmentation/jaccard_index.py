#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from torchmetrics import Metric as TorchmetricsMetric

from lightly_train._metrics.metric_args import MetricArgs


class JaccardIndexArgs(MetricArgs):
    """Jaccard Index (IoU/mIoU) metric for semantic segmentation."""

    def get_torchmetrics_instances(
        self,
        *,
        classwise: bool,
        num_classes: int,
        ignore_index: int | None,
    ) -> dict[str, TorchmetricsMetric]:
        """Create JaccardIndex metric instance for semantic segmentation.

        Args:
            classwise: If True, compute per-class IoU
            num_classes: Number of classes
            ignore_index: Class index to ignore in computation

        Returns:
            Dictionary with "miou" or "iou" key containing the metric instance
        """
        # Type ignore for old torchmetrics versions
        from torchmetrics.classification import (  # type: ignore[attr-defined]
            MulticlassJaccardIndex as TorchmetricsMulticlassJaccardIndex,
        )

        metrics: dict[str, TorchmetricsMetric] = {}

        if classwise:
            # Per-class IoU
            metrics["iou"] = TorchmetricsMulticlassJaccardIndex(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average="none",
                validate_args=False,
            )
        else:
            # Mean IoU
            metrics["miou"] = TorchmetricsMulticlassJaccardIndex(
                num_classes=num_classes,
                ignore_index=ignore_index,
                average="macro",
                validate_args=False,
            )

        return metrics

    def supports_classwise(self) -> bool:
        return True

    def get_metric_names(self) -> list[str]:
        return ["iou", "miou"]
