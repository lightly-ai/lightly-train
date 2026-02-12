#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from torchmetrics import Metric
from torchmetrics.detection import PanopticQuality  # type: ignore[attr-defined]

from lightly_train._metrics.metric_args import MetricArgs


class PanopticQualityArgs(MetricArgs):
    """Panoptic Quality metric for panoptic segmentation."""

    def get_metrics(  # type: ignore[override]
        self,
        *,
        classwise: bool = False,
        things: list[int],
        stuffs: list[int],
    ) -> dict[str, Metric]:
        """Create PanopticQuality metric instance for panoptic segmentation.

        Args:
            classwise: If True, compute per-class metrics (not used for PQ)
            things: List of thing class IDs
            stuffs: List of stuff class IDs

        Returns:
            Dictionary with single "pq" key containing the metric instance
        """
        metrics: dict[str, Metric] = {}

        metrics["pq"] = PanopticQuality(
            things=things,
            stuffs=stuffs,
            return_sq_and_rq=True,
            return_per_class=False,
        )

        return metrics

    def supports_classwise(self) -> bool:
        """PanopticQuality does not support classwise computation in our implementation."""
        return False
