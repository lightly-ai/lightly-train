#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Any

import torch
from pydantic import Field
from torch import Tensor
from torchmetrics import Metric

from lightly_train._metrics.metric_args import MetricArgs
from lightly_train._metrics.panoptic_segmentation.panoptic_quality_args import (
    PanopticQualityArgs,
)
from lightly_train._metrics.task_metric import TaskMetric, TaskMetricArgs

# Explicit mapping of base metric names to display name suffixes
BASE_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "pq": "PQ",
    "sq": "SQ",
    "rq": "RQ",
}


class PanopticSegmentationTaskMetricArgs(TaskMetricArgs):
    """Metrics configuration for panoptic segmentation tasks."""

    panoptic_quality: PanopticQualityArgs | None = Field(
        default_factory=PanopticQualityArgs
    )

    def get_metrics(  # type: ignore[override]
        self,
        *,
        prefix: str,
        things: list[int],
        stuffs: list[int],
    ) -> PanopticSegmentationTaskMetric:
        """Create PanopticSegmentationTaskMetric instance for panoptic segmentation.

        Args:
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            things: List of thing class IDs
            stuffs: List of stuff class IDs
        """
        return PanopticSegmentationTaskMetric(
            metric_args=self,
            prefix=prefix,
            things=things,
            stuffs=stuffs,
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
        metric_args: PanopticSegmentationTaskMetricArgs,
        prefix: str,
        things: list[int],
        stuffs: list[int],
    ) -> None:
        """Initialize panoptic segmentation metrics container.

        Args:
            metric_args: Metrics configuration
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            things: List of thing class IDs
            stuffs: List of stuff class IDs
        """
        super().__init__()

        self.metric_args = metric_args
        self.prefix = prefix
        self.things = things
        self.stuffs = stuffs

        # Build metrics
        metrics_dict = self._build_metrics(
            metric_args=metric_args,
            things=things,
            stuffs=stuffs,
        )

        # Store metrics as ModuleDict for proper device handling
        self.metrics = torch.nn.ModuleDict(metrics_dict)  # type: ignore[arg-type]

    def _build_metrics(
        self,
        metric_args: PanopticSegmentationTaskMetricArgs,
        things: list[int],
        stuffs: list[int],
    ) -> dict[str, Metric]:
        """Build metrics from args."""
        all_metrics: dict[str, Metric] = {}

        for field_name in metric_args.__class__.model_fields:
            individual_metric_args = getattr(metric_args, field_name)
            if not isinstance(individual_metric_args, MetricArgs):
                continue
            if individual_metric_args is not None:
                metrics = individual_metric_args.get_metrics(  # type: ignore[call-arg]
                    classwise=False,
                    things=things,
                    stuffs=stuffs,
                )
                all_metrics.update(metrics)

        return all_metrics

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
        for metric in self.metrics.values():
            metric.update(preds, target)  # type: ignore[operator]

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()  # type: ignore[operator]

    def compute(self) -> dict[str, Any]:
        """Compute all metrics and return combined results.

        Returns:
            Dictionary of all metric values with PQ, SQ, and RQ expanded
        """
        result: dict[str, Any] = {}

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
                result[f"{self.prefix}pq"] = metric_result[0]  # type: ignore[operator]
                result[f"{self.prefix}sq"] = metric_result[1]  # type: ignore[operator]
                result[f"{self.prefix}rq"] = metric_result[2]  # type: ignore[operator]
            else:
                result[f"{self.prefix}{key}"] = metric_result

        return result

    def get_display_names(self) -> dict[str, str]:
        """Get display names for metrics"""
        display_names: dict[str, str] = {}

        # PanopticQuality returns PQ, SQ, RQ
        for key in ["pq", "sq", "rq"]:
            metric_name = f"{self.prefix}{key}"
            display_names[metric_name] = self._format_display_name(metric_name)

        return display_names

    def _format_display_name(self, metric_name: str) -> str:
        """Format a metric name into a human-readable display name."""
        # Remove prefix to get base metric name
        if metric_name.startswith(self.prefix):
            base_name = metric_name[len(self.prefix) :]
        else:
            base_name = metric_name

        # Extract split name from prefix (e.g., "val" from "val_metric/")
        prefix_without_slash = (
            self.prefix[:-1] if self.prefix.endswith("/") else self.prefix
        )
        split = prefix_without_slash.split("_")[0].capitalize()

        # Look up in explicit mapping
        if base_name in BASE_METRIC_DISPLAY_NAMES:
            return f"{split} {BASE_METRIC_DISPLAY_NAMES[base_name]}"

        # Fallback: capitalize and format with spaces
        return f"{split} {base_name.replace('_', ' ').title()}"
