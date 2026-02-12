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
from lightly_train._metrics.semantic_segmentation.jaccard_index_args import (
    JaccardIndexArgs,
)
from lightly_train._metrics.task_metric import TaskMetric, TaskMetricArgs

# Explicit mapping of base metric names to display name suffixes
BASE_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "miou": "mIoU",
    "iou": "IoU",
}


class SemanticSegmentationTaskMetricArgs(TaskMetricArgs):
    """Metrics configuration for semantic segmentation tasks."""

    jaccard_index: JaccardIndexArgs | None = Field(default_factory=JaccardIndexArgs)

    def get_metrics(  # type: ignore[override]
        self,
        *,
        prefix: str,
        num_classes: int,
        ignore_index: int | None = None,
        log_classwise: bool,
    ) -> SemanticSegmentationTaskMetric:
        """Create SemanticSegmentationTaskMetric instance for semantic segmentation.

        Args:
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            num_classes: Number of classes
            ignore_index: Class index to ignore in computation
            log_classwise: Whether to log classwise metrics
        """
        return SemanticSegmentationTaskMetric(
            metric_args=self,
            prefix=prefix,
            num_classes=num_classes,
            ignore_index=ignore_index,
            log_classwise=log_classwise,
        )


class SemanticSegmentationTaskMetric(TaskMetric):
    """Container for all metrics for semantic segmentation tasks.

    Inherits from TaskMetric which inherits from nn.Module.
    All metrics stored as attributes are automatically detected as child modules
    and handled by Lightning Fabric for device transfer.
    """

    def __init__(
        self,
        *,
        metric_args: SemanticSegmentationTaskMetricArgs,
        prefix: str,
        num_classes: int,
        ignore_index: int | None,
        log_classwise: bool,
    ) -> None:
        """Initialize semantic segmentation metrics container.

        Args:
            metric_args: Metrics configuration
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            num_classes: Number of classes
            ignore_index: Class index to ignore in computation
            log_classwise: Whether to log classwise metrics
        """
        super().__init__()

        self.metric_args = metric_args
        self.num_classes = num_classes
        self.prefix = prefix
        self.ignore_index = ignore_index
        self.log_classwise = log_classwise

        # Build regular metrics
        metrics_dict = self._build_metrics(
            metric_args=metric_args,
            classwise=False,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        # Store metrics as ModuleDict for proper device handling
        self.metrics = torch.nn.ModuleDict(metrics_dict)  # type: ignore[arg-type]

        # Build classwise metrics
        self.metrics_classwise: torch.nn.ModuleDict | None = None  # type: ignore[assignment]
        if log_classwise:
            metrics_classwise_dict = self._build_metrics(
                metric_args=metric_args,
                classwise=True,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            self.metrics_classwise = torch.nn.ModuleDict(metrics_classwise_dict)  # type: ignore[arg-type]

    def _build_metrics(
        self,
        metric_args: SemanticSegmentationTaskMetricArgs,
        classwise: bool,
        num_classes: int,
        ignore_index: int | None,
    ) -> dict[str, Metric]:
        """Build metrics from args."""
        all_metrics: dict[str, Metric] = {}

        for field_name in metric_args.__class__.model_fields:
            individual_metric_args = getattr(metric_args, field_name)
            if not isinstance(individual_metric_args, MetricArgs):
                continue
            if individual_metric_args is not None:
                if classwise and not individual_metric_args.supports_classwise():
                    continue

                metrics = individual_metric_args.get_metrics(  # type: ignore[call-arg]
                    classwise=classwise,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
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
            preds: Prediction tensor of shape (B, H, W) with class indices
            target: Target tensor of shape (B, H, W) with class indices
        """
        for metric in self.metrics.values():
            metric.update(preds, target)  # type: ignore[operator]
        if self.metrics_classwise is not None:
            for metric in self.metrics_classwise.values():
                metric.update(preds, target)  # type: ignore[operator]

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()  # type: ignore[operator]
        if self.metrics_classwise is not None:
            for metric in self.metrics_classwise.values():
                metric.reset()  # type: ignore[operator]

    def compute(self) -> dict[str, Any]:
        """Compute all metrics and return combined results.

        Returns:
            Combined dictionary of all metric values from both regular and classwise metrics
        """
        result: dict[str, Any] = {}

        # Compute regular metrics
        for key, metric in self.metrics.items():
            metric_result = metric.compute()  # type: ignore[operator]
            result[f"{self.prefix}{key}"] = metric_result

        # Compute classwise metrics
        if self.metrics_classwise is not None:
            prefix_without_slash = (
                self.prefix[:-1] if self.prefix.endswith("/") else self.prefix
            )
            classwise_prefix = f"{prefix_without_slash}_classwise/"

            for key, metric in self.metrics_classwise.items():
                metric_result = metric.compute()  # type: ignore[operator]
                # metric_result is a tensor of shape (num_classes,) with per-class IoU
                if isinstance(metric_result, Tensor) and metric_result.ndim == 1:  # type: ignore[operator]
                    for class_idx in range(len(metric_result)):  # type: ignore[operator]
                        result[f"{classwise_prefix}{key}_{class_idx}"] = metric_result[  # type: ignore[operator]
                            class_idx
                        ]
                else:
                    result[f"{classwise_prefix}{key}"] = metric_result

        return result

    def get_display_names(self) -> dict[str, str]:
        """Get display names for metrics"""
        display_names: dict[str, str] = {}

        # Regular metrics
        for key in self.metrics.keys():
            metric_name = f"{self.prefix}{key}"
            display_names[metric_name] = self._format_display_name(metric_name)

        # Classwise metrics
        if self.metrics_classwise is not None:
            prefix_without_slash = (
                self.prefix[:-1] if self.prefix.endswith("/") else self.prefix
            )
            classwise_prefix = f"{prefix_without_slash}_classwise/"

            # Per-class metrics
            for key in self.metrics_classwise.keys():
                for class_idx in range(self.num_classes):  # type: ignore[operator]
                    metric_name = f"{classwise_prefix}{key}_{class_idx}"  # type: ignore[operator]
                    display_names[metric_name] = self._format_display_name(metric_name)  # type: ignore[operator]

        return display_names

    def _format_display_name(self, metric_name: str) -> str:
        """Format a metric name into a human-readable display name."""
        # Remove prefix to get base metric name
        prefix_without_slash = (
            self.prefix[:-1] if self.prefix.endswith("/") else self.prefix
        )
        classwise_prefix = f"{prefix_without_slash}_classwise/"
        if metric_name.startswith(classwise_prefix):
            base_name = metric_name[len(classwise_prefix) :]
        elif metric_name.startswith(self.prefix):
            base_name = metric_name[len(self.prefix) :]
        else:
            base_name = metric_name

        # Extract split name from prefix (e.g., "val" from "val_metric/")
        split = prefix_without_slash.split("_")[0].capitalize()

        # For classwise metrics that end with class index, strip it
        # e.g., "iou_0" -> look up "iou"
        for base_key in BASE_METRIC_DISPLAY_NAMES:
            if base_name == base_key:
                return f"{split} {BASE_METRIC_DISPLAY_NAMES[base_key]}"
            if base_name.startswith(f"{base_key}_"):
                # Has class suffix, still use base display name
                return f"{split} {BASE_METRIC_DISPLAY_NAMES[base_key]}"

        # Fallback: capitalize and format with spaces
        return f"{split} {base_name.replace('_', ' ').title()}"
