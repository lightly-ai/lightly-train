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
from torchmetrics import Metric, MetricCollection

from lightly_train._metrics.detection.mean_average_precision_args import (
    MeanAveragePrecisionArgs,
)
from lightly_train._metrics.metric_args import MetricArgs
from lightly_train._metrics.task_metric import (
    MetricComputeResult,
    TaskMetric,
    TaskMetricArgs,
)

# Explicit mapping of base metric names to display name suffixes
BASE_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "map": "mAP@0.5:0.95",
    "map_50": "mAP@0.5",
    "map_75": "mAP@0.75",
    "map_small": "mAP (small)",
    "map_medium": "mAP (medium)",
    "map_large": "mAP (large)",
}


class ObjectDetectionTaskMetricArgs(TaskMetricArgs):
    """Metrics configuration for object detection tasks."""

    mean_average_precision: MeanAveragePrecisionArgs | None = Field(
        default_factory=MeanAveragePrecisionArgs
    )

    def get_metrics(  # type: ignore[override]
        self,
        *,
        prefix: str,
        class_names: list[str],
        log_classwise: bool,
        classwise_metric_args: ObjectDetectionTaskMetricArgs | None,
    ) -> ObjectDetectionTaskMetric:
        """Create ObjectDetectionTaskMetric instance for object detection.

        Args:
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            class_names: Class names for all metrics
            log_classwise: Whether to log classwise metrics
            classwise_metric_args: Optional separate args for classwise metrics
        """
        return ObjectDetectionTaskMetric(
            metric_args=self,
            prefix=prefix,
            class_names=class_names,
            log_classwise=log_classwise,
            classwise_metric_args=classwise_metric_args,
            best_metric_key=f"{prefix}map",
        )


class ObjectDetectionTaskMetric(TaskMetric):
    """Container for all metrics for object detection tasks.

    Inherits from TaskMetric which inherits from nn.Module.
    All metrics stored as attributes are automatically detected as child modules
    and handled by Lightning Fabric for device transfer.
    """

    def __init__(
        self,
        *,
        metric_args: ObjectDetectionTaskMetricArgs,
        prefix: str,
        class_names: list[str],
        log_classwise: bool,
        classwise_metric_args: ObjectDetectionTaskMetricArgs | None,
        best_metric_key: str,
    ) -> None:
        """Initialize object detection metrics container.

        Args:
            metric_args: Metrics configuration
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            class_names: Class names for all metrics
            log_classwise: Whether to log classwise metrics
            classwise_metric_args: Optional separate args for classwise metrics
            best_metric_key: Key of the metric used for model selection
        """
        super().__init__()

        self.metric_args = metric_args
        self.num_classes = len(class_names)
        self.prefix = prefix
        self.class_names = class_names
        self.log_classwise = log_classwise
        self._best_metric_key = best_metric_key

        # Build regular metrics
        metrics_dict = self._build_metrics(metric_args=metric_args, classwise=False)

        # Store metrics as ModuleDict for proper device handling
        self.metrics = torch.nn.ModuleDict(metrics_dict)  # type: ignore[arg-type]

        # Build classwise metrics
        self.metrics_classwise: torch.nn.ModuleDict | None = None  # type: ignore[assignment]
        if log_classwise:
            if classwise_metric_args is None:
                classwise_metric_args = metric_args.model_copy()
            metrics_classwise_dict = self._build_metrics(
                metric_args=classwise_metric_args, classwise=True
            )
            self.metrics_classwise = torch.nn.ModuleDict(metrics_classwise_dict)  # type: ignore[arg-type]

    def _build_metrics(
        self,
        metric_args: ObjectDetectionTaskMetricArgs,
        classwise: bool,
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

                metrics = individual_metric_args.get_metrics(
                    classwise=classwise,
                    num_classes=self.num_classes,
                )
                all_metrics.update(metrics)

        return all_metrics

    def update(
        self,
        preds: list[dict[str, Any]],
        target: list[dict[str, Any]],
    ) -> None:
        """Update all metrics with inputs.

        Args:
            preds: List of prediction dictionaries with keys "boxes", "scores", "labels"
            target: List of target dictionaries with keys "boxes", "labels"
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

    def compute(self) -> MetricComputeResult:
        """Compute all metrics and return combined results.

        Returns:
            MetricComputeResult with metrics dict, best_metric_key, and best_metric_value
        """
        result: dict[str, float] = {}

        # Compute regular metrics
        for key, metric in self.metrics.items():
            metric_result = metric.compute()  # type: ignore[operator]
            if isinstance(metric_result, dict):
                # MeanAveragePrecision returns a dict with multiple keys
                for sub_key, value in metric_result.items():
                    # Skip non-scalar metrics
                    if sub_key in ["map_per_class", "mar_100_per_class", "classes"]:
                        continue
                    result[f"{self.prefix}{sub_key}"] = float(value)
            else:
                result[f"{self.prefix}{key}"] = float(metric_result)

        # Compute classwise metrics
        if self.metrics_classwise is not None:
            prefix_without_slash = (
                self.prefix[:-1] if self.prefix.endswith("/") else self.prefix
            )
            classwise_prefix = f"{prefix_without_slash}_classwise/"

            for key, metric in self.metrics_classwise.items():
                metric_result = metric.compute()  # type: ignore[operator]
                if isinstance(metric_result, dict):
                    # MeanAveragePrecision returns a dict with per-class tensors
                    # Extract the classes tensor to map per-class values to class names
                    classes_tensor = metric_result.get("classes", None)

                    for sub_key, value in metric_result.items():
                        if sub_key.endswith("_per_class"):
                            # Expand per-class metrics into individual keys
                            # "map_per_class" -> "map_cat", "map_dog", etc.
                            base_key = sub_key[: -len("_per_class")]
                            # Check if value is a tensor and has the right dimensions
                            if (  # type: ignore[operator]
                                isinstance(value, torch.Tensor)
                                and classes_tensor is not None
                            ):
                                # Handle scalar tensors (single class case)
                                if value.ndim == 0:  # type: ignore[operator]
                                    # Single class - check if classes_tensor is also scalar
                                    if classes_tensor.ndim == 0:  # type: ignore[operator]
                                        class_idx = int(classes_tensor.item())  # type: ignore[operator]
                                        if class_idx < len(self.class_names):
                                            result[
                                                f"{classwise_prefix}{base_key}_{self.class_names[class_idx]}"
                                            ] = float(value)
                                    elif len(classes_tensor) > 0:  # type: ignore[operator]
                                        class_idx = int(classes_tensor[0].item())  # type: ignore[operator]
                                        if class_idx < len(self.class_names):
                                            result[
                                                f"{classwise_prefix}{base_key}_{self.class_names[class_idx]}"
                                            ] = float(value)
                                # Handle 1-d tensors (multiple classes)
                                elif value.ndim == 1:  # type: ignore[operator]
                                    # classes_tensor might be scalar if only one class
                                    if classes_tensor.ndim == 0:  # type: ignore[operator]
                                        class_idx = int(classes_tensor.item())  # type: ignore[operator]
                                        if (
                                            class_idx < len(self.class_names)
                                            and len(value) == 1
                                        ):
                                            result[
                                                f"{classwise_prefix}{base_key}_{self.class_names[class_idx]}"
                                            ] = float(value[0])
                                    elif len(value) == len(classes_tensor):  # type: ignore[operator]
                                        for i, class_idx_tensor in enumerate(  # type: ignore[operator]
                                            classes_tensor
                                        ):
                                            class_idx = int(class_idx_tensor.item())  # type: ignore[operator]
                                            if class_idx < len(self.class_names):
                                                result[
                                                    f"{classwise_prefix}{base_key}_{self.class_names[class_idx]}"
                                                ] = float(value[i])
                        elif sub_key not in ["classes"]:
                            # Regular scalar metrics (map, map_50, etc.)
                            result[f"{classwise_prefix}{sub_key}"] = float(value)
                else:
                    result[f"{classwise_prefix}{key}"] = float(metric_result)

        best_metric_value = float(result.get(self._best_metric_key, 0.0))
        return MetricComputeResult(
            metrics=result,
            best_metric_key=self._best_metric_key,
            best_metric_value=best_metric_value,
            best_head_name="",
            best_head_metrics=result,
        )

    def get_display_names(self) -> dict[str, str]:
        """Get display names for metrics"""
        display_names: dict[str, str] = {}

        # For regular metrics, get all possible keys from the metric
        # MeanAveragePrecision returns: map, map_50, map_75, map_small, map_medium, map_large
        for key in ["map", "map_50", "map_75", "map_small", "map_medium", "map_large"]:
            metric_name = f"{self.prefix}{key}"
            display_names[metric_name] = self._format_display_name(metric_name)

        # Classwise metrics
        if self.metrics_classwise is not None:
            prefix_without_slash = (
                self.prefix[:-1] if self.prefix.endswith("/") else self.prefix
            )
            classwise_prefix = f"{prefix_without_slash}_classwise/"

            # Base metrics (non-classwise)
            for key in [
                "map",
                "map_50",
                "map_75",
                "map_small",
                "map_medium",
                "map_large",
            ]:
                metric_name = f"{classwise_prefix}{key}"
                display_names[metric_name] = self._format_display_name(metric_name)

            # Per-class metrics
            for class_name in self.class_names:
                for key in ["map", "map_50", "map_75"]:
                    metric_name = f"{classwise_prefix}{key}_{class_name}"
                    display_names[metric_name] = self._format_display_name(metric_name)

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

        # Look up in explicit mapping
        if base_name in BASE_METRIC_DISPLAY_NAMES:
            return f"{split} {BASE_METRIC_DISPLAY_NAMES[base_name]}"

        # For classwise metrics that end with class name, strip it
        # e.g., "map_cat" -> look up "map"
        for base_key in BASE_METRIC_DISPLAY_NAMES:
            if base_name.startswith(f"{base_key}_"):
                return f"{split} {BASE_METRIC_DISPLAY_NAMES[base_key]}"

        # Fallback: capitalize and format with spaces
        return f"{split} {base_name.replace('_', ' ').title()}"


class _ClasswiseMetricCollection(MetricCollection):  # type: ignore[misc]
    """Renames classwise metric keys to handle class names with underscores.

    Replaces unique separator with underscore, avoiding conflicts when class names
    themselves contain underscores (e.g., "cat__type_a").
    """

    _SEPARATOR = "<SEP>"

    def compute(self) -> dict[str, Any]:  # type: ignore[override]
        """Compute metrics and rename keys by replacing separator with underscore."""
        result = super().compute()
        # ClasswiseWrapper joins metric_name with prefix as: metric_name + "_" + prefix + class_name
        # So with prefix="<SEP>" we get: metric_name_<SEP>class_name
        # Replace "_<SEP>" with "_" to get the desired format: metric_name_class_name
        return {
            key.replace(f"_{self._SEPARATOR}", "_"): value
            for key, value in result.items()
        }
