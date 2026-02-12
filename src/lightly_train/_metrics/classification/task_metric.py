# Copyright (c) 2025. Lightly AG and its affiliates.
# All Rights Reserved

from __future__ import annotations

from typing import Any, Literal

from torchmetrics import ClasswiseWrapper, MeanMetric, Metric, MetricCollection

from lightly_train._metrics.base.task_metric import TaskMetric
from lightly_train._metrics.classification.task_metric_args import (
    ClassificationTaskMetricArgs,
)


class ClassificationTaskMetric(TaskMetric):
    """Container for all metrics for classification tasks.

    Inherits from TaskMetric which inherits from nn.Module.
    All metrics stored as attributes are automatically detected as child modules
    and handled by Lightning Fabric for device transfer.
    """

    def __init__(
        self,
        *,
        metric_args: ClassificationTaskMetricArgs,
        num_classes: int,
        classification_task: Literal["multiclass", "multilabel"],
        prefix: str,
        class_names: list[str] | None = None,
        log_classwise: bool = False,
        classwise_metric_args: ClassificationTaskMetricArgs | None = None,
    ) -> None:
        """Initialize classification metrics container.

        Args:
            metric_args: Metrics configuration
            num_classes: Number of classes
            classification_task: Type of classification task
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            class_names: Optional class names for classwise metrics
            log_classwise: Whether to log classwise metrics
            classwise_metric_args: Optional separate args for classwise metrics
        """
        # CRITICAL: Call parent __init__ to properly initialize nn.Module
        super().__init__()

        self.metric_args = metric_args
        self.num_classes = num_classes
        self.classification_task = classification_task
        self.prefix = prefix
        self.class_names = class_names
        self.log_classwise = log_classwise

        # ✅ Stored as attributes - automatically detected as child modules
        self.loss = MeanMetric()

        # ✅ MetricCollection is also nn.Module, properly detected
        self.metrics = self._build_metric_collection(
            metric_args=metric_args,
            prefix=prefix,
            classwise=False,
        )

        # ✅ Optional classwise metrics also properly detected
        self.metrics_classwise: MetricCollection | None = None
        if log_classwise:
            if classwise_metric_args is None:
                classwise_metric_args = self._auto_classwise_args(metric_args)

            if class_names is None:
                raise ValueError("class_names is required when log_classwise=True")

            self.metrics_classwise = self._build_classwise_metric_collection(
                metric_args=classwise_metric_args,
                prefix=f"{prefix}classwise/",
                class_names=class_names,
            )

    def _build_metric_collection(
        self,
        metric_args: ClassificationTaskMetricArgs,
        prefix: str,
        classwise: bool,
    ) -> MetricCollection:
        """Build a MetricCollection from args."""
        all_metrics: dict[str, Metric] = {}

        # Get relevant field prefix based on task
        field_prefix = (
            "multiclass_" if self.classification_task == "multiclass" else "multilabel_"
        )

        for field_name in metric_args.model_fields:
            # Skip metrics not for this task
            if not field_name.startswith(field_prefix):
                continue

            individual_metric_args = getattr(metric_args, field_name)
            if individual_metric_args is not None:
                if classwise and not individual_metric_args.supports_classwise():
                    continue

                metrics = individual_metric_args.get_metrics(
                    classwise=classwise,
                    num_classes=self.num_classes,
                )
                all_metrics.update(metrics)

        return MetricCollection(all_metrics, prefix=prefix)  # type: ignore[arg-type]

    def _build_classwise_metric_collection(
        self,
        metric_args: ClassificationTaskMetricArgs,
        prefix: str,
        class_names: list[str],
    ) -> MetricCollection:
        """Build a classwise MetricCollection."""
        base_metrics = self._build_metric_collection(
            metric_args=metric_args,
            prefix="",
            classwise=True,
        )

        classwise_metrics: dict[str, Metric] = {}
        for key, base_metric in base_metrics.items():
            classwise_metrics[key] = ClasswiseWrapper(  # type: ignore[call-arg]
                base_metric,
                prefix="_",
                labels=class_names,
            )

        return MetricCollection(classwise_metrics, prefix=prefix)  # type: ignore[arg-type]

    def _auto_classwise_args(
        self, metric_args: ClassificationTaskMetricArgs
    ) -> ClassificationTaskMetricArgs:
        """Auto-generate classwise args by filtering unsupported metrics."""
        if self.classification_task == "multiclass":
            # topk>1 accuracy doesn't support classwise
            return ClassificationTaskMetricArgs(
                multiclass_accuracy=None,  # Filtered - topk>1 doesn't work
                multiclass_f1=metric_args.multiclass_f1,
                multiclass_precision=metric_args.multiclass_precision,
                multiclass_recall=metric_args.multiclass_recall,
            )
        elif self.classification_task == "multilabel":
            # All multilabel metrics support classwise
            return metric_args.model_copy()
        else:
            raise ValueError(
                f"Unsupported classification task: {self.classification_task}"
            )

    def items(self) -> dict[str, Any]:
        """Get all metric instances for adding to log_dict."""
        result = dict(self.metrics.items())
        if self.metrics_classwise is not None:
            result.update(dict(self.metrics_classwise.items()))
        return result

    def get_display_names(self) -> dict[str, str]:
        """Get display names for metrics (for logging)."""
        display_names: dict[str, str] = {}

        # Standard metrics
        for name in self.metrics.keys():
            display_name = self._format_display_name(name)
            display_names[name] = display_name

        # Classwise metrics
        if self.metrics_classwise is not None:
            for name in self.metrics_classwise.keys():
                display_name = self._format_display_name(name)
                display_names[name] = display_name

        return display_names

    def _format_display_name(self, metric_name: str) -> str:
        """Format a metric name into a human-readable display name.

        Removes prefix and formats the rest in title case with proper formatting.
        """
        # Remove the prefix (e.g., "val_metric/" or "train_metric/")
        name = metric_name.replace(self.prefix, "")
        name = name.replace(f"{self.prefix}classwise/", "")

        # Extract the split name from prefix (e.g., "val" from "val_metric/")
        split = self.prefix.rstrip("/").split("_")[0].capitalize()

        # Handle special cases
        if "top" in name:
            # top1_acc_micro -> Top-1 Acc (Micro)
            parts = name.split("_")
            if len(parts) >= 2:
                topk = parts[0].replace("top", "Top-")
                metric = parts[1].upper() if len(parts) > 1 else ""
                avg = parts[2].capitalize() if len(parts) > 2 else ""
                result = f"{split} {topk} {metric}"
                if avg:
                    result += f" ({avg})"
                return result

        # General case: accuracy_micro -> Accuracy (Micro)
        parts = name.split("_")
        metric = parts[0].capitalize()
        avg = parts[1].capitalize() if len(parts) > 1 else ""
        if avg:
            return f"{split} {metric} ({avg})"
        return f"{split} {metric}"
