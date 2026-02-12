#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Any, Literal

from torchmetrics import ClasswiseWrapper, MeanMetric, Metric, MetricCollection

from lightly_train._metrics.classification.task_metric_args import (
    ClassificationTaskMetricArgs,
)
from lightly_train._metrics.task_metric import TaskMetric

# Explicit mapping of base metric names to display name suffixes
BASE_METRIC_DISPLAY_NAMES: dict[str, str] = {
    # Multiclass/Multilabel F1
    "f1_micro": "F1 (Micro)",
    "f1_macro": "F1 (Macro)",
    "f1_weighted": "F1 (Weighted)",
    # Multiclass/Multilabel Precision
    "precision_micro": "Precision (Micro)",
    "precision_macro": "Precision (Macro)",
    "precision_weighted": "Precision (Weighted)",
    # Multiclass/Multilabel Recall
    "recall_micro": "Recall (Micro)",
    "recall_macro": "Recall (Macro)",
    "recall_weighted": "Recall (Weighted)",
    # Multilabel Accuracy
    "accuracy_micro": "Accuracy (Micro)",
    "accuracy_macro": "Accuracy (Macro)",
    "accuracy_weighted": "Accuracy (Weighted)",
    # Multilabel AUROC
    "auroc_micro": "AUROC (Micro)",
    "auroc_macro": "AUROC (Macro)",
    "auroc_weighted": "AUROC (Weighted)",
    # Multilabel Average Precision
    "avg_precision_micro": "Avg Precision (Micro)",
    "avg_precision_macro": "Avg Precision (Macro)",
    "avg_precision_weighted": "Avg Precision (Weighted)",
    # Multilabel Hamming Distance
    "hamming_distance": "Hamming Distance",
}


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

        self.loss = MeanMetric()
        self.metrics = self._build_metric_collection(
            metric_args=metric_args,
            prefix=prefix,
            classwise=False,
        )
        self.metrics_classwise: MetricCollection | None = None
        if log_classwise:
            if classwise_metric_args is None:
                classwise_metric_args = metric_args.model_copy()

            if class_names is None:
                raise ValueError("class_names is required when log_classwise=True")

            self.metrics_classwise = self._build_classwise_metric_collection(
                metric_args=classwise_metric_args,
                prefix=f"{prefix}_classwise/",
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

        for field_name in metric_args.model_fields:
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
            name_str = str(name)
            display_name = self._format_display_name(name_str)
            display_names[name_str] = display_name

        # Classwise metrics
        if self.metrics_classwise is not None:
            for name in self.metrics_classwise.keys():
                name_str = str(name)
                display_name = self._format_display_name(name_str)
                display_names[name_str] = display_name

        return display_names

    def _format_display_name(self, metric_name: str) -> str:
        """Format a metric name into a human-readable display name."""
        # Remove prefix to get base metric name
        # Handle both regular and classwise prefixes
        classwise_prefix = f"{self.prefix}classwise/"
        if metric_name.startswith(classwise_prefix):
            base_name = metric_name[len(classwise_prefix) :]
        elif metric_name.startswith(self.prefix):
            base_name = metric_name[len(self.prefix) :]
        else:
            base_name = metric_name

        # Extract split name from prefix (e.g., "val" from "val_metric/")
        # Python 3.8 compatible: use string slicing instead of removeprefix
        prefix_without_slash = (
            self.prefix[:-1] if self.prefix.endswith("/") else self.prefix
        )
        split = prefix_without_slash.split("_")[0].capitalize()

        # Handle multiclass top-k accuracy (e.g., "top1_acc_micro" -> "Top-1 Acc (Micro)")
        if base_name.startswith("top") and "_acc_" in base_name:
            parts = base_name.split("_")
            if len(parts) >= 3:
                # Extract k from "topK"
                k = parts[0][3:]  # Remove "top" prefix
                avg = parts[2].capitalize()
                return f"{split} Top-{k} Acc ({avg})"

        # Look up in explicit mapping
        if base_name in BASE_METRIC_DISPLAY_NAMES:
            return f"{split} {BASE_METRIC_DISPLAY_NAMES[base_name]}"

        # Fallback: capitalize and format with spaces
        return f"{split} {base_name.replace('_', ' ').title()}"
