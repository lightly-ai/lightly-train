#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Any

from pydantic import Field
from torch import Tensor
from torchmetrics import ClasswiseWrapper, Metric, MetricCollection

from lightly_train._metrics.classification.multiclass_metric_args import (
    MulticlassAccuracyArgs,
    MulticlassF1Args,
    MulticlassPrecisionArgs,
    MulticlassRecallArgs,
)
from lightly_train._metrics.classification.multilabel_metric_args import (
    MultilabelAccuracyArgs,
    MultilabelAUROCArgs,
    MultilabelAveragePrecisionArgs,
    MultilabelF1Args,
    MultilabelHammingDistanceArgs,
    MultilabelPrecisionArgs,
    MultilabelRecallArgs,
)
from lightly_train._metrics.metric_args import MetricArgs
from lightly_train._metrics.task_metric import TaskMetric, TaskMetricArgs

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


class MulticlassClassificationTaskMetricArgs(TaskMetricArgs):
    """Metrics configuration for multiclass classification tasks."""

    accuracy: MulticlassAccuracyArgs | None = Field(
        default_factory=MulticlassAccuracyArgs
    )
    f1: MulticlassF1Args | None = Field(default_factory=MulticlassF1Args)
    precision: MulticlassPrecisionArgs | None = Field(
        default_factory=MulticlassPrecisionArgs
    )
    recall: MulticlassRecallArgs | None = Field(default_factory=MulticlassRecallArgs)

    def get_metrics(  # type: ignore[override]
        self,
        *,
        prefix: str,
        class_names: list[str],
        log_classwise: bool,
        classwise_metric_args: MulticlassClassificationTaskMetricArgs | None,
    ) -> ClassificationTaskMetric:
        """Create ClassificationTaskMetric instance for multiclass classification.

        Args:
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            class_names: Class names for all metrics
            log_classwise: Whether to log classwise metrics
            classwise_metric_args: Optional separate args for classwise metrics
        """
        return ClassificationTaskMetric(
            metric_args=self,
            prefix=prefix,
            class_names=class_names,
            log_classwise=log_classwise,
            classwise_metric_args=classwise_metric_args,
        )


class MultilabelClassificationTaskMetricArgs(TaskMetricArgs):
    """Metrics configuration for multilabel classification tasks."""

    accuracy: MultilabelAccuracyArgs | None = Field(
        default_factory=MultilabelAccuracyArgs
    )
    f1: MultilabelF1Args | None = Field(default_factory=MultilabelF1Args)
    precision: MultilabelPrecisionArgs | None = Field(default=None)
    recall: MultilabelRecallArgs | None = Field(default=None)
    auroc: MultilabelAUROCArgs | None = Field(default_factory=MultilabelAUROCArgs)
    average_precision: MultilabelAveragePrecisionArgs | None = Field(
        default_factory=MultilabelAveragePrecisionArgs
    )
    hamming_distance: MultilabelHammingDistanceArgs | None = Field(
        default_factory=MultilabelHammingDistanceArgs
    )

    def get_metrics(  # type: ignore[override]
        self,
        *,
        prefix: str,
        class_names: list[str],
        log_classwise: bool,
        classwise_metric_args: MultilabelClassificationTaskMetricArgs | None,
    ) -> ClassificationTaskMetric:
        """Create ClassificationTaskMetric instance for multilabel classification.

        Args:
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            class_names: Class names for all metrics
            log_classwise: Whether to log classwise metrics
            classwise_metric_args: Optional separate args for classwise metrics
        """
        return ClassificationTaskMetric(
            metric_args=self,
            prefix=prefix,
            class_names=class_names,
            log_classwise=log_classwise,
            classwise_metric_args=classwise_metric_args,
        )


ClassificationTaskMetricArgs = (
    MulticlassClassificationTaskMetricArgs | MultilabelClassificationTaskMetricArgs
)


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
        prefix: str,
        class_names: list[str],
        log_classwise: bool,
        classwise_metric_args: ClassificationTaskMetricArgs | None,
    ) -> None:
        """Initialize classification metrics container.

        Args:
            metric_args: Metrics configuration
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            class_names: Class names for all metrics
            log_classwise: Whether to log classwise metrics
            classwise_metric_args: Optional separate args for classwise metrics
        """
        super().__init__()

        self.metric_args = metric_args
        self.num_classes = len(class_names)
        self.prefix = prefix
        self.class_names = class_names
        self.log_classwise = log_classwise

        self.metrics = self._build_metric_collection(
            metric_args=metric_args,
            prefix=prefix,
            classwise=False,
        )
        self.metrics_classwise: MetricCollection | None = None
        if log_classwise:
            if classwise_metric_args is None:
                classwise_metric_args = metric_args.model_copy()

            # Remove trailing slash from prefix for Python 3.8 compatibility
            # (avoiding str.removesuffix which is Python 3.9+)
            prefix_without_slash = prefix[:-1] if prefix.endswith("/") else prefix
            self.metrics_classwise = self._build_classwise_metric_collection(
                metric_args=classwise_metric_args,
                prefix=f"{prefix_without_slash}_classwise/",
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
            # Use unique separator - _ClasswiseMetricCollection will replace it with "_"
            classwise_metrics[key] = ClasswiseWrapper(  # type: ignore[call-arg]
                base_metric,
                prefix=_ClasswiseMetricCollection._SEPARATOR,
                labels=class_names,
            )

        # Use custom MetricCollection subclass that handles key renaming
        return _ClasswiseMetricCollection(classwise_metrics, prefix=prefix)  # type: ignore[arg-type]

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update all metrics with inputs.

        Args:
            preds: Predictions tensor
            target: Target tensor
        """
        self.metrics.update(preds, target)
        if self.metrics_classwise is not None:
            self.metrics_classwise.update(preds, target)

    def compute(self) -> dict[str, Any]:
        """Compute all metrics and return combined results.

        Returns:
            Combined dictionary of all metric values from both regular and classwise metrics
        """
        result = self.metrics.compute()
        if self.metrics_classwise is not None:
            result.update(self.metrics_classwise.compute())
        return result

    def get_display_names(self) -> dict[str, str]:
        """Get display names for metrics"""
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
        # Python 3.8 compatible: use string slicing instead of removeprefix
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
