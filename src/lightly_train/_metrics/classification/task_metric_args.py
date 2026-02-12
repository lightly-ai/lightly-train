#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field

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
from lightly_train._metrics.task_metric_args import TaskMetricArgs

if TYPE_CHECKING:
    from lightly_train._metrics.classification.task_metric import (
        ClassificationTaskMetric,
    )


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
        num_classes: int,
        prefix: str,
        class_names: list[str] | None = None,
        log_classwise: bool = False,
        classwise_metric_args: MulticlassClassificationTaskMetricArgs | None = None,
    ) -> ClassificationTaskMetric:
        """Create ClassificationTaskMetric instance for multiclass classification.

        Args:
            num_classes: Number of classes
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            class_names: Optional class names for classwise metrics
            log_classwise: Whether to log classwise metrics
            classwise_metric_args: Optional separate args for classwise metrics

        Returns:
            ClassificationTaskMetric instance.
        """
        from lightly_train._metrics.classification.task_metric import (
            ClassificationTaskMetric,
        )

        return ClassificationTaskMetric(
            metric_args=self,
            num_classes=num_classes,
            classification_task="multiclass",
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
        num_classes: int,
        prefix: str,
        class_names: list[str] | None = None,
        log_classwise: bool = False,
        classwise_metric_args: MultilabelClassificationTaskMetricArgs | None = None,
    ) -> ClassificationTaskMetric:
        """Create ClassificationTaskMetric instance for multilabel classification.

        Args:
            num_classes: Number of classes
            prefix: Prefix for metric names (e.g., "val_metric/", "train_metric/")
            class_names: Optional class names for classwise metrics
            log_classwise: Whether to log classwise metrics
            classwise_metric_args: Optional separate args for classwise metrics

        Returns:
            ClassificationTaskMetric instance.
        """
        from lightly_train._metrics.classification.task_metric import (
            ClassificationTaskMetric,
        )

        return ClassificationTaskMetric(
            metric_args=self,
            num_classes=num_classes,
            classification_task="multilabel",
            prefix=prefix,
            class_names=class_names,
            log_classwise=log_classwise,
            classwise_metric_args=classwise_metric_args,
        )


ClassificationTaskMetricArgs = (
    MulticlassClassificationTaskMetricArgs | MultilabelClassificationTaskMetricArgs
)
