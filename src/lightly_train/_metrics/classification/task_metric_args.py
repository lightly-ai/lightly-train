# Copyright (c) 2025. Lightly AG and its affiliates.
# All Rights Reserved

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lightly_train._metrics.base.task_metric_args import TaskMetricArgs
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

if TYPE_CHECKING:
    from lightly_train._metrics.classification.task_metric import (
        ClassificationTaskMetric,
    )


class ClassificationTaskMetricArgs(TaskMetricArgs):
    """Metrics configuration for classification tasks (multiclass or multilabel).

    Set only the fields appropriate for your task:
    - Multiclass: Use multiclass_* fields
    - Multilabel: Use multilabel_* fields
    """

    # Multiclass metrics
    multiclass_accuracy: MulticlassAccuracyArgs | None = None
    multiclass_f1: MulticlassF1Args | None = None
    multiclass_precision: MulticlassPrecisionArgs | None = None
    multiclass_recall: MulticlassRecallArgs | None = None

    # Multilabel metrics
    multilabel_accuracy: MultilabelAccuracyArgs | None = None
    multilabel_f1: MultilabelF1Args | None = None
    multilabel_precision: MultilabelPrecisionArgs | None = None
    multilabel_recall: MultilabelRecallArgs | None = None
    multilabel_auroc: MultilabelAUROCArgs | None = None
    multilabel_average_precision: MultilabelAveragePrecisionArgs | None = None
    multilabel_hamming_distance: MultilabelHammingDistanceArgs | None = None

    def get_metrics(
        self,
        *,
        num_classes: int,
        classification_task: Literal["multiclass", "multilabel"],
        prefix: str,
        class_names: list[str] | None = None,
        log_classwise: bool = False,
        classwise_metric_args: ClassificationTaskMetricArgs | None = None,
    ) -> ClassificationTaskMetric:
        """Create ClassificationTaskMetric instance.

        Args:
            num_classes: Number of classes
            classification_task: Type of classification
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
            classification_task=classification_task,
            prefix=prefix,
            class_names=class_names,
            log_classwise=log_classwise,
            classwise_metric_args=classwise_metric_args,
        )
