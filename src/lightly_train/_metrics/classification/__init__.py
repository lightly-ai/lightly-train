# Copyright (c) 2025. Lightly AG and its affiliates.
# All Rights Reserved

"""Classification metrics."""

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
from lightly_train._metrics.classification.task_metric import ClassificationTaskMetric
from lightly_train._metrics.classification.task_metric_args import (
    ClassificationTaskMetricArgs,
)

__all__ = [
    "ClassificationTaskMetric",
    "ClassificationTaskMetricArgs",
    "MulticlassAccuracyArgs",
    "MulticlassF1Args",
    "MulticlassPrecisionArgs",
    "MulticlassRecallArgs",
    "MultilabelAccuracyArgs",
    "MultilabelAUROCArgs",
    "MultilabelAveragePrecisionArgs",
    "MultilabelF1Args",
    "MultilabelHammingDistanceArgs",
    "MultilabelPrecisionArgs",
    "MultilabelRecallArgs",
]
