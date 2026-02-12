# Copyright (c) 2025. Lightly AG and its affiliates.
# All Rights Reserved

"""Metrics system for training tasks."""

from lightly_train._metrics.base import MetricArgs, TaskMetric, TaskMetricArgs
from lightly_train._metrics.classification import (
    ClassificationTaskMetric,
    ClassificationTaskMetricArgs,
    MulticlassAccuracyArgs,
    MulticlassF1Args,
    MulticlassPrecisionArgs,
    MulticlassRecallArgs,
    MultilabelAccuracyArgs,
    MultilabelAUROCArgs,
    MultilabelAveragePrecisionArgs,
    MultilabelF1Args,
    MultilabelHammingDistanceArgs,
    MultilabelPrecisionArgs,
    MultilabelRecallArgs,
)

__all__ = [
    # Base classes
    "MetricArgs",
    "TaskMetric",
    "TaskMetricArgs",
    # Classification
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
