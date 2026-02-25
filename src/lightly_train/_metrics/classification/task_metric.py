#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from pydantic import Field
from torch import Tensor

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
from lightly_train._metrics.loss_metrics import LossMetrics
from lightly_train._metrics.metric_args import (
    translate_watch_metric,
)
from lightly_train._metrics.task_metric import (
    MetricComputeResult,
    TaskMetric,
    TaskMetricArgs,
)


class MulticlassClassificationTaskMetricArgs(TaskMetricArgs):
    loss_names: ClassVar[list[str]] = ["loss"]

    watch_metric: str = "val_metric/top1_acc_micro"
    accuracy: MulticlassAccuracyArgs | None = Field(
        default_factory=MulticlassAccuracyArgs
    )
    f1: MulticlassF1Args | None = Field(default_factory=MulticlassF1Args)
    precision: MulticlassPrecisionArgs | None = Field(
        default_factory=MulticlassPrecisionArgs
    )
    recall: MulticlassRecallArgs | None = Field(default_factory=MulticlassRecallArgs)

    def get_task_metric(  # type: ignore[override]
        self,
        *,
        split: str,
        class_names: list[str],
        log_classwise: bool,
        classwise_metric_args: MulticlassClassificationTaskMetricArgs | None,
    ) -> ClassificationTaskMetric:
        return ClassificationTaskMetric(
            task_metric_args=self,
            split=split,
            class_names=class_names,
            log_classwise=log_classwise,
            classwise_metric_args=classwise_metric_args,
        )


class MultilabelClassificationTaskMetricArgs(TaskMetricArgs):
    loss_names: ClassVar[list[str]] = ["loss"]

    watch_metric: str = "val_metric/f1_macro"
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


ClassificationTaskMetricArgs = (
    MulticlassClassificationTaskMetricArgs | MultilabelClassificationTaskMetricArgs
)


class ClassificationTaskMetric(TaskMetric):
    def __init__(
        self,
        *,
        task_metric_args: ClassificationTaskMetricArgs,
        split: str,
        class_names: list[str],
        log_classwise: bool,
        classwise_metric_args: ClassificationTaskMetricArgs | None,
    ) -> None:
        """Initialize classification metrics container.

        Args:
            task_metric_args: Metrics configuration
            split: Split name (e.g., "val", "train")
            class_names: Class names for all metrics
            log_classwise: Whether to log classwise metrics
            classwise_metric_args: Optional separate args for classwise metrics
        """
        super().__init__(task_metric_args=task_metric_args)
        self.split = split
        self.class_names = class_names
        self.log_classwise = log_classwise
        self._best_metric_key = translate_watch_metric(
            task_metric_args.watch_metric, split
        )

        self.metrics = task_metric_args.build_metric_collection(
            prefix=f"{self.split}_metric/",
            num_classes=len(class_names),
        )
        self.metrics_classwise = task_metric_args.build_classwise_metric_collection(
            log_classwise=log_classwise,
            prefix=f"{self.split}_metric_classwise/",
            classwise_metrics_args=classwise_metric_args,
            class_names=class_names,
            num_classes=len(class_names),
        )
        self.loss_metrics = LossMetrics(
            split=split, loss_names=task_metric_args.loss_names
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update all quality metrics with inputs.

        Args:
            preds: Predictions tensor
            target: Target tensor
        """
        self.metrics.update(preds, target)
        if self.metrics_classwise is not None:
            self.metrics_classwise.update(preds, target)

    def update_loss(
        self,
        loss_dict: Mapping[str, float | Tensor],
        weight: int,
    ) -> None:
        """Accumulate loss values.

        For validation: call with weight=batch_size to get weighted average.
        For training: call each step; loss is overwritten (use compute() after update).

        Args:
            loss_dict: Mapping from loss name (e.g., "loss", "loss_vfl") to value.
                       Only names present in metric_args.loss_names are tracked.
            weight: Sample weight for accumulation (typically batch size).
        """
        self.loss_metrics.update(loss_dict, weight=weight)  # type: ignore[operator]

    def compute(self) -> MetricComputeResult:
        """Compute all metrics and return combined results.

        Returns:
            MetricComputeResult with metrics dict, best_metric_key, and best_metric_value
        """
        result = self.loss_metrics.compute()
        result.update(self.metrics.compute())
        if self.metrics_classwise is not None:
            result.update(self.metrics_classwise.compute())

        best_val = result.get(self._best_metric_key)
        return MetricComputeResult(
            metrics=result,
            best_metric_key=self._best_metric_key if best_val is not None else None,
            best_metric_value=float(best_val) if best_val is not None else None,
            best_head_name=None,
            best_head_metrics=None,
        )
