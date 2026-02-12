# Copyright (c) 2025. Lightly AG and its affiliates.
# All Rights Reserved

from __future__ import annotations

from typing import Literal

from pydantic import Field
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from lightly_train._metrics.base.metric_args import MetricArgs


class MulticlassAccuracyArgs(MetricArgs):
    """Accuracy metric for multiclass classification."""

    topk: list[int] = Field(default_factory=lambda: [1])
    average: list[Literal["micro", "macro", "weighted"]] = Field(
        default_factory=lambda: ["micro"]
    )

    def get_metrics(
        self,
        *,
        classwise: bool,
        num_classes: int,
    ) -> dict[str, Metric]:
        metrics: dict[str, Metric] = {}

        for k in self.topk:
            if k > num_classes:
                continue
            # Topk>1 accuracy doesn't support classwise for multiclass
            if classwise and k > 1:
                continue
            for avg in self.average:
                key = f"top{k}_acc_{avg}"
                metrics[key] = MulticlassAccuracy(
                    num_classes=num_classes,
                    top_k=k,
                    average="none" if (classwise and k == 1) else avg,
                )

        return metrics

    def supports_classwise(self) -> bool:
        # Only top-1 supports classwise
        return 1 in self.topk


class MulticlassF1Args(MetricArgs):
    """F1 score for multiclass classification."""

    average: list[Literal["micro", "macro", "weighted"]] = Field(
        default_factory=lambda: ["macro"]
    )

    def get_metrics(
        self,
        *,
        classwise: bool,
        num_classes: int,
    ) -> dict[str, Metric]:
        metrics: dict[str, Metric] = {}

        for avg in self.average:
            key = f"f1_{avg}"
            metrics[key] = MulticlassF1Score(
                num_classes=num_classes,
                average="none" if classwise else avg,
            )

        return metrics

    def supports_classwise(self) -> bool:
        return True


class MulticlassPrecisionArgs(MetricArgs):
    """Precision metric for multiclass classification."""

    average: list[Literal["micro", "macro", "weighted"]] = Field(
        default_factory=lambda: ["macro"]
    )

    def get_metrics(
        self,
        *,
        classwise: bool,
        num_classes: int,
    ) -> dict[str, Metric]:
        metrics: dict[str, Metric] = {}

        for avg in self.average:
            key = f"precision_{avg}"
            metrics[key] = MulticlassPrecision(
                num_classes=num_classes,
                average="none" if classwise else avg,
            )

        return metrics

    def supports_classwise(self) -> bool:
        return True


class MulticlassRecallArgs(MetricArgs):
    """Recall metric for multiclass classification."""

    average: list[Literal["micro", "macro", "weighted"]] = Field(
        default_factory=lambda: ["macro"]
    )

    def get_metrics(
        self,
        *,
        classwise: bool,
        num_classes: int,
    ) -> dict[str, Metric]:
        metrics: dict[str, Metric] = {}

        for avg in self.average:
            key = f"recall_{avg}"
            metrics[key] = MulticlassRecall(
                num_classes=num_classes,
                average="none" if classwise else avg,
            )

        return metrics

    def supports_classwise(self) -> bool:
        return True
