#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field
from torchmetrics import Metric
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelHammingDistance,
    MultilabelPrecision,
    MultilabelRecall,
)

from lightly_train._metrics.metric_args import MetricArgs


class MultilabelAccuracyArgs(MetricArgs):
    """Accuracy metric for multilabel classification."""

    threshold: float = 0.5
    average: set[Literal["micro", "macro", "weighted"]] = Field(
        default_factory=lambda: {"micro"},  # type: ignore[arg-type]
        strict=False,
    )

    def get_metrics(
        self,
        *,
        classwise: bool,
        num_classes: int,
    ) -> dict[str, Metric]:
        metrics: dict[str, Metric] = {}

        for avg in self.average:
            key = f"accuracy_{avg}"
            metrics[key] = MultilabelAccuracy(
                num_labels=num_classes,
                threshold=self.threshold,
                average="none" if classwise else avg,
            )

        return metrics

    def supports_classwise(self) -> bool:
        return True


class MultilabelF1Args(MetricArgs):
    """F1 score for multilabel classification."""

    threshold: float = 0.5
    average: set[Literal["micro", "macro", "weighted"]] = Field(
        default_factory=lambda: {"macro"},  # type: ignore[arg-type]
        strict=False,
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
            metrics[key] = MultilabelF1Score(
                num_labels=num_classes,
                threshold=self.threshold,
                average="none" if classwise else avg,
            )

        return metrics

    def supports_classwise(self) -> bool:
        return True


class MultilabelPrecisionArgs(MetricArgs):
    """Precision metric for multilabel classification."""

    threshold: float = 0.5
    average: set[Literal["micro", "macro", "weighted"]] = Field(
        default_factory=lambda: {"macro"},  # type: ignore[arg-type]
        strict=False,
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
            metrics[key] = MultilabelPrecision(
                num_labels=num_classes,
                threshold=self.threshold,
                average="none" if classwise else avg,
            )

        return metrics

    def supports_classwise(self) -> bool:
        return True


class MultilabelRecallArgs(MetricArgs):
    """Recall metric for multilabel classification."""

    threshold: float = 0.5
    average: set[Literal["micro", "macro", "weighted"]] = Field(
        default_factory=lambda: {"macro"},  # type: ignore[arg-type]
        strict=False,
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
            metrics[key] = MultilabelRecall(
                num_labels=num_classes,
                threshold=self.threshold,
                average="none" if classwise else avg,
            )

        return metrics

    def supports_classwise(self) -> bool:
        return True


class MultilabelAUROCArgs(MetricArgs):
    """AUROC metric for multilabel classification."""

    average: set[Literal["micro", "macro", "weighted"]] = Field(
        default_factory=lambda: {"macro"},  # type: ignore[arg-type]
        strict=False,
    )

    def get_metrics(
        self,
        *,
        classwise: bool,
        num_classes: int,
    ) -> dict[str, Metric]:
        metrics: dict[str, Metric] = {}

        for avg in self.average:
            key = f"auroc_{avg}"
            metrics[key] = MultilabelAUROC(
                num_labels=num_classes,
                average="none" if classwise else avg,
            )

        return metrics

    def supports_classwise(self) -> bool:
        return True


class MultilabelAveragePrecisionArgs(MetricArgs):
    """Average Precision metric for multilabel classification."""

    average: set[Literal["micro", "macro", "weighted"]] = Field(
        default_factory=lambda: {"macro"},  # type: ignore[arg-type]
        strict=False,
    )

    def get_metrics(
        self,
        *,
        classwise: bool,
        num_classes: int,
    ) -> dict[str, Metric]:
        metrics: dict[str, Metric] = {}

        for avg in self.average:
            key = f"avg_precision_{avg}"
            metrics[key] = MultilabelAveragePrecision(
                num_labels=num_classes,
                average="none" if classwise else avg,
            )

        return metrics

    def supports_classwise(self) -> bool:
        return True


class MultilabelHammingDistanceArgs(MetricArgs):
    """Hamming Distance metric for multilabel classification."""

    mode: ClassVar[Literal["min", "max"]] = "min"

    threshold: float = 0.5

    def get_metrics(
        self,
        *,
        classwise: bool,
        num_classes: int,
    ) -> dict[str, Metric]:
        # Hamming distance doesn't support classwise or averaging
        if classwise:
            return {}

        metrics: dict[str, Metric] = {}
        metrics["hamming_distance"] = MultilabelHammingDistance(
            num_labels=num_classes,
            threshold=self.threshold,
        )

        return metrics

    def supports_classwise(self) -> bool:
        return False
