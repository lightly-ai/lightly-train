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

from lightly_train._metrics.classification.metric_args import ClassificationMetricArgs


class MultilabelAccuracyArgs(ClassificationMetricArgs):
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
        # Type ignore for old torchmetrics versions
        from torchmetrics.classification import (  # type: ignore[attr-defined]
            MultilabelAccuracy,
        )

        if classwise:
            return {
                "accuracy": MultilabelAccuracy(
                    num_labels=num_classes,
                    threshold=self.threshold,
                    average="none",
                )
            }
        return {
            f"accuracy_{avg}": MultilabelAccuracy(
                num_labels=num_classes,
                threshold=self.threshold,
                average=avg,
            )
            for avg in self.average
        }

    def supports_classwise(self) -> bool:
        return True

    def get_metric_names(self) -> list[str]:
        return ["accuracy"]


class MultilabelF1Args(ClassificationMetricArgs):
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
        # Type ignore for old torchmetrics versions
        from torchmetrics.classification import (  # type: ignore[attr-defined]
            MultilabelF1Score,
        )

        if classwise:
            return {
                "f1": MultilabelF1Score(
                    num_labels=num_classes,
                    threshold=self.threshold,
                    average="none",
                )
            }
        return {
            f"f1_{avg}": MultilabelF1Score(
                num_labels=num_classes,
                threshold=self.threshold,
                average=avg,
            )
            for avg in self.average
        }

    def supports_classwise(self) -> bool:
        return True

    def get_metric_names(self) -> list[str]:
        return ["f1"]


class MultilabelPrecisionArgs(ClassificationMetricArgs):
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
        # Type ignore for old torchmetrics versions
        from torchmetrics.classification import (  # type: ignore[attr-defined]
            MultilabelPrecision,
        )

        if classwise:
            return {
                "precision": MultilabelPrecision(
                    num_labels=num_classes,
                    threshold=self.threshold,
                    average="none",
                )
            }
        return {
            f"precision_{avg}": MultilabelPrecision(
                num_labels=num_classes,
                threshold=self.threshold,
                average=avg,
            )
            for avg in self.average
        }

    def supports_classwise(self) -> bool:
        return True

    def get_metric_names(self) -> list[str]:
        return ["precision"]


class MultilabelRecallArgs(ClassificationMetricArgs):
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
        # Type ignore for old torchmetrics versions
        from torchmetrics.classification import (  # type: ignore[attr-defined]
            MultilabelRecall,
        )

        if classwise:
            return {
                "recall": MultilabelRecall(
                    num_labels=num_classes,
                    threshold=self.threshold,
                    average="none",
                )
            }
        return {
            f"recall_{avg}": MultilabelRecall(
                num_labels=num_classes,
                threshold=self.threshold,
                average=avg,
            )
            for avg in self.average
        }

    def supports_classwise(self) -> bool:
        return True

    def get_metric_names(self) -> list[str]:
        return ["recall"]


class MultilabelAUROCArgs(ClassificationMetricArgs):
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
        # Type ignore for old torchmetrics versions
        from torchmetrics.classification import (  # type: ignore[attr-defined]
            MultilabelAUROC,
        )

        if classwise:
            return {"auroc": MultilabelAUROC(num_labels=num_classes, average="none")}
        return {
            f"auroc_{avg}": MultilabelAUROC(num_labels=num_classes, average=avg)
            for avg in self.average
        }

    def supports_classwise(self) -> bool:
        return True

    def get_metric_names(self) -> list[str]:
        return ["auroc"]


class MultilabelAveragePrecisionArgs(ClassificationMetricArgs):
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
        # Type ignore for old torchmetrics versions
        from torchmetrics.classification import (  # type: ignore[attr-defined]
            MultilabelAveragePrecision,
        )

        if classwise:
            return {
                "avg_precision": MultilabelAveragePrecision(
                    num_labels=num_classes, average="none"
                )
            }
        return {
            f"avg_precision_{avg}": MultilabelAveragePrecision(
                num_labels=num_classes, average=avg
            )
            for avg in self.average
        }

    def supports_classwise(self) -> bool:
        return True

    def get_metric_names(self) -> list[str]:
        return ["avg_precision"]


class MultilabelHammingDistanceArgs(ClassificationMetricArgs):
    """Hamming Distance metric for multilabel classification."""

    watch_mode: ClassVar[Literal["min", "max"]] = "min"

    threshold: float = 0.5

    def get_metrics(
        self,
        *,
        classwise: bool,
        num_classes: int,
    ) -> dict[str, Metric]:
        # Type ignore for old torchmetrics versions
        from torchmetrics.classification import (  # type: ignore[attr-defined]
            MultilabelHammingDistance,
        )

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

    def get_metric_names(self) -> list[str]:
        return ["hamming_distance"]
