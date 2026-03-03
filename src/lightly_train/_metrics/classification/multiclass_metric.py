#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Literal

from pydantic import Field
from torchmetrics import Metric

from lightly_train._metrics.classification.metric_args import ClassificationMetricArgs


class MulticlassAccuracyArgs(ClassificationMetricArgs):
    """Accuracy metric for multiclass classification."""

    topk: list[int] = Field(default_factory=lambda: [1, 5], strict=False)
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
        from torchmetrics.classification import MulticlassAccuracy

        metrics: dict[str, Metric] = {}

        for k in self.topk:
            if k > num_classes:
                continue
            # Topk>1 accuracy doesn't support classwise for multiclass
            if classwise and k > 1:
                continue
            if classwise:
                metrics[f"top{k}_acc"] = MulticlassAccuracy(
                    num_classes=num_classes,
                    top_k=k,
                    average="none",
                )
            else:
                for avg in self.average:
                    metrics[f"top{k}_acc_{avg}"] = MulticlassAccuracy(
                        num_classes=num_classes,
                        top_k=k,
                        average=avg,
                    )

        return metrics

    def supports_classwise(self) -> bool:
        # Only top-1 supports classwise
        return 1 in self.topk

    def get_metric_names(self) -> list[str]:
        return [f"top{k}_acc" for k in self.topk]


class MulticlassF1Args(ClassificationMetricArgs):
    """F1 score for multiclass classification."""

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
        from torchmetrics.classification import MulticlassF1Score

        if classwise:
            return {"f1": MulticlassF1Score(num_classes=num_classes, average="none")}
        return {
            f"f1_{avg}": MulticlassF1Score(num_classes=num_classes, average=avg)
            for avg in self.average
        }

    def supports_classwise(self) -> bool:
        return True

    def get_metric_names(self) -> list[str]:
        return ["f1"]


class MulticlassPrecisionArgs(ClassificationMetricArgs):
    """Precision metric for multiclass classification."""

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
        from torchmetrics.classification import MulticlassPrecision

        if classwise:
            return {
                "precision": MulticlassPrecision(
                    num_classes=num_classes, average="none"
                )
            }
        return {
            f"precision_{avg}": MulticlassPrecision(
                num_classes=num_classes, average=avg
            )
            for avg in self.average
        }

    def supports_classwise(self) -> bool:
        return True

    def get_metric_names(self) -> list[str]:
        return ["precision"]


class MulticlassRecallArgs(ClassificationMetricArgs):
    """Recall metric for multiclass classification."""

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
        from torchmetrics.classification import MulticlassRecall

        if classwise:
            return {"recall": MulticlassRecall(num_classes=num_classes, average="none")}
        return {
            f"recall_{avg}": MulticlassRecall(num_classes=num_classes, average=avg)
            for avg in self.average
        }

    def supports_classwise(self) -> bool:
        return True

    def get_metric_names(self) -> list[str]:
        return ["recall"]
