#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from torch import Tensor
from torchmetrics import Metric as TorchmetricsMetric
from torchmetrics.detection.mean_ap import (
    MeanAveragePrecision as TorchmetricsMeanAveragePrecision,
)

from lightly_train._metrics.metric_args import MetricArgs


class MeanAveragePrecisionArgs(MetricArgs):
    iou_thresholds: list[float] | None = None
    rec_thresholds: list[float] | None = None
    max_detection_thresholds: list[int] | None = None
    average: Literal["macro", "micro"] = "macro"
    backend: Literal["pycocotools", "faster_coco_eval"] = "pycocotools"

    def get_metrics(
        self,
        *,
        classwise: bool,
        prefix: str,
        class_names: Sequence[str],
        box_format: Literal["xyxy", "xywh", "cxcywh"],
        iou_type: Literal["bbox", "segm"] | tuple[Literal["bbox", "segm"], ...],
    ) -> dict[str, TorchmetricsMetric]:
        """Create MeanAveragePrecision metric instance.

        Args:
            classwise: If True, compute per-class metrics
            num_classes: Number of classes (unused for detection metrics)

        Returns:
            Dictionary with single "map" key containing the metric instance
        """
        metrics: dict[str, TorchmetricsMetric] = {}

        map_metric = MeanAveragePrecision(
            prefix=prefix,
            class_names=class_names,
            class_metrics=classwise,
            box_format=box_format,
            iou_type=iou_type,
            iou_thresholds=self.iou_thresholds,
            rec_thresholds=self.rec_thresholds,
            max_detection_thresholds=self.max_detection_thresholds,
            backend=self.backend,
            average=self.average,
        )
        map_metric.warn_on_many_detections = False  # type: ignore[attr-defined]
        metrics["map"] = map_metric
        return metrics

    def supports_classwise(self) -> bool:
        """MeanAveragePrecision supports classwise computation."""
        return True

    def get_metric_names(self) -> list[str]:
        names = [
            "map",
            "map_small",
            "map_medium",
            "map_large",
            "mar_small",
            "mar_medium",
            "mar_large",
        ]
        thresholds = self.iou_thresholds
        if thresholds is None or 0.5 in thresholds:
            names.append("map_50")
        if thresholds is None or 0.75 in thresholds:
            names.append("map_75")
        max_det = (
            self.max_detection_thresholds
            if self.max_detection_thresholds is not None
            else [1, 10, 100]
        )
        names.extend(f"mar_{n}" for n in max_det)
        return names


class MeanAveragePrecision(TorchmetricsMeanAveragePrecision):
    """Wrapper around torchmetrics MeanAveragePrecision to flatten results from
    .compute() calls into a single-level dictionary.

    It also handles class names and prefixes for metric keys.
    """

    def __init__(
        self,
        prefix: str,
        class_names: Sequence[str],
        box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
        iou_type: Literal["bbox", "segm"]
        | tuple[Literal["bbox", "segm"], ...] = "bbox",
        iou_thresholds: Sequence[float] | None = None,
        rec_thresholds: Sequence[float] | None = None,
        max_detection_thresholds: Sequence[int] | None = None,
        class_metrics: bool = False,
        extended_summary: bool = False,
        average: Literal["macro", "micro"] = "macro",
        backend: Literal["pycocotools", "faster_coco_eval"] = "pycocotools",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            box_format=box_format,
            iou_type=iou_type,  # type: ignore
            iou_thresholds=iou_thresholds,  # type: ignore
            rec_thresholds=rec_thresholds,  # type: ignore
            max_detection_thresholds=max_detection_thresholds,  # type: ignore
            class_metrics=class_metrics,
            extended_summary=extended_summary,  # type: ignore
            average=average,  # type: ignore
            backend=backend,  # type: ignore
            **kwargs,
        )
        self.prefix = prefix
        self.class_names = class_names

    def compute(self) -> dict[str, Tensor]:
        metrics = super().compute()
        result = {}
        for name, value in metrics.items():
            if "class" in name:
                # Skip classwise metrics
                continue
            result[f"{self.prefix}/{name}"] = value

        if self.class_metrics:
            # Flatten classwise map
            classes = metrics["classes"]
            map_per_class = metrics["map_per_class"]
            for i, class_idx in enumerate(classes):
                class_name = self.class_names[class_idx.item()]
                class_value = map_per_class[i]
                result[f"{self.prefix}_classwise/map_{class_name}"] = class_value
        return result
