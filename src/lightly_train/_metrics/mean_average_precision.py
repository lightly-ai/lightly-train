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

from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torchmetrics import Metric as TorchmetricsMetric
from torchmetrics.detection.mean_ap import (
    MeanAveragePrecision as TorchmetricsMeanAveragePrecision,
)

from lightly_train._metrics.metric_args import MetricArgs

# Use the same requirement string as torchmetrics' own _FASTER_COCO_EVAL_AVAILABLE so
# that our check can never disagree with the backend validation inside torchmetrics.
FASTER_COCO_EVAL_AVAILABLE = RequirementCache("faster_coco_eval")


def get_default_backend(
    iou_type: Literal["bbox", "segm"] | tuple[Literal["bbox", "segm"], ...],
) -> Literal["pycocotools", "faster_coco_eval"]:
    """Returns the mAP backend to use when the backend is set to "auto".

    faster_coco_eval is a C++ reimplementation of COCOeval. Both backends return
    identical values, but they have very different performance characteristics:

    - The COCOeval part, which runs in a single blocking call at the end of every
      validation run, is ~5x faster with faster_coco_eval.
    - The mask RLE encoding, which runs on every update and therefore only matters for
      iou_type="segm", is ~25% slower with faster_coco_eval.

    For "bbox" there is no mask encoding, so faster_coco_eval is a clear win. For
    "segm" the encoding dominates the total cost because masks are encoded at full
    image resolution for every instance, which makes faster_coco_eval a net loss.
    Hence we only default to it when no mask encoding is involved. Users can still
    select either backend explicitly.
    """
    if not FASTER_COCO_EVAL_AVAILABLE:
        return "pycocotools"
    iou_types = (iou_type,) if isinstance(iou_type, str) else tuple(iou_type)
    if "segm" in iou_types:
        return "pycocotools"
    return "faster_coco_eval"


class MeanAveragePrecisionArgs(MetricArgs):
    iou_thresholds: list[float] | None = None
    rec_thresholds: list[float] | None = None
    max_detection_thresholds: list[int] | None = None
    average: Literal["macro", "micro"] = "macro"
    # "auto" picks the fastest backend available for the given iou_type, see
    # get_default_backend.
    backend: Literal["auto", "pycocotools", "faster_coco_eval"] = "auto"

    def get_torchmetrics_instances(
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
            backend=None if self.backend == "auto" else self.backend,
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
        backend: Literal["pycocotools", "faster_coco_eval"] | None = None,
        **kwargs: Any,
    ) -> None:
        if backend is None:
            backend = get_default_backend(iou_type=iou_type)
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
