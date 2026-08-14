#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Any, Literal

import pytest
import torch
from lightning_utilities.core.imports import RequirementCache
from pytest import MonkeyPatch
from torch import Tensor

if RequirementCache("torchmetrics<1.5"):
    # Skip test if torchmetrics version is too old. This can happen if SuperGradients
    # is installed which requires torchmetrics==0.8
    pytest.skip("Old torchmetrics version", allow_module_level=True)


from lightly_train._metrics import mean_average_precision
from lightly_train._metrics.detection.task_metric import (
    ObjectDetectionTaskMetric,
    ObjectDetectionTaskMetricArgs,
)
from lightly_train._metrics.instance_segmentation.task_metric import (
    InstanceSegmentationTaskMetric,
    InstanceSegmentationTaskMetricArgs,
)
from lightly_train._metrics.mean_average_precision import (
    MeanAveragePrecisionArgs,
    get_default_backend,
)

skip_if_no_faster_coco_eval = pytest.mark.skipif(
    not mean_average_precision.FASTER_COCO_EVAL_AVAILABLE,
    reason="faster_coco_eval is not installed",
)

_CANVAS_SIZE = 400
_LOSS_NAMES = ["loss", "loss_vfl", "loss_bbox", "loss_giou"]

# Two images with boxes spanning the small (<32**2), medium, and large (>96**2) COCO
# area buckets and both classes, with imperfect predictions so that the metrics are
# neither 0 nor 1. This makes the backend comparison meaningful across all reported
# metric keys.
_TARGETS: list[dict[str, list[Any]]] = [
    {
        # 100x100 (large) and 20x20 (small).
        "boxes": [[10.0, 10.0, 110.0, 110.0], [200.0, 200.0, 220.0, 220.0]],
        "labels": [0, 1],
    },
    {
        # 60x60 (medium).
        "boxes": [[50.0, 50.0, 110.0, 110.0]],
        "labels": [1],
    },
]
_PREDICTIONS: list[dict[str, list[Any]]] = [
    {
        "boxes": [
            [12.0, 12.0, 108.0, 108.0],  # Good match for the large box.
            [200.0, 200.0, 225.0, 225.0],  # Loose match for the small box.
            [300.0, 300.0, 340.0, 340.0],  # False positive.
        ],
        "scores": [0.9, 0.7, 0.4],
        "labels": [0, 1, 0],
    },
    {
        "boxes": [
            [52.0, 48.0, 112.0, 108.0],  # Good match for the medium box.
            [0.0, 0.0, 20.0, 20.0],  # False positive.
        ],
        "scores": [0.85, 0.3],
        "labels": [1, 0],
    },
]


def _boxes_to_masks(boxes: list[list[float]]) -> Tensor:
    """Converts xyxy boxes to rectangular instance masks of shape (N, H, W)."""
    masks = torch.zeros((len(boxes), _CANVAS_SIZE, _CANVAS_SIZE), dtype=torch.bool)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        masks[i, int(y1) : int(y2), int(x1) : int(x2)] = True
    return masks


def _detection_metric_values(
    backend: Literal["pycocotools", "faster_coco_eval"],
) -> dict[str, float]:
    metric = ObjectDetectionTaskMetric(
        task_metric_args=ObjectDetectionTaskMetricArgs(
            map=MeanAveragePrecisionArgs(backend=backend)
        ),
        split="val",
        class_names=["cat", "dog"],
        box_format="xyxy",
        loss_names=_LOSS_NAMES,
    )
    metric.update_with_predictions(
        preds=[
            {
                "boxes": torch.tensor(pred["boxes"]),
                "scores": torch.tensor(pred["scores"]),
                "labels": torch.tensor(pred["labels"]),
            }
            for pred in _PREDICTIONS
        ],
        target=[
            {
                "boxes": torch.tensor(target["boxes"]),
                "labels": torch.tensor(target["labels"]),
            }
            for target in _TARGETS
        ],
    )
    metric.update_with_losses(
        {name: torch.tensor(0.5) for name in _LOSS_NAMES}, weight=1
    )
    return metric.compute_aggregated_values().metric_values


def _instance_segmentation_metric_values(
    backend: Literal["pycocotools", "faster_coco_eval"],
) -> dict[str, float]:
    metric = InstanceSegmentationTaskMetric(
        task_metric_args=InstanceSegmentationTaskMetricArgs(
            map=MeanAveragePrecisionArgs(backend=backend)
        ),
        split="val",
        class_names=["cat", "dog"],
        loss_names=_LOSS_NAMES,
    )
    metric.update_with_predictions(
        preds=[
            {
                "masks": _boxes_to_masks(pred["boxes"]),
                "scores": torch.tensor(pred["scores"]),
                "labels": torch.tensor(pred["labels"]),
            }
            for pred in _PREDICTIONS
        ],
        target=[
            {
                "masks": _boxes_to_masks(target["boxes"]),
                "labels": torch.tensor(target["labels"]),
            }
            for target in _TARGETS
        ],
    )
    metric.update_with_losses(
        {name: torch.tensor(0.5) for name in _LOSS_NAMES}, weight=1
    )
    return metric.compute_aggregated_values().metric_values


class TestGetDefaultBackend:
    @pytest.mark.parametrize(
        "iou_type, expected",
        [
            # faster_coco_eval only wins when no mask encoding is involved.
            ("bbox", "faster_coco_eval"),
            (("bbox",), "faster_coco_eval"),
            ("segm", "pycocotools"),
            (("segm",), "pycocotools"),
            (("bbox", "segm"), "pycocotools"),
        ],
    )
    def test_available(
        self, monkeypatch: MonkeyPatch, iou_type: Any, expected: str
    ) -> None:
        monkeypatch.setattr(
            mean_average_precision, "FASTER_COCO_EVAL_AVAILABLE", True, raising=True
        )
        assert get_default_backend(iou_type=iou_type) == expected

    @pytest.mark.parametrize("iou_type", ["bbox", "segm", ("bbox", "segm")])
    def test_not_available(self, monkeypatch: MonkeyPatch, iou_type: Any) -> None:
        monkeypatch.setattr(
            mean_average_precision, "FASTER_COCO_EVAL_AVAILABLE", False, raising=True
        )
        assert get_default_backend(iou_type=iou_type) == "pycocotools"

    def test_default_is_auto(self) -> None:
        assert MeanAveragePrecisionArgs().backend == "auto"

    @pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
    def test_explicit_backend_is_kept(self, backend: str) -> None:
        assert MeanAveragePrecisionArgs(backend=backend).backend == backend  # type: ignore[arg-type]

    @skip_if_no_faster_coco_eval
    def test_auto_resolves_on_the_metric(self) -> None:
        """The resolved backend must reach the underlying torchmetrics instance."""
        detection = MeanAveragePrecisionArgs().get_torchmetrics_instances(
            classwise=False,
            prefix="val_metric",
            class_names=["cat"],
            box_format="xyxy",
            iou_type="bbox",
        )["map"]
        segmentation = MeanAveragePrecisionArgs().get_torchmetrics_instances(
            classwise=False,
            prefix="val_metric",
            class_names=["cat"],
            box_format="xyxy",
            iou_type="segm",
        )["map"]
        assert detection._coco_backend.backend == "faster_coco_eval"  # type: ignore[union-attr]
        assert segmentation._coco_backend.backend == "pycocotools"  # type: ignore[union-attr]


class TestBackendParity:
    """The mAP backend feeds watch_metric and therefore best-checkpoint selection.

    These tests guard that switching the backend does not change the reported values.
    """

    @skip_if_no_faster_coco_eval
    def test_detection(self) -> None:
        pycocotools_values = _detection_metric_values(backend="pycocotools")
        faster_values = _detection_metric_values(backend="faster_coco_eval")

        assert pycocotools_values.keys() == faster_values.keys()
        # Guard against the metrics being trivially degenerate, which would make the
        # comparison meaningless.
        assert 0.0 < pycocotools_values["val_metric/map"] < 1.0
        for name, value in pycocotools_values.items():
            assert faster_values[name] == pytest.approx(value, rel=1e-6, abs=1e-6), name

    @skip_if_no_faster_coco_eval
    def test_instance_segmentation(self) -> None:
        pycocotools_values = _instance_segmentation_metric_values(backend="pycocotools")
        faster_values = _instance_segmentation_metric_values(backend="faster_coco_eval")

        assert pycocotools_values.keys() == faster_values.keys()
        assert 0.0 < pycocotools_values["val_metric/map"] < 1.0
        for name, value in pycocotools_values.items():
            assert faster_values[name] == pytest.approx(value, rel=1e-6, abs=1e-6), name
