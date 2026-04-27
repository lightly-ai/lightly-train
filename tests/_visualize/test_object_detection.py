#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math

import torch
from torch import Tensor

from lightly_train._visualize import object_detection
from lightly_train.types import ObjectDetectionBatch


def _make_batch(
    *,
    batch_size: int = 1,
    height: int = 32,
    width: int = 32,
    bboxes: list[Tensor] | None = None,
    classes: list[Tensor] | None = None,
) -> ObjectDetectionBatch:
    images = torch.rand(batch_size, 3, height, width)
    if bboxes is None:
        bboxes = [torch.zeros(0, 4) for _ in range(batch_size)]
    if classes is None:
        classes = [torch.zeros(0, dtype=torch.long) for _ in range(batch_size)]
    return ObjectDetectionBatch(
        image_path=[f"img_{i}.jpg" for i in range(batch_size)],
        image=images,
        bboxes=bboxes,
        classes=classes,
        original_size=[(width, height)] * batch_size,
    )


def _make_results(
    *,
    batch_size: int = 1,
    boxes_per_image: int = 0,
    score: float = 1.0,
    img_h: int = 32,
    img_w: int = 32,
) -> list[dict[str, Tensor]]:
    results = []
    for _ in range(batch_size):
        if boxes_per_image == 0:
            results.append(
                {
                    "boxes": torch.zeros(0, 4),
                    "labels": torch.zeros(0, dtype=torch.long),
                    "scores": torch.zeros(0),
                }
            )
        else:
            results.append(
                {
                    "boxes": torch.tensor(
                        [[0.0, 0.0, img_w / 2, img_h / 2]] * boxes_per_image
                    ),
                    "labels": torch.zeros(boxes_per_image, dtype=torch.long),
                    "scores": torch.full((boxes_per_image,), score),
                }
            )
    return results


class TestPlotObjectDetectionLabels:
    def test_plot_object_detection_labels__output_size_single_image(self) -> None:
        batch = _make_batch(height=32, width=64)
        result = object_detection.plot_object_detection_labels(
            batch=batch,
            included_classes={},
            max_images=1,
        )
        assert result.size == (64, 32)

    def test_plot_object_detection_labels__max_images_limits_grid(self) -> None:
        batch = _make_batch(batch_size=4, height=16, width=16)
        result = object_detection.plot_object_detection_labels(
            batch=batch,
            included_classes={},
            max_images=2,
        )
        n_cols = math.ceil(math.sqrt(2))
        n_rows = math.ceil(2 / n_cols)
        assert result.size == (n_cols * 16, n_rows * 16)

    def test_plot_object_detection_labels__four_images_grid_size(self) -> None:
        batch = _make_batch(batch_size=4, height=8, width=8)
        result = object_detection.plot_object_detection_labels(
            batch=batch,
            included_classes={},
            max_images=4,
        )
        # ceil(sqrt(4))=2 cols, ceil(4/2)=2 rows → (16, 16)
        assert result.size == (16, 16)

    def test_plot_object_detection_labels__with_bboxes(self) -> None:
        bboxes = [torch.tensor([[0.5, 0.5, 0.4, 0.4]])]
        classes = [torch.tensor([1], dtype=torch.long)]
        batch = _make_batch(bboxes=bboxes, classes=classes)
        result = object_detection.plot_object_detection_labels(
            batch=batch,
            included_classes={1: "dog"},
            max_images=1,
        )
        assert result.size == (32, 32)

    def test_plot_object_detection_labels__unknown_class_uses_fallback(self) -> None:
        bboxes = [torch.tensor([[0.5, 0.5, 0.4, 0.4]])]
        classes = [torch.tensor([99], dtype=torch.long)]
        batch = _make_batch(bboxes=bboxes, classes=classes)
        result = object_detection.plot_object_detection_labels(
            batch=batch,
            included_classes={},
            max_images=1,
        )
        assert result.size == (32, 32)

    def test_plot_object_detection_labels__with_mean_std(self) -> None:
        batch = _make_batch()
        result = object_detection.plot_object_detection_labels(
            batch=batch,
            included_classes={},
            max_images=1,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        assert result.size == (32, 32)


class TestPlotObjectDetectionPredictions:
    def test_plot_object_detection_predictions__output_size_single_image(self) -> None:
        batch = _make_batch(height=32, width=64)
        results = _make_results(img_h=32, img_w=64)
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=results,
            included_classes={},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.size == (64, 32)

    def test_plot_object_detection_predictions__max_images_limits_grid(self) -> None:
        batch = _make_batch(batch_size=4, height=16, width=16)
        results = _make_results(batch_size=4, img_h=16, img_w=16)
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=results,
            included_classes={},
            max_images=2,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        n_cols = math.ceil(math.sqrt(2))
        n_rows = math.ceil(2 / n_cols)
        assert result.size == (n_cols * 16, n_rows * 16)

    def test_plot_object_detection_predictions__empty_boxes(self) -> None:
        batch = _make_batch()
        results = _make_results(boxes_per_image=0, img_h=32, img_w=32)
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=results,
            included_classes={},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.size == (32, 32)

    def test_plot_object_detection_predictions__score_threshold_filters_low_scores(
        self,
    ) -> None:
        # All scores (0.3) below threshold (0.5) → no boxes drawn.
        batch = _make_batch()
        results = _make_results(boxes_per_image=3, score=0.3)
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=results,
            included_classes={0: "cat"},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.size == (32, 32)

    def test_plot_object_detection_predictions__max_pred_boxes_limits_drawn_boxes(
        self,
    ) -> None:
        # 10 boxes with max_pred_boxes=3; only top 3 by score should be drawn.
        batch = _make_batch()
        results = _make_results(boxes_per_image=10, score=0.9, img_h=32, img_w=32)
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=results,
            included_classes={0: "cat"},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=3,
        )
        assert result.size == (32, 32)

    def test_plot_object_detection_predictions__unknown_class_uses_fallback(
        self,
    ) -> None:
        batch = _make_batch()
        results = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 16.0, 16.0]]),
                "labels": torch.tensor([42], dtype=torch.long),
                "scores": torch.tensor([0.9]),
            }
        ]
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=results,
            included_classes={},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.size == (32, 32)

    def test_plot_object_detection_predictions__with_mean_std(self) -> None:
        batch = _make_batch()
        results = _make_results()
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=results,
            included_classes={},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        assert result.size == (32, 32)
