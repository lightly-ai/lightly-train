#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from lightly_train._visualize import object_detection
from lightly_train.types import ObjectDetectionBatch


def _make_batch(
    *,
    batch_size: int = 1,
    height: int = 32,
    width: int = 32,
    image: Tensor | None = None,
    bboxes: list[Tensor] | None = None,
    classes: list[Tensor] | None = None,
) -> ObjectDetectionBatch:
    if image is None:
        image = torch.rand(batch_size, 3, height, width)
    else:
        batch_size = image.shape[0]
        height = image.shape[2]
        width = image.shape[3]
    if bboxes is None:
        bboxes = [torch.zeros(0, 4) for _ in range(batch_size)]
    if classes is None:
        classes = [torch.zeros(0, dtype=torch.long) for _ in range(batch_size)]
    return ObjectDetectionBatch(
        image_path=[f"img_{i}.jpg" for i in range(batch_size)],
        image=image,
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
    def test_plot_object_detection_labels_grid_caps_at_max_images(self) -> None:
        batch = _make_batch(batch_size=4, height=16, width=16)
        result = object_detection.plot_object_detection_labels(
            batch=batch, included_classes={}, max_images=2
        )
        n_cols = math.ceil(math.sqrt(2))
        n_rows = math.ceil(2 / n_cols)
        assert result.size == (n_cols * 16, n_rows * 16)

    def test_plot_object_detection_labels_bboxes_drawn(self) -> None:
        # cxcywh [0.25, 0.25, 0.5, 0.5] maps to xyxy [0, 0, 16, 16] on a 32×32 image.
        bboxes = [torch.tensor([[0.25, 0.25, 0.5, 0.5]])]
        classes = [torch.tensor([1], dtype=torch.long)]
        batch = _make_batch(
            image=torch.zeros(1, 3, 32, 32), bboxes=bboxes, classes=classes
        )
        result = object_detection.plot_object_detection_labels(
            batch=batch, included_classes={1: "dog"}, max_images=1
        )
        assert result.getpixel((0, 0)) != (0, 0, 0)

    def test_plot_object_detection_labels_unknown_class_draws_box(self) -> None:
        # cxcywh [0.25, 0.25, 0.5, 0.5] maps to xyxy [0, 0, 16, 16] on a 32×32 image.
        bboxes = [torch.tensor([[0.25, 0.25, 0.5, 0.5]])]
        classes = [torch.tensor([99], dtype=torch.long)]
        batch = _make_batch(
            image=torch.zeros(1, 3, 32, 32), bboxes=bboxes, classes=classes
        )
        result = object_detection.plot_object_detection_labels(
            batch=batch, included_classes={}, max_images=1
        )
        assert result.getpixel((0, 0)) != (0, 0, 0)

    def test_plot_object_detection_labels_mean_std_denormalizes_image(self) -> None:
        batch = _make_batch(image=torch.zeros(1, 3, 32, 32))
        result = object_detection.plot_object_detection_labels(
            batch=batch,
            included_classes={},
            max_images=1,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        assert result.getextrema() != ((0, 0), (0, 0), (0, 0))


class TestPlotObjectDetectionPredictions:
    def test_plot_object_detection_predictions_grid_caps_at_max_images(self) -> None:
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

    def test_plot_object_detection_predictions_empty_boxes_produces_clean_image(
        self,
    ) -> None:
        batch = _make_batch(image=torch.zeros(1, 3, 32, 32))
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=_make_results(
                batch_size=1, boxes_per_image=0, img_h=32, img_w=32
            ),
            included_classes={},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.getextrema() == ((0, 0), (0, 0), (0, 0))

    @pytest.mark.parametrize("score,drawn", [(0.9, True), (0.3, False)])
    def test_plot_object_detection_predictions_score_threshold(
        self, score: float, drawn: bool
    ) -> None:
        batch = _make_batch(image=torch.zeros(1, 3, 64, 64))
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=[
                {
                    "boxes": torch.tensor([[0.0, 0.0, 32.0, 32.0]]),
                    "labels": torch.zeros(1, dtype=torch.long),
                    "scores": torch.tensor([score]),
                }
            ],
            included_classes={0: "cat"},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert (result.getpixel((0, 0)) != (0, 0, 0)) == drawn

    def test_plot_object_detection_predictions_max_pred_boxes_limits_drawn_boxes(
        self,
    ) -> None:
        batch = _make_batch(image=torch.zeros(1, 3, 10, 400))
        # Generate 5 boxes (xyxy) with descending scores; only the top 3 should be drawn.
        # Boxes are 10×10 (square), spaced 20 pixels apart, starting at x=0.
        boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 0.0, 30.0, 10.0],
                [40.0, 0.0, 50.0, 10.0],
                [300.0, 0.0, 310.0, 10.0],  # suppressed by max_pred_boxes=3
                [350.0, 0.0, 360.0, 10.0],  # suppressed by max_pred_boxes=3
            ]
        )
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=[
                {
                    "boxes": boxes,
                    "labels": torch.zeros(5, dtype=torch.long),
                    "scores": torch.tensor([0.90, 0.80, 0.70, 0.60, 0.55]),
                }
            ],
            included_classes={0: "cat"},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=3,
        )
        # Only the first 3 boxes should be drawn; the last 2 should be suppressed.
        assert result.getpixel((5, 0)) != (0, 0, 0) 
        assert result.getpixel((25, 0)) != (0, 0, 0)
        assert result.getpixel((45, 0)) != (0, 0, 0)
        assert result.getpixel((305, 0)) == (0, 0, 0)
        assert result.getpixel((355, 0)) == (0, 0, 0)

    def test_plot_object_detection_predictions_unknown_class_draws_box(self) -> None:
        # check that a box is drawn even when the class ID isn't in included_classes; the label will just show the numeric class ID.
        batch = _make_batch(image=torch.zeros(1, 3, 32, 32))
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=[
                {
                    "boxes": torch.tensor([[0.0, 0.0, 16.0, 16.0]]),
                    "labels": torch.tensor([42], dtype=torch.long),
                    "scores": torch.tensor([0.9]),
                }
            ],
            included_classes={},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.getpixel((0, 0)) != (0, 0, 0)

    def test_plot_object_detection_predictions_mean_std_denormalizes_image(
        self,
    ) -> None:
        # check that the image is denormalized when mean and std are provided, by verifying that the output image isn't all black.
        batch = _make_batch(image=torch.zeros(1, 3, 32, 32))
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=_make_results(batch_size=1, boxes_per_image=0, img_h=32, img_w=32),
            included_classes={},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        assert result.getextrema() != ((0, 0), (0, 0), (0, 0))

    def test_plot_object_detection_predictions_bbox_scaling_uniform(self) -> None:
        # check that bounding boxes are correctly scaled from original image coordinates to tensor coordinates, when the scaling is uniform (same factor for x and y).
        # Original image is 128×128; tensor is 64×64 (uniform 2× downscale).
        # Box at [64, 64, 128, 128] in original coords maps to [32, 32, 64, 64].
        batch = ObjectDetectionBatch(
            image_path=["img_0.jpg"],
            image=torch.zeros(1, 3, 64, 64),
            bboxes=[torch.zeros(0, 4)],
            classes=[torch.zeros(0, dtype=torch.long)],
            original_size=[(128, 128)],
        )
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=[
                {
                    "boxes": torch.tensor([[64.0, 64.0, 128.0, 128.0]]),
                    "labels": torch.zeros(1, dtype=torch.long),
                    "scores": torch.tensor([0.9]),
                }
            ],
            included_classes={0: "obj"},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.getpixel((32, 32)) != (0, 0, 0)  # scaled box top-left corner
        assert result.getpixel((0, 0)) == (0, 0, 0)  # outside the scaled box

    def test_plot_object_detection_predictions_bbox_scaling_asymmetric(self) -> None:
        # check that bounding boxes are correctly scaled from original image coordinates to tensor coordinates, when the scaling is asymmetric (different factors for x and y).
        # Original image is 128 wide × 64 tall; tensor is 64×64.
        # x-coords are halved, y-coords are unchanged.
        # Box at [96, 32, 128, 64] in original → [48, 32, 64, 64] in tensor.
        batch = ObjectDetectionBatch(
            image_path=["img_0.jpg"],
            image=torch.zeros(1, 3, 64, 64),
            bboxes=[torch.zeros(0, 4)],
            classes=[torch.zeros(0, dtype=torch.long)],
            original_size=[(128, 64)],
        )
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=[
                {
                    "boxes": torch.tensor([[96.0, 32.0, 128.0, 64.0]]),
                    "labels": torch.zeros(1, dtype=torch.long),
                    "scores": torch.tensor([0.9]),
                }
            ],
            included_classes={0: "obj"},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.getpixel((48, 40)) != (0, 0, 0)  # left edge of scaled box
        assert result.getpixel((20, 40)) == (0, 0, 0)  # left of the scaled box
