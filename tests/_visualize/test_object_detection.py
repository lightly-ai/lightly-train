#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
import torch
from PIL.Image import Image as PILImage
from torch import Tensor

from lightly_train._visualize import object_detection, utils
from lightly_train.types import ObjectDetectionBatch

_BACKGROUND_COLOR: float = 0.0
_BACKGROUND_PIXEL: tuple[int, int, int] = (0, 0, 0)


def _make_batch(
    *,
    batch_size: int = 1,
    height: int = 32,
    width: int = 32,
    bboxes: list[Tensor] | None = None,
    classes: list[Tensor] | None = None,
) -> ObjectDetectionBatch:
    image = torch.rand(batch_size, 3, height, width)
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


def _make_batch_from_image(
    *,
    image: Tensor,
    bboxes: list[Tensor] | None = None,
    classes: list[Tensor] | None = None,
) -> ObjectDetectionBatch:
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


def _make_empty_results(*, batch_size: int = 1) -> list[dict[str, Tensor]]:
    return [
        {
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "scores": torch.zeros(0),
        }
        for _ in range(batch_size)
    ]


def _assert_bbox_corners_have_color(
    *,
    image: PILImage,
    xyxy: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = xyxy
    assert image.getpixel((x1, y1)) == color
    assert image.getpixel((x2, y1)) == color
    assert image.getpixel((x1, y2)) == color
    assert image.getpixel((x2, y2)) == color


class TestPlotObjectDetectionLabels:
    def test_plot_object_detection_labels_grid_caps_at_max_images(self) -> None:
        batch = _make_batch(batch_size=4, height=16, width=16)
        result = object_detection.plot_object_detection_labels(
            batch=batch, included_classes={}, max_images=2
        )
        assert result.size == (32, 16)

    def test_plot_object_detection_labels_bboxes_drawn(self) -> None:
        # cxcywh [0.25, 0.25, 0.5, 0.5] maps to xyxy [0, 0, 64, 64] on a 128×128 image.
        # The image is large enough that the box interior (center) is clear of both
        # the outline and the label rectangle, so it stays background color.
        bboxes = [torch.tensor([[0.25, 0.25, 0.5, 0.5]])]
        classes = [torch.tensor([1], dtype=torch.long)]
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 128, 128), _BACKGROUND_COLOR),
            bboxes=bboxes,
            classes=classes,
        )
        result = object_detection.plot_object_detection_labels(
            batch=batch, included_classes={1: "dog"}, max_images=1
        )
        # All four corners of the bbox outline must be painted with class 1's color.
        _assert_bbox_corners_have_color(
            image=result, xyxy=(0, 0, 64, 64), color=utils._get_class_color(1)
        )
        # The label rectangle fills its interior with the class color. (2, 2) sits
        # in the top-left padding of the label — off the outline, before any glyph.
        assert result.getpixel((2, 2)) == utils._get_class_color(1)
        # The box interior is not filled — only the outline and label are drawn.
        assert result.getpixel((32, 32)) == _BACKGROUND_PIXEL
        # A pixel outside the box stays untouched.
        assert result.getpixel((127, 127)) == _BACKGROUND_PIXEL

    def test_plot_object_detection_labels_unknown_class_draws_box(self) -> None:
        # cxcywh [0.25, 0.25, 0.5, 0.5] maps to xyxy [0, 0, 64, 64] on a 128×128 image.
        bboxes = [torch.tensor([[0.25, 0.25, 0.5, 0.5]])]
        classes = [torch.tensor([99], dtype=torch.long)]
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 128, 128), _BACKGROUND_COLOR),
            bboxes=bboxes,
            classes=classes,
        )
        result = object_detection.plot_object_detection_labels(
            batch=batch, included_classes={}, max_images=1
        )
        # Unknown class IDs still get a deterministic color from `_get_class_color`.
        _assert_bbox_corners_have_color(
            image=result, xyxy=(0, 0, 64, 64), color=utils._get_class_color(99)
        )
        assert result.getpixel((2, 2)) == utils._get_class_color(99)
        assert result.getpixel((32, 32)) == _BACKGROUND_PIXEL
        assert result.getpixel((127, 127)) == _BACKGROUND_PIXEL

    @pytest.mark.parametrize(
        "image_value, mean, std, expected_pixel",
        [
            # image=0 -> denormalized = mean. Per-channel means verify channel order.
            # (0.2, 0.4, 0.6) * 255 = (51, 102, 153).
            (0.0, (0.2, 0.4, 0.6), (0.5, 0.5, 0.5), (51, 102, 153)),
            # Non-zero image: pixel = image * std + mean.
            # 0.4 * 0.5 + 0.5 = 0.7 -> 178 (PIL truncates 0.7 * 255 = 178.5).
            (0.4, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (178, 178, 178)),
            # Per-channel std with zero mean: 0.5 * (0.2, 0.4, 0.6) = (0.1, 0.2, 0.3)
            # -> (25, 51, 76).
            (0.5, (0.0, 0.0, 0.0), (0.2, 0.4, 0.6), (25, 51, 76)),
            # Values > 1 are clamped to 1 -> 255.
            (5.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (255, 255, 255)),
            # Values < 0 are clamped to 0.
            (-5.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0, 0, 0)),
        ],
    )
    def test_plot_object_detection_labels_mean_std_denormalizes_image(
        self,
        image_value: float,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        expected_pixel: tuple[int, int, int],
    ) -> None:
        batch = _make_batch_from_image(image=torch.full((1, 3, 32, 32), image_value))
        result = object_detection.plot_object_detection_labels(
            batch=batch,
            included_classes={},
            max_images=1,
            mean=mean,
            std=std,
        )
        # No bboxes are drawn, so every pixel reflects the denormalized image.
        assert result.getpixel((0, 0)) == expected_pixel
        assert result.getpixel((31, 31)) == expected_pixel

    def test_plot_object_detection_labels_no_mean_std_skips_denormalization(
        self,
    ) -> None:
        # Without mean/std, the image tensor is passed through unchanged.
        # Uniform 0.4 -> 102.
        batch = _make_batch_from_image(image=torch.full((1, 3, 32, 32), 0.4))
        result = object_detection.plot_object_detection_labels(
            batch=batch, included_classes={}, max_images=1
        )
        assert result.getpixel((0, 0)) == (102, 102, 102)
        assert result.getpixel((31, 31)) == (102, 102, 102)

    def test_plot_object_detection_labels_mixed_empty_nonempty_annotations(
        self,
    ) -> None:
        # Image 0 has one box; image 1 has none.
        # cxcywh [0.5, 0.5, 0.75, 0.75] maps to xyxy [8, 8, 56, 56] on a 64×64 image.
        bboxes = [torch.tensor([[0.5, 0.5, 0.75, 0.75]]), torch.zeros(0, 4)]
        classes = [
            torch.tensor([0], dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        ]
        batch = _make_batch_from_image(
            image=torch.full((2, 3, 64, 64), _BACKGROUND_COLOR),
            bboxes=bboxes,
            classes=classes,
        )
        result = object_detection.plot_object_detection_labels(
            batch=batch, included_classes={0: "cat"}, max_images=2
        )
        # Grid is 2×1 (128 wide, 64 tall): image 0 at x=0..63, image 1 at x=64..127.
        _assert_bbox_corners_have_color(
            image=result, xyxy=(8, 8, 56, 56), color=utils._get_class_color(0)
        )
        assert result.getpixel((60, 60)) == _BACKGROUND_PIXEL
        assert result.getpixel((32, 32)) == _BACKGROUND_PIXEL
        assert result.getpixel((64, 0)) == _BACKGROUND_PIXEL


class TestPlotObjectDetectionPredictions:
    def test_plot_object_detection_predictions_grid_caps_at_max_images(self) -> None:
        batch = _make_batch(batch_size=4, height=16, width=16)
        results = _make_empty_results(batch_size=4)
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=results,
            included_classes={},
            max_images=2,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.size == (32, 16)

    def test_plot_object_detection_predictions_empty_boxes_produces_clean_image(
        self,
    ) -> None:
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 32, 32), _BACKGROUND_COLOR)
        )
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=_make_empty_results(batch_size=1),
            included_classes={},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.getpixel((0, 0)) == _BACKGROUND_PIXEL
        assert result.getpixel((31, 31)) == _BACKGROUND_PIXEL

    def test_plot_object_detection_predictions_mixed_empty_nonempty_annotations(
        self,
    ) -> None:
        # Image 0 has one predicted box; image 1 has none.
        batch = _make_batch_from_image(
            image=torch.full((2, 3, 64, 64), _BACKGROUND_COLOR)
        )
        results = [
            {
                "boxes": torch.tensor([[8.0, 8.0, 56.0, 56.0]]),
                "labels": torch.zeros(1, dtype=torch.long),
                "scores": torch.tensor([0.9]),
            },
            {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.long),
                "scores": torch.zeros(0),
            },
        ]
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=results,
            included_classes={0: "cat"},
            max_images=2,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        # Grid is 2×1 (128 wide, 64 tall): image 0 at x=0...63, image 1 at x=64...127.
        _assert_bbox_corners_have_color(
            image=result, xyxy=(8, 8, 56, 56), color=utils._get_class_color(0)
        )
        assert result.getpixel((60, 60)) == _BACKGROUND_PIXEL
        assert result.getpixel((32, 32)) == _BACKGROUND_PIXEL
        assert result.getpixel((64, 0)) == _BACKGROUND_PIXEL

    @pytest.mark.parametrize("score,drawn", [(0.9, True), (0.3, False)])
    def test_plot_object_detection_predictions_score_threshold(
        self, score: float, drawn: bool
    ) -> None:
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 128, 128), _BACKGROUND_COLOR)
        )
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=[
                {
                    "boxes": torch.tensor([[0.0, 0.0, 64.0, 64.0]]),
                    "labels": torch.zeros(1, dtype=torch.long),
                    "scores": torch.tensor([score]),
                }
            ],
            included_classes={0: "cat"},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        # When the score is above threshold, the four bbox corners must be
        # painted with class 0's color; when below, the box is filtered out and
        # all four corner pixels stay black.
        expected = utils._get_class_color(0) if drawn else _BACKGROUND_PIXEL
        _assert_bbox_corners_have_color(
            image=result, xyxy=(0, 0, 64, 64), color=expected
        )
        # The box interior is never filled — only the outline is drawn.
        assert result.getpixel((32, 32)) == _BACKGROUND_PIXEL

    def test_plot_object_detection_predictions_max_pred_boxes_limits_drawn_boxes(
        self,
    ) -> None:
        # Each box gets a distinct class (and therefore a distinct color), and
        # scores are scrambled so the kept set is determined by score rank rather
        # than insertion order. Scores: top-3 are box 1 (0.95), box 3 (0.85),
        # box 4 (0.75). Boxes 0 (0.60) and 2 (0.55) are dropped, even though box
        # 0 comes first and box 2 sits between two kept boxes.
        # All scores are > score_threshold=0.5, so the only filter is max_pred_boxes.
        # Boxes are 24 px tall so the label rectangle (~19 px tall with the default
        # font) does not reach the bottom corners. Boxes are spaced 120 px apart so
        # each kept box's label (~80 px wide) cannot reach the next box's region.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 32, 600), _BACKGROUND_COLOR)
        )
        boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 24.0],  # class 0, score 0.60 -> suppressed
                [120.0, 0.0, 130.0, 24.0],  # class 1, score 0.95 -> kept
                [240.0, 0.0, 250.0, 24.0],  # class 2, score 0.55 -> suppressed
                [360.0, 0.0, 370.0, 24.0],  # class 3, score 0.85 -> kept
                [480.0, 0.0, 490.0, 24.0],  # class 4, score 0.75 -> kept
            ]
        )
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=[
                {
                    "boxes": boxes,
                    "labels": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
                    "scores": torch.tensor([0.60, 0.95, 0.55, 0.85, 0.75]),
                }
            ],
            included_classes={i: f"c{i}" for i in range(5)},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=3,
        )
        # Suppressed boxes leave all four corners black.
        _assert_bbox_corners_have_color(
            image=result, xyxy=(0, 0, 10, 24), color=_BACKGROUND_PIXEL
        )
        _assert_bbox_corners_have_color(
            image=result, xyxy=(240, 0, 250, 24), color=_BACKGROUND_PIXEL
        )
        # Kept boxes paint all four corners with their own class color, so a
        # color regression (e.g. all boxes drawn in class 0's color) fails here.
        _assert_bbox_corners_have_color(
            image=result, xyxy=(120, 0, 130, 24), color=utils._get_class_color(1)
        )
        _assert_bbox_corners_have_color(
            image=result, xyxy=(360, 0, 370, 24), color=utils._get_class_color(3)
        )
        _assert_bbox_corners_have_color(
            image=result, xyxy=(480, 0, 490, 24), color=utils._get_class_color(4)
        )

    def test_plot_object_detection_predictions_unknown_class_draws_box(self) -> None:
        # Check that a box is drawn even when the class ID isn't in included_classes;
        # the label shows the numeric class ID.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 128, 128), _BACKGROUND_COLOR)
        )
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=[
                {
                    "boxes": torch.tensor([[0.0, 0.0, 64.0, 64.0]]),
                    "labels": torch.tensor([42], dtype=torch.long),
                    "scores": torch.tensor([0.9]),
                }
            ],
            included_classes={},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        # Class 42 is not in `included_classes` but still gets its deterministic
        # color from `_get_class_color` (label shows "Class 42" in white).
        _assert_bbox_corners_have_color(
            image=result, xyxy=(0, 0, 64, 64), color=utils._get_class_color(42)
        )
        assert result.getpixel((32, 32)) == _BACKGROUND_PIXEL

    @pytest.mark.parametrize(
        "image_value, mean, std, expected_pixel",
        [
            # image=0 -> denormalized = mean.
            (0.0, (0.2, 0.4, 0.6), (0.5, 0.5, 0.5), (51, 102, 153)),
            # Non-zero image: pixel = image * std + mean = 0.7 -> 178.
            (0.4, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (178, 178, 178)),
            # Per-channel std: 0.5 * (0.2, 0.4, 0.6) = (0.1, 0.2, 0.3) -> (25, 51, 76).
            (0.5, (0.0, 0.0, 0.0), (0.2, 0.4, 0.6), (25, 51, 76)),
            # Clamping above 1 -> 255 and below 0 -> 0.
            (5.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (255, 255, 255)),
            (-5.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0, 0, 0)),
        ],
    )
    def test_plot_object_detection_predictions_mean_std_denormalizes_image(
        self,
        image_value: float,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        expected_pixel: tuple[int, int, int],
    ) -> None:
        batch = _make_batch_from_image(image=torch.full((1, 3, 32, 32), image_value))
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=_make_empty_results(batch_size=1),
            included_classes={},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
            mean=mean,
            std=std,
        )
        # No predicted boxes pass the threshold, so every pixel reflects the
        # denormalized image.
        assert result.getpixel((0, 0)) == expected_pixel
        assert result.getpixel((31, 31)) == expected_pixel

    def test_plot_object_detection_predictions_no_mean_std_skips_denormalization(
        self,
    ) -> None:
        # Without mean/std, the image tensor is passed through unchanged.
        # Uniform 0.4 -> 102.
        batch = _make_batch_from_image(image=torch.full((1, 3, 32, 32), 0.4))
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=_make_empty_results(batch_size=1),
            included_classes={},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.getpixel((0, 0)) == (102, 102, 102)
        assert result.getpixel((31, 31)) == (102, 102, 102)

    def test_plot_object_detection_predictions_bbox_scaling_uniform(self) -> None:
        # Original image is 128×128; tensor is 64×64 (uniform 2× downscale).
        # Box at [8, 8, 120, 120] in original coords maps to [4, 4, 60, 60] in tensor.
        batch = ObjectDetectionBatch(
            image_path=["img_0.jpg"],
            image=torch.full((1, 3, 64, 64), _BACKGROUND_COLOR),
            bboxes=[torch.zeros(0, 4)],
            classes=[torch.zeros(0, dtype=torch.long)],
            original_size=[(128, 128)],
        )
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=[
                {
                    "boxes": torch.tensor([[8.0, 8.0, 120.0, 120.0]]),
                    "labels": torch.zeros(1, dtype=torch.long),
                    "scores": torch.tensor([0.9]),
                }
            ],
            included_classes={0: "obj"},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.size == (64, 64)
        _assert_bbox_corners_have_color(
            image=result, xyxy=(4, 4, 60, 60), color=utils._get_class_color(0)
        )
        assert result.getpixel((32, 32)) == _BACKGROUND_PIXEL  # box interior not filled
        assert result.getpixel((0, 0)) == _BACKGROUND_PIXEL  # outside the scaled box

    def test_plot_object_detection_predictions_bbox_scaling_asymmetric(self) -> None:
        # Original image is 128 wide × 64 tall; tensor is 64×64.
        # x-coords are halved (scale 0.5), y-coords are unchanged (scale 1.0).
        # Box at [8, 8, 120, 48] in original -> [4, 8, 60, 48] in tensor.
        batch = ObjectDetectionBatch(
            image_path=["img_0.jpg"],
            image=torch.full((1, 3, 64, 64), _BACKGROUND_COLOR),
            bboxes=[torch.zeros(0, 4)],
            classes=[torch.zeros(0, dtype=torch.long)],
            original_size=[(128, 64)],
        )
        result = object_detection.plot_object_detection_predictions(
            batch=batch,
            results=[
                {
                    "boxes": torch.tensor([[8.0, 8.0, 120.0, 48.0]]),
                    "labels": torch.zeros(1, dtype=torch.long),
                    "scores": torch.tensor([0.9]),
                }
            ],
            included_classes={0: "obj"},
            max_images=1,
            score_threshold=0.5,
            max_pred_boxes=10,
        )
        assert result.size == (64, 64)
        _assert_bbox_corners_have_color(
            image=result, xyxy=(4, 8, 60, 48), color=utils._get_class_color(0)
        )
        assert result.getpixel((32, 38)) == _BACKGROUND_PIXEL  # box interior not filled
        assert result.getpixel((0, 0)) == _BACKGROUND_PIXEL  # outside the scaled box
