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
from PIL import Image, ImageChops
from PIL.ImageDraw import ImageDraw

from lightly_train._visualize import utils

_BACKGROUND_PIXEL: tuple[int, int, int] = (0, 0, 0)
_WHITE_PIXEL: tuple[int, int, int] = (255, 255, 255)
# These come from the deterministic golden-ratio palette in utils._get_class_color.
_CLASS_0_COLOR: tuple[int, int, int] = (242, 24, 24)
_CLASS_1_COLOR: tuple[int, int, int] = (24, 87, 242)


def test__cxcywh_to_xyxy() -> None:
    # Two boxes at once on a non-square image: a centered partial box and a
    # full-image box. Catches per-row math, x/y scaling and width/height halving.
    boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]])
    expected = torch.tensor([[50.0, 25.0, 150.0, 75.0], [0.0, 0.0, 200.0, 100.0]])
    result = utils._cxcywh_to_xyxy(boxes=boxes, w=200, h=100)
    assert torch.allclose(result, expected)


def test__render_grid__centers_heterogeneous_tiles_in_max_size_cells() -> None:
    # Four distinct-color tiles of different sizes. Cell size should be
    # max(w), max(h) = (10, 10); grid is 2x2 so the result is (20, 20). Each
    # tile is centered in its cell, so cell-center pixels match the tile color
    # and cell corners outside the centered tile remain background.
    red, green, blue, yellow = (255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0)
    images = [
        Image.new("RGB", (10, 6), color=red),
        Image.new("RGB", (4, 8), color=green),
        Image.new("RGB", (6, 10), color=blue),
        Image.new("RGB", (8, 4), color=yellow),
    ]
    result = utils._render_grid(images)
    assert result.size == (20, 20)
    assert result.getpixel((5, 5)) == red
    assert result.getpixel((15, 5)) == green
    assert result.getpixel((5, 15)) == blue
    assert result.getpixel((15, 15)) == yellow
    # Red tile is 10x6 in a 10x10 cell: top row is padding.
    assert result.getpixel((0, 0)) == _BACKGROUND_PIXEL
    # Green tile is 4x8 in a 10x10 cell at top-right: left column is padding.
    assert result.getpixel((10, 5)) == _BACKGROUND_PIXEL


def test__draw_bbox_label__draws_above_when_space() -> None:
    image = Image.new("RGB", (200, 200), color=_BACKGROUND_PIXEL)
    draw = ImageDraw(image)
    color = (255, 0, 0)
    utils._draw_bbox_label(draw=draw, x1=10, y1=100, text="dog", color=color)
    assert image.getpixel((11, 99)) == color
    assert image.getpixel((150, 150)) == _BACKGROUND_PIXEL


def test__draw_bbox_label__draws_below_when_no_space() -> None:
    image = Image.new("RGB", (200, 200), color=_BACKGROUND_PIXEL)
    draw = ImageDraw(image)
    color = (0, 255, 0)
    utils._draw_bbox_label(draw=draw, x1=10, y1=2, text="cat", color=color)
    assert image.getpixel((11, 3)) == color
    assert image.getpixel((150, 150)) == _BACKGROUND_PIXEL


def test__denormalize_image__per_channel_math() -> None:
    image = torch.tensor([[[0.0]], [[0.5]], [[1.0]]])
    result = utils._denormalize_image(
        image=image, mean=(0.1, 0.2, 0.3), std=(0.5, 0.4, 0.3)
    )
    # x * std + mean per channel.
    assert result[0, 0, 0].item() == pytest.approx(0.1)
    assert result[1, 0, 0].item() == pytest.approx(0.4)
    assert result[2, 0, 0].item() == pytest.approx(0.6)


def test__denormalize_image__clamps_to_zero_one() -> None:
    image = torch.tensor([[[3.0]], [[-2.0]], [[0.5]]])
    result = utils._denormalize_image(
        image=image, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
    )
    assert result[0, 0, 0].item() == pytest.approx(1.0)
    assert result[1, 0, 0].item() == pytest.approx(0.0)
    assert result[2, 0, 0].item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    "class_id, expected",
    [
        (0, (242, 24, 24)),
        (1, (24, 87, 242)),
        (42, (242, 24, 79)),
        (99, (217, 242, 24)),
    ],
)
def test__get_class_color__exact_rgb(
    class_id: int, expected: tuple[int, int, int]
) -> None:
    assert utils._get_class_color(class_id) == expected


def test__draw_class_legend__empty_labels_returns_input_unchanged() -> None:
    image = Image.new("RGB", (64, 64), color=(0, 200, 100))
    result = utils._draw_class_legend(image=image, labels=[], colors=None)
    assert ImageChops.difference(result, image).getbbox() is None


def test__draw_class_legend__with_colors_renders_patch_color() -> None:
    # A red patch must produce red-dominant pixels somewhere in the upper-left
    # legend area.
    image = Image.new("RGB", (256, 256), color=_WHITE_PIXEL)
    result = utils._draw_class_legend(image=image, labels=["cat"], colors=[(255, 0, 0)])
    pixels = result.load()
    assert pixels is not None
    has_red = any(
        isinstance(pixel := pixels[x, y], tuple)
        and pixel[0] > 200
        and pixel[1] < 80
        and pixel[2] < 80
        for x in range(128)
        for y in range(128)
    )
    assert has_red


def test__draw_class_legend__mismatched_colors_and_labels_raises() -> None:
    image = Image.new("RGB", (64, 64), color=_WHITE_PIXEL)
    with pytest.raises(ValueError, match="must have the same length"):
        utils._draw_class_legend(image=image, labels=["a", "b"], colors=[(255, 0, 0)])


def test__build_semantic_mask_overlay__colors_known_classes_skips_unknown_and_resizes() -> (
    None
):
    # Top half is class 0 (in class_names -> colored). Bottom half is class 5
    # (not in class_names -> stays black). Output size differs from mask, so
    # nearest-neighbor resize is exercised.
    mask = torch.zeros(4, 4, dtype=torch.long)
    mask[2:, :] = 5
    result = utils._build_semantic_mask_overlay(
        mask=mask, size=(8, 8), class_names={0: "a"}
    )
    assert result.mode == "RGB"
    assert result.size == (8, 8)
    assert result.getpixel((0, 0)) == _CLASS_0_COLOR
    assert result.getpixel((0, 7)) == _BACKGROUND_PIXEL


def test__bboxes_from_masks__tight_bounds_and_skips_empty() -> None:
    # Three instances on an 8x8 canvas:
    #   0: rows 1-3, cols 2-5 -> tight box [2, 1, 5, 3].
    #   1: empty (no foreground pixels) -> skipped.
    #   2: a single pixel at (col=6, row=7) -> degenerate box [6, 7, 6, 7].
    # Note: bboxes use inclusive max coords (xs.max() / ys.max(), no +1).
    masks = torch.zeros(3, 8, 8, dtype=torch.bool)
    masks[0, 1:4, 2:6] = True
    masks[2, 7, 6] = True
    boxes, keep = utils._bboxes_from_masks(masks=masks)
    # The keep tensor preserves the original instance order so callers can use
    # it to filter parallel arrays (labels, scores) — index 1 must be False.
    assert keep.tolist() == [True, False, True]
    assert boxes.shape == (2, 4)
    assert boxes[0].tolist() == [2.0, 1.0, 5.0, 3.0]
    assert boxes[1].tolist() == [6.0, 7.0, 6.0, 7.0]


def test__draw_labeled_boxes__draws_outline_and_label_in_class_color() -> None:
    # Single 64x64 box at (32, 32) on a 128x128 canvas. y1=32 leaves room for
    # the label above the box, so the label rectangle is drawn above and is
    # filled with the class color.
    image = Image.new("RGB", (128, 128), color=_BACKGROUND_PIXEL)
    bboxes = torch.tensor([[32.0, 32.0, 96.0, 96.0]])
    labels = torch.tensor([1], dtype=torch.long)
    utils._draw_labeled_boxes(
        image=image,
        bboxes_xyxy=bboxes,
        labels=labels,
        scores=None,
        class_names={1: "dog"},
    )
    # Each of the four corners is painted in class 1's color — catches a
    # regression where the bbox is drawn in the wrong color (e.g. a hard-coded
    # color or class 0's color regardless of label).
    assert image.getpixel((32, 32)) == _CLASS_1_COLOR
    assert image.getpixel((96, 32)) == _CLASS_1_COLOR
    assert image.getpixel((32, 96)) == _CLASS_1_COLOR
    assert image.getpixel((96, 96)) == _CLASS_1_COLOR
    # Box interior remains background (only the outline is drawn).
    assert image.getpixel((64, 64)) == _BACKGROUND_PIXEL
    # The label rectangle sits above the box and is filled with class 1's
    # color, so a pixel just above the box at x=34 lands inside the label.
    assert image.getpixel((34, 30)) == _CLASS_1_COLOR


def test__draw_mask_contours__paints_boundary_keeps_interior() -> None:
    # Two stacked regions split between rows 3 and 4. Boundary pixels on both
    # sides of the split should be black; interior pixels of each region are
    # untouched.
    mask = torch.zeros(8, 8, dtype=torch.long)
    mask[4:, :] = 1
    image = Image.new("RGB", (8, 8), color=_WHITE_PIXEL)
    result = utils._draw_mask_contours(image=image, mask=mask)
    assert result.getpixel((4, 3)) == _BACKGROUND_PIXEL
    assert result.getpixel((4, 4)) == _BACKGROUND_PIXEL
    assert result.getpixel((4, 1)) == _WHITE_PIXEL
    assert result.getpixel((4, 6)) == _WHITE_PIXEL


def test__legend_entries_for_mask__returns_sorted_filtered_entries() -> None:
    # Mask contains classes 0, 1 and 5. Class 5 is not in class_names -> skipped.
    # Class 1 appears before class 0 in the mask, so output must be sorted by id.
    mask = torch.tensor([[1, 5], [0, 1]], dtype=torch.long)
    labels, colors = utils._legend_entries_for_mask(
        mask=mask, class_names={0: "cat", 1: "dog"}
    )
    assert labels == ["cat", "dog"]
    assert colors == [_CLASS_0_COLOR, _CLASS_1_COLOR]
