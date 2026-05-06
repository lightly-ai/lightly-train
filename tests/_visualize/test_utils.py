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
from PIL import Image, ImageChops
from PIL.Image import Image as PILImage
from PIL.ImageDraw import ImageDraw

from lightly_train._visualize import utils

_BACKGROUND_PIXEL: tuple[int, int, int] = (0, 0, 0)
_WHITE_PIXEL: tuple[int, int, int] = (255, 255, 255)
# These come from the deterministic golden-ratio palette in utils._get_class_color.
_CLASS_0_COLOR: tuple[int, int, int] = (242, 24, 24)
_CLASS_1_COLOR: tuple[int, int, int] = (24, 87, 242)


def _non_white_bbox(image: PILImage) -> tuple[int, int, int, int] | None:
    """Bounding box of pixels that differ from pure white, or None if all white."""
    white = Image.new(image.mode, image.size, _WHITE_PIXEL)
    return ImageChops.difference(image, white).getbbox()


class TestCxcywhToXyxy:
    def test__cxcywh_to_xyxy_center_box(self) -> None:
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        result = utils._cxcywh_to_xyxy(boxes=boxes, w=100, h=80)
        assert torch.allclose(result, torch.tensor([[25.0, 20.0, 75.0, 60.0]]))

    def test__cxcywh_to_xyxy_full_image_box(self) -> None:
        boxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        result = utils._cxcywh_to_xyxy(boxes=boxes, w=200, h=100)
        assert torch.allclose(result, torch.tensor([[0.0, 0.0, 200.0, 100.0]]))

    def test__cxcywh_to_xyxy_multiple_boxes(self) -> None:
        boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.75, 0.75, 0.5, 0.5]])
        result = utils._cxcywh_to_xyxy(boxes=boxes, w=100, h=100)
        expected = torch.tensor([[0.0, 0.0, 50.0, 50.0], [50.0, 50.0, 100.0, 100.0]])
        assert torch.allclose(result, expected)

    def test__cxcywh_to_xyxy_non_square_image(self) -> None:
        boxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        result = utils._cxcywh_to_xyxy(boxes=boxes, w=400, h=200)
        assert torch.allclose(result, torch.tensor([[0.0, 0.0, 400.0, 200.0]]))

    def test__cxcywh_to_xyxy_does_not_modify_input(self) -> None:
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        original = boxes.clone()
        utils._cxcywh_to_xyxy(boxes=boxes, w=100, h=100)
        assert torch.equal(boxes, original)


class TestRenderGrid:
    def test__render_grid_single_image(self) -> None:
        img = Image.new("RGB", (50, 40), color=(255, 0, 0))
        assert utils._render_grid([img]).size == (50, 40)

    def test__render_grid_four_images_form_2x2(self) -> None:
        images = [Image.new("RGB", (10, 10), color=_BACKGROUND_PIXEL) for _ in range(4)]
        assert utils._render_grid(images).size == (20, 20)

    def test__render_grid_nine_images_form_3x3(self) -> None:
        images = [Image.new("RGB", (8, 6), color=_BACKGROUND_PIXEL) for _ in range(9)]
        assert utils._render_grid(images).size == (24, 18)

    def test__render_grid_five_images_grid_dimensions(self) -> None:
        images = [Image.new("RGB", (10, 10), color=_BACKGROUND_PIXEL) for _ in range(5)]
        n_cols = math.ceil(math.sqrt(5))
        n_rows = math.ceil(5 / n_cols)
        assert utils._render_grid(images).size == (n_cols * 10, n_rows * 10)

    def test__render_grid_pixels_placed_correctly(self) -> None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0)]
        images = [Image.new("RGB", (10, 10), color=c) for c in colors]
        result = utils._render_grid(images)
        assert result.getpixel((5, 5)) == colors[0]  # top-left
        assert result.getpixel((15, 5)) == colors[1]  # top-right
        assert result.getpixel((5, 15)) == colors[2]  # bottom-left
        assert result.getpixel((15, 15)) == colors[3]  # bottom-right


class TestDrawBboxLabel:
    def test__draw_bbox_label_label_above_when_space(self) -> None:
        image = Image.new("RGB", (200, 200), color=_BACKGROUND_PIXEL)
        draw = ImageDraw(image)
        color = (255, 0, 0)
        utils._draw_bbox_label(draw=draw, x1=10, y1=100, text="dog", color=color)
        assert image.getpixel((11, 99)) == color
        assert image.getpixel((150, 150)) == _BACKGROUND_PIXEL

    def test__draw_bbox_label_label_below_when_no_space(self) -> None:
        image = Image.new("RGB", (200, 200), color=_BACKGROUND_PIXEL)
        draw = ImageDraw(image)
        color = (0, 255, 0)
        utils._draw_bbox_label(draw=draw, x1=10, y1=2, text="cat", color=color)
        assert image.getpixel((11, 3)) == color
        assert image.getpixel((150, 150)) == _BACKGROUND_PIXEL

    def test__draw_bbox_label_at_boundary(self) -> None:
        image = Image.new("RGB", (200, 200), color=_BACKGROUND_PIXEL)
        draw = ImageDraw(image)
        color = (0, 0, 255)
        text = "bird"
        bbox = draw.textbbox((0, 0), text, font=utils._DEFAULT_FONT)
        label_height = int(bbox[3] - bbox[1]) + 8  # text height + 2 * padding(4)
        utils._draw_bbox_label(
            draw=draw, x1=10, y1=label_height, text=text, color=color
        )
        assert image.getpixel((11, label_height - 1)) == color
        assert image.getpixel((150, 150)) == _BACKGROUND_PIXEL


class TestDenormalizeImage:
    def test__denormalize_image_basic_math(self) -> None:
        image = torch.zeros(3, 2, 2)
        result = utils._denormalize_image(
            image=image, mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)
        )
        assert torch.allclose(result, torch.full((3, 2, 2), 0.5))

    def test__denormalize_image_identity(self) -> None:
        image = torch.tensor([[[0.3]], [[0.6]], [[0.9]]])
        result = utils._denormalize_image(
            image=image, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        )
        assert torch.allclose(result, image)

    def test__denormalize_image_clamps_to_zero_one(self) -> None:
        image = torch.tensor([[[2.0]], [[-2.0]], [[0.0]]])
        result = utils._denormalize_image(
            image=image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )
        assert result[0, 0, 0].item() == pytest.approx(1.0)
        assert result[1, 0, 0].item() == pytest.approx(0.0)
        assert result[2, 0, 0].item() == pytest.approx(0.5)

    def test__denormalize_image_per_channel(self) -> None:
        image = torch.zeros(3, 1, 1)
        result = utils._denormalize_image(
            image=image, mean=(0.1, 0.2, 0.3), std=(0.5, 0.5, 0.5)
        )
        assert result[0, 0, 0].item() == pytest.approx(0.1)
        assert result[1, 0, 0].item() == pytest.approx(0.2)
        assert result[2, 0, 0].item() == pytest.approx(0.3)

    def test__denormalize_image_preserves_dtype(self) -> None:
        image = torch.rand(3, 4, 4, dtype=torch.float64)
        result = utils._denormalize_image(
            image=image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )
        assert result.dtype == torch.float64


class TestGetClassColor:
    def test__get_class_color_values_in_valid_range(self) -> None:
        for class_id in range(20):
            r, g, b = utils._get_class_color(class_id)
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test__get_class_color_deterministic(self) -> None:
        for class_id in range(10):
            assert utils._get_class_color(class_id) == utils._get_class_color(class_id)

    def test__get_class_color_distinct_colors_for_nearby_ids(self) -> None:
        colors = [utils._get_class_color(i) for i in range(10)]
        assert len(set(colors)) == len(colors)

    def test__get_class_color_large_class_id(self) -> None:
        r, g, b = utils._get_class_color(10_000)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    @pytest.mark.parametrize(
        "class_id, expected",
        [
            (0, (242, 24, 24)),
            (1, (24, 87, 242)),
            (2, (151, 242, 24)),
            (3, (242, 24, 215)),
            (4, (24, 242, 205)),
            (42, (242, 24, 79)),
            (99, (217, 242, 24)),
        ],
    )
    def test__get_class_color_exact_rgb(
        self, class_id: int, expected: tuple[int, int, int]
    ) -> None:
        assert utils._get_class_color(class_id) == expected


class TestDrawClassLegend:
    def test__draw_class_legend_empty_labels_returns_input_unchanged(self) -> None:
        image = Image.new("RGB", (64, 64), color=(0, 200, 100))
        result = utils._draw_class_legend(image=image, labels=[], colors=None)
        # Empty labels short-circuits and returns the image untouched.
        assert ImageChops.difference(result, image).getbbox() is None

    def test__draw_class_legend_text_only_anchored_to_upper_left(self) -> None:
        # Text-only legend (colors=None) draws into the upper-left corner and
        # leaves the rest of the image untouched.
        image = Image.new("RGB", (256, 256), color=_WHITE_PIXEL)
        result = utils._draw_class_legend(image=image, labels=["cat"], colors=None)
        bbox = _non_white_bbox(result)
        assert bbox is not None
        # Legend stays in the upper-left half of the image.
        assert bbox[2] <= 128 and bbox[3] <= 128
        # Far corner is preserved.
        assert result.getpixel((255, 255)) == _WHITE_PIXEL

    def test__draw_class_legend_with_colors_renders_patch_color(self) -> None:
        # A red color patch must produce red-dominant pixels in the legend area.
        image = Image.new("RGB", (256, 256), color=_WHITE_PIXEL)
        result = utils._draw_class_legend(
            image=image, labels=["cat"], colors=[(255, 0, 0)]
        )
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

    def test__draw_class_legend_more_labels_extends_legend_downward(self) -> None:
        # Stacking a second label extends the legend further down the image.
        image = Image.new("RGB", (256, 256), color=_WHITE_PIXEL)
        result_one = utils._draw_class_legend(
            image=image, labels=["a"], colors=[(255, 0, 0)]
        )
        result_two = utils._draw_class_legend(
            image=image, labels=["a", "b"], colors=[(255, 0, 0), (0, 255, 0)]
        )
        bbox_one = _non_white_bbox(result_one)
        bbox_two = _non_white_bbox(result_two)
        assert bbox_one is not None and bbox_two is not None
        assert bbox_two[3] > bbox_one[3]

    def test__draw_class_legend_mismatched_colors_and_labels_raises(self) -> None:
        image = Image.new("RGB", (64, 64), color=_WHITE_PIXEL)
        with pytest.raises(ValueError, match="must have the same length"):
            utils._draw_class_legend(
                image=image, labels=["a", "b"], colors=[(255, 0, 0)]
            )


class TestBuildMaskOverlay:
    def test__build_mask_overlay__returns_rgb_image_of_requested_size(self) -> None:
        mask = torch.zeros(8, 8, dtype=torch.long)
        result = utils._build_mask_overlay(
            mask=mask, size=(16, 12), class_names={0: "a"}
        )
        assert result.mode == "RGB"
        assert result.size == (16, 12)

    def test__build_mask_overlay__class_pixels_get_expected_color(self) -> None:
        mask = torch.zeros(4, 4, dtype=torch.long)
        result = utils._build_mask_overlay(mask=mask, size=(4, 4), class_names={0: "a"})
        assert result.getpixel((0, 0)) == _CLASS_0_COLOR
        assert result.getpixel((3, 3)) == _CLASS_0_COLOR

    def test__build_mask_overlay__two_classes_get_distinct_colors(self) -> None:
        mask = torch.zeros(4, 4, dtype=torch.long)
        mask[2:, :] = 1
        result = utils._build_mask_overlay(
            mask=mask, size=(4, 4), class_names={0: "a", 1: "b"}
        )
        assert result.getpixel((0, 0)) == _CLASS_0_COLOR
        assert result.getpixel((0, 3)) == _CLASS_1_COLOR

    def test__build_mask_overlay__resizes_when_size_differs_from_mask_shape(
        self,
    ) -> None:
        mask = torch.zeros(4, 4, dtype=torch.long)
        result = utils._build_mask_overlay(mask=mask, size=(8, 8), class_names={0: "a"})
        assert result.size == (8, 8)

    def test__build_mask_overlay__ids_not_in_class_names_are_black(self) -> None:
        mask = torch.zeros(4, 4, dtype=torch.long)
        mask[2:, :] = 1
        result = utils._build_mask_overlay(mask=mask, size=(4, 4), class_names={0: "a"})
        assert result.getpixel((0, 0)) == _CLASS_0_COLOR
        assert result.getpixel((0, 3)) == _BACKGROUND_PIXEL


class TestDrawMaskContours:
    def test__draw_mask_contours__uniform_mask_has_no_contours(self) -> None:
        mask = torch.zeros(4, 4, dtype=torch.long)
        image = Image.new("RGB", (4, 4), color=_WHITE_PIXEL)
        result = utils._draw_mask_contours(image=image, mask=mask)
        assert result.getpixel((0, 0)) == _WHITE_PIXEL
        assert result.getpixel((3, 3)) == _WHITE_PIXEL

    def test__draw_mask_contours__boundary_pixels_are_black(self) -> None:
        mask = torch.zeros(8, 8, dtype=torch.long)
        mask[4:, :] = 1
        image = Image.new("RGB", (8, 8), color=_WHITE_PIXEL)
        result = utils._draw_mask_contours(image=image, mask=mask)
        assert result.getpixel((4, 3)) == _BACKGROUND_PIXEL
        assert result.getpixel((4, 4)) == _BACKGROUND_PIXEL
        # Pixels well inside each region are not touched.
        assert result.getpixel((4, 1)) == _WHITE_PIXEL
        assert result.getpixel((4, 6)) == _WHITE_PIXEL


class TestLegendEntriesForMask:
    def test__legend_entries_for_mask__returns_labels_and_colors(self) -> None:
        mask = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        labels, colors = utils._legend_entries_for_mask(
            mask=mask, class_names={0: "cat", 1: "dog"}
        )
        assert labels == ["cat", "dog"]
        assert colors == [_CLASS_0_COLOR, _CLASS_1_COLOR]

    def test__legend_entries_for_mask__skips_classes_not_in_class_names(self) -> None:
        mask = torch.tensor([[255]], dtype=torch.long)
        labels, colors = utils._legend_entries_for_mask(
            mask=mask, class_names={0: "cat"}
        )
        assert labels == []
        assert colors == []

    def test__legend_entries_for_mask__sorted_by_class_id(self) -> None:
        mask = torch.tensor([[1, 0]], dtype=torch.long)
        labels, _ = utils._legend_entries_for_mask(
            mask=mask, class_names={0: "a", 1: "b"}
        )
        assert labels == ["a", "b"]

    def test__legend_entries_for_mask__empty_when_no_matching_classes(self) -> None:
        mask = torch.zeros(4, 4, dtype=torch.long)
        labels, colors = utils._legend_entries_for_mask(mask=mask, class_names={})
        assert labels == []
        assert colors == []
