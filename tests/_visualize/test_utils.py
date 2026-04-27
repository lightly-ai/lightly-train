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
from PIL import Image
from PIL.ImageDraw import ImageDraw

from lightly_train._visualize import utils


class TestCxcywhToXyxy:
    def test__cxcywh_to_xyxy__center_box(self) -> None:
        # cx=0.5, cy=0.5, bw=0.5, bh=0.5 on a 100x80 image
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        result = utils._cxcywh_to_xyxy(boxes=boxes, w=100, h=80)
        expected = torch.tensor([[25.0, 20.0, 75.0, 60.0]])
        assert torch.allclose(result, expected)

    def test__cxcywh_to_xyxy__full_image_box(self) -> None:
        # Box spanning the full image.
        boxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        result = utils._cxcywh_to_xyxy(boxes=boxes, w=200, h=100)
        expected = torch.tensor([[0.0, 0.0, 200.0, 100.0]])
        assert torch.allclose(result, expected)

    def test__cxcywh_to_xyxy__multiple_boxes(self) -> None:
        boxes = torch.tensor(
            [
                [0.25, 0.25, 0.5, 0.5],
                [0.75, 0.75, 0.5, 0.5],
            ]
        )
        result = utils._cxcywh_to_xyxy(boxes=boxes, w=100, h=100)
        expected = torch.tensor(
            [
                [0.0, 0.0, 50.0, 50.0],
                [50.0, 50.0, 100.0, 100.0],
            ]
        )
        assert torch.allclose(result, expected)

    def test__cxcywh_to_xyxy__non_square_image(self) -> None:
        # Verify that w and h scale different axes independently.
        boxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        result = utils._cxcywh_to_xyxy(boxes=boxes, w=400, h=200)
        assert result[0, 0].item() == pytest.approx(0.0)
        assert result[0, 1].item() == pytest.approx(0.0)
        assert result[0, 2].item() == pytest.approx(400.0)
        assert result[0, 3].item() == pytest.approx(200.0)

    def test__cxcywh_to_xyxy__does_not_modify_input(self) -> None:
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        original = boxes.clone()
        utils._cxcywh_to_xyxy(boxes=boxes, w=100, h=100)
        assert torch.equal(boxes, original)

    def test__cxcywh_to_xyxy__output_shape(self) -> None:
        boxes = torch.rand(7, 4)
        result = utils._cxcywh_to_xyxy(boxes=boxes, w=64, h=64)
        assert result.shape == (7, 4)

    def test__cxcywh_to_xyxy__x1_less_than_x2_and_y1_less_than_y2(self) -> None:
        boxes = torch.rand(10, 4)
        result = utils._cxcywh_to_xyxy(boxes=boxes, w=100, h=100)
        assert (result[:, 0] <= result[:, 2]).all()
        assert (result[:, 1] <= result[:, 3]).all()


class TestRenderGrid:
    def test__render_grid__empty_list(self) -> None:
        result = utils._render_grid([])
        assert result.size == (1, 1)

    def test__render_grid__single_image(self) -> None:
        img = Image.new("RGB", (50, 40), color=(255, 0, 0))
        result = utils._render_grid([img])
        assert result.size == (50, 40)

    def test__render_grid__four_images_form_2x2(self) -> None:
        images = [Image.new("RGB", (10, 10)) for _ in range(4)]
        result = utils._render_grid(images)
        # ceil(sqrt(4))=2 cols, ceil(4/2)=2 rows
        assert result.size == (20, 20)

    def test__render_grid__nine_images_form_3x3(self) -> None:
        images = [Image.new("RGB", (8, 6)) for _ in range(9)]
        result = utils._render_grid(images)
        assert result.size == (24, 18)

    def test__render_grid__five_images_grid_dimensions(self) -> None:
        images = [Image.new("RGB", (10, 10)) for _ in range(5)]
        n_cols = math.ceil(math.sqrt(5))  # 3
        n_rows = math.ceil(5 / n_cols)  # 2
        result = utils._render_grid(images)
        assert result.size == (n_cols * 10, n_rows * 10)

    def test__render_grid__pixels_placed_correctly(self) -> None:
        # Assign distinct solid colors to four images and verify each
        # occupies the expected quadrant in the 2x2 grid.
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0)]
        images = [Image.new("RGB", (10, 10), color=c) for c in colors]
        result = utils._render_grid(images)

        # Grid layout (2 cols): idx 0→(col=0,row=0), 1→(col=1,row=0),
        #                        2→(col=0,row=1), 3→(col=1,row=1)
        assert result.getpixel((5, 5)) == colors[0]  # top-left
        assert result.getpixel((15, 5)) == colors[1]  # top-right
        assert result.getpixel((5, 15)) == colors[2]  # bottom-left
        assert result.getpixel((15, 15)) == colors[3]  # bottom-right

    def test__render_grid__returns_rgb_image(self) -> None:
        images = [Image.new("RGB", (4, 4)) for _ in range(2)]
        result = utils._render_grid(images)
        assert result.mode == "RGB"


class Test_draw_bbox_label:
    def test__draw_bbox_label_label_above_when_space(self) -> None:
        image = Image.new("RGB", (200, 200), color=(255, 255, 255))
        draw = ImageDraw(image)
        color = (255, 0, 0)
        x1, y1 = 10, 100  # plenty of space above

        utils._draw_bbox_label(draw=draw, x1=x1, y1=y1, text="dog", color=color)

        # Label is drawn above y1: pixel just inside the top-left of the label rect is colored.
        px = image.getpixel((x1 + 1, y1 - 1))
        assert isinstance(px, tuple)
        assert px[:3] == color

    def test__draw_bbox_label_label_below_when_no_space(self) -> None:
        image = Image.new("RGB", (200, 200), color=(255, 255, 255))
        draw = ImageDraw(image)
        color = (0, 255, 0)
        x1, y1 = 10, 2  # y1 too small to draw above

        utils._draw_bbox_label(draw=draw, x1=x1, y1=y1, text="cat", color=color)

        # Label is drawn below y1: pixel just inside the top-left of the label rect is colored.
        px = image.getpixel((x1 + 1, y1 + 1))
        assert isinstance(px, tuple)
        assert px[:3] == color

    def test__draw_bbox_label_at_boundary(self) -> None:
        image = Image.new("RGB", (200, 200), color=(255, 255, 255))
        draw = ImageDraw(image)
        color = (0, 0, 255)
        text = "bird"

        # Measure label height so we can set y1 exactly at the boundary.
        bbox = draw.textbbox((0, 0), text)
        label_height = (
            int(bbox[3] - bbox[1]) + 8
        )  # 2 * padding (4); int() handles float textbbox

        x1, y1 = 10, label_height  # y1 == label_height → condition is >=, so draw above

        utils._draw_bbox_label(draw=draw, x1=x1, y1=y1, text=text, color=color)

        # Should draw above: pixel just above y1 is within the label rect and colored.
        px_above = image.getpixel((x1 + 1, y1 - 1))
        assert isinstance(px_above, tuple)
        assert px_above[:3] == color


class Test_denormalize_image:
    def test__denormalize_image_basic_math(self) -> None:
        mean = (0.5, 0.5, 0.5)
        std = (0.2, 0.2, 0.2)
        image = torch.zeros(3, 2, 2)

        result = utils._denormalize_image(image=image, mean=mean, std=std)

        # x_denorm = 0 * 0.2 + 0.5 = 0.5
        assert torch.allclose(result, torch.full((3, 2, 2), 0.5))

    def test__denormalize_image_known_values(self) -> None:
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
        image = torch.tensor([[[0.3]], [[0.6]], [[0.9]]])

        result = utils._denormalize_image(image=image, mean=mean, std=std)

        assert torch.allclose(result, image)

    def test__denormalize_image_preserves_shape(self) -> None:
        image = torch.rand(3, 64, 128)
        result = utils._denormalize_image(
            image=image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        assert result.shape == (3, 64, 128)

    def test__denormalize_image_clamps_to_zero_one(self) -> None:
        # x_denorm = 2.0 * 0.5 + 0.5 = 1.5 → clamped to 1.0
        # x_denorm = -2.0 * 0.5 + 0.5 = -0.5 → clamped to 0.0
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        image = torch.tensor([[[2.0]], [[-2.0]], [[0.0]]])

        result = utils._denormalize_image(image=image, mean=mean, std=std)

        assert result[0, 0, 0].item() == pytest.approx(1.0)
        assert result[1, 0, 0].item() == pytest.approx(0.0)
        assert result[2, 0, 0].item() == pytest.approx(0.5)

    def test__denormalize_image_per_channel_mean_and_std(self) -> None:
        mean = (0.1, 0.2, 0.3)
        std = (0.5, 0.5, 0.5)
        image = torch.zeros(3, 1, 1)

        result = utils._denormalize_image(image=image, mean=mean, std=std)

        assert result[0, 0, 0].item() == pytest.approx(0.1)
        assert result[1, 0, 0].item() == pytest.approx(0.2)
        assert result[2, 0, 0].item() == pytest.approx(0.3)

    def test__denormalize_image_preserves_dtype(self) -> None:
        image = torch.rand(3, 4, 4, dtype=torch.float64)
        result = utils._denormalize_image(
            image=image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )
        assert result.dtype == torch.float64


class Test_get_class_color:
    def test__get_class_color_returns_rgb_tuple(self) -> None:
        color = utils._get_class_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3

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
