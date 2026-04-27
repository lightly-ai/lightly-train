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

from lightly_train._visualize import object_detection


class TestCxcywhToXyxy:
    def test__cxcywh_to_xyxy__center_box(self) -> None:
        # cx=0.5, cy=0.5, bw=0.5, bh=0.5 on a 100x80 image
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        result = object_detection._cxcywh_to_xyxy(boxes=boxes, w=100, h=80)
        expected = torch.tensor([[25.0, 20.0, 75.0, 60.0]])
        assert torch.allclose(result, expected)

    def test__cxcywh_to_xyxy__full_image_box(self) -> None:
        # Box spanning the full image.
        boxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        result = object_detection._cxcywh_to_xyxy(boxes=boxes, w=200, h=100)
        expected = torch.tensor([[0.0, 0.0, 200.0, 100.0]])
        assert torch.allclose(result, expected)

    def test__cxcywh_to_xyxy__multiple_boxes(self) -> None:
        boxes = torch.tensor(
            [
                [0.25, 0.25, 0.5, 0.5],
                [0.75, 0.75, 0.5, 0.5],
            ]
        )
        result = object_detection._cxcywh_to_xyxy(boxes=boxes, w=100, h=100)
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
        result = object_detection._cxcywh_to_xyxy(boxes=boxes, w=400, h=200)
        assert result[0, 0].item() == pytest.approx(0.0)
        assert result[0, 1].item() == pytest.approx(0.0)
        assert result[0, 2].item() == pytest.approx(400.0)
        assert result[0, 3].item() == pytest.approx(200.0)

    def test__cxcywh_to_xyxy__does_not_modify_input(self) -> None:
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        original = boxes.clone()
        object_detection._cxcywh_to_xyxy(boxes=boxes, w=100, h=100)
        assert torch.equal(boxes, original)

    def test__cxcywh_to_xyxy__output_shape(self) -> None:
        boxes = torch.rand(7, 4)
        result = object_detection._cxcywh_to_xyxy(boxes=boxes, w=64, h=64)
        assert result.shape == (7, 4)

    def test__cxcywh_to_xyxy__x1_less_than_x2_and_y1_less_than_y2(self) -> None:
        boxes = torch.rand(10, 4)
        result = object_detection._cxcywh_to_xyxy(boxes=boxes, w=100, h=100)
        assert (result[:, 0] <= result[:, 2]).all()
        assert (result[:, 1] <= result[:, 3]).all()


class TestRenderGrid:
    def test__render_grid__empty_list(self) -> None:
        result = object_detection._render_grid([])
        assert result.size == (1, 1)

    def test__render_grid__single_image(self) -> None:
        img = Image.new("RGB", (50, 40), color=(255, 0, 0))
        result = object_detection._render_grid([img])
        assert result.size == (50, 40)

    def test__render_grid__four_images_form_2x2(self) -> None:
        images = [Image.new("RGB", (10, 10)) for _ in range(4)]
        result = object_detection._render_grid(images)
        # ceil(sqrt(4))=2 cols, ceil(4/2)=2 rows
        assert result.size == (20, 20)

    def test__render_grid__nine_images_form_3x3(self) -> None:
        images = [Image.new("RGB", (8, 6)) for _ in range(9)]
        result = object_detection._render_grid(images)
        assert result.size == (24, 18)

    def test__render_grid__five_images_grid_dimensions(self) -> None:
        images = [Image.new("RGB", (10, 10)) for _ in range(5)]
        n_cols = math.ceil(math.sqrt(5))  # 3
        n_rows = math.ceil(5 / n_cols)  # 2
        result = object_detection._render_grid(images)
        assert result.size == (n_cols * 10, n_rows * 10)

    def test__render_grid__pixels_placed_correctly(self) -> None:
        # Assign distinct solid colors to four images and verify each
        # occupies the expected quadrant in the 2x2 grid.
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0)]
        images = [Image.new("RGB", (10, 10), color=c) for c in colors]
        result = object_detection._render_grid(images)

        # Grid layout (2 cols): idx 0→(col=0,row=0), 1→(col=1,row=0),
        #                        2→(col=0,row=1), 3→(col=1,row=1)
        assert result.getpixel((5, 5)) == colors[0]  # top-left
        assert result.getpixel((15, 5)) == colors[1]  # top-right
        assert result.getpixel((5, 15)) == colors[2]  # bottom-left
        assert result.getpixel((15, 15)) == colors[3]  # bottom-right

    def test__render_grid__returns_rgb_image(self) -> None:
        images = [Image.new("RGB", (4, 4)) for _ in range(2)]
        result = object_detection._render_grid(images)
        assert result.mode == "RGB"
