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
from PIL import Image
from PIL.ImageDraw import ImageDraw

from lightly_train._visualize import utils


class Test_draw_bbox_label:
    def test__draw_bbox_label_label_above_when_space(self) -> None:
        image = Image.new("RGB", (200, 200), color=(255, 255, 255))
        draw = ImageDraw(image)
        color = (255, 0, 0)
        x1, y1 = 10, 100  # plenty of space above

        utils._draw_bbox_label(
            draw=draw, x1=x1, y1=y1, text="dog", color=color
        )

        # Label is drawn above y1: pixel just inside the top-left of the label rect is colored.
        px = image.getpixel((x1 + 1, y1 - 1))
        assert isinstance(px, tuple)
        assert px[:3] == color

    def test__draw_bbox_label_label_below_when_no_space(self) -> None:
        image = Image.new("RGB", (200, 200), color=(255, 255, 255))
        draw = ImageDraw(image)
        color = (0, 255, 0)
        x1, y1 = 10, 2  # y1 too small to draw above

        utils._draw_bbox_label(
            draw=draw, x1=x1, y1=y1, text="cat", color=color
        )

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

        utils._draw_bbox_label(
            draw=draw, x1=x1, y1=y1, text=text, color=color
        )

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