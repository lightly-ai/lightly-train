#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import colorsys
import math

import torch
from PIL import Image, ImageFont
from PIL.Image import Image as PILImage
from PIL.ImageDraw import ImageDraw as PILDraw
from torch import Tensor

try:
    _DEFAULT_FONT = ImageFont.load_default(size=20)
except TypeError:
    _DEFAULT_FONT = ImageFont.load_default()


def _draw_bbox_label(
    draw: PILDraw,
    x1: float,
    y1: float,
    text: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a highlighted label rectangle near a bounding box.

    Draws above the box when there is enough space; otherwise draws below.
    """
    padding = 4

    bbox = draw.textbbox((0, 0), text, font=_DEFAULT_FONT)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x1 = float(x1)
    y1 = float(y1)

    label_width = text_width + 2 * padding
    label_height = text_height + 2 * padding

    if y1 >= label_height:
        rect_top = y1 - label_height
        rect_bottom = y1
    else:
        rect_top = y1
        rect_bottom = y1 + label_height

    rect_left = x1
    rect_right = x1 + label_width

    x0, x1_rect = sorted((rect_left, rect_right))
    y0, y1_rect = sorted((rect_top, rect_bottom))

    draw.rectangle([x0, y0, x1_rect, y1_rect], fill=color, outline=color)
    draw.text((x0 + padding, y0 + padding), text, fill="white", font=_DEFAULT_FONT)


def _denormalize_image(
    image: Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Tensor:
    """Denormalize an image tensor using mean and std.

    Args:
        image: Tensor of shape (3, H, W) with normalized values.
        mean: Tuple of mean values used for normalization.
        std: Tuple of std values used for normalization.

    Returns:
        Denormalized tensor with values clamped to the [0, 1] range.
    """
    mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype).view(
        -1, 1, 1
    )
    std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype).view(
        -1, 1, 1
    )

    denormalized = image * std_tensor + mean_tensor
    denormalized = torch.clamp(denormalized, 0, 1)

    return denormalized


def _get_class_color(class_id: int) -> tuple[int, int, int]:
    """Generate a deterministic RGB color for a class ID.

    Uses HSV color space with varying hue to ensure the same class ID always gets
    the same color, with good visual distinction between different classes.
    This maintains color consistency throughout the training process.

    Args:
        class_id: The class ID to generate a color for.

    Returns:
        RGB tuple with values in range [0, 255].
    """
    # Use modulo to cycle through hue values with good distribution
    hue = (
        class_id * 0.618033988749895
    ) % 1.0  # Golden ratio for good color distribution

    # Use high saturation and value for vibrant, distinct colors
    saturation = 0.9
    value = 0.95

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def _render_grid(pil_images: list[PILImage]) -> PILImage:
    """Arrange PIL images into a square-ish grid.

    Args:
        pil_images: List of PIL images, all the same size.

    Returns:
        Single PIL image with all inputs tiled into a grid.
    """
    n = len(pil_images)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    w, h = pil_images[0].size
    mode = pil_images[0].mode
    grid = Image.new(mode, (n_cols * w, n_rows * h))
    for idx, img in enumerate(pil_images):
        row, col = divmod(idx, n_cols)
        grid.paste(img, (col * w, row * h))
    return grid


def _cxcywh_to_xyxy(boxes: Tensor, w: int, h: int) -> Tensor:
    """Convert bounding boxes from cxcywh format to xyxy format.

    Args:
        boxes: Tensor of shape (n_boxes, 4) in cxcywh format (center_x, center_y,
            width, height). Values are normalized to [0, 1].
        w: Width of the image.
        h: Height of the image.

    Returns:
        Tensor of shape (n_boxes, 4) in xyxy format (x1, y1, x2, y2).
    """
    boxes_xyxy = boxes.clone()
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    boxes_xyxy[:, 0] = (cx - bw / 2) * w
    boxes_xyxy[:, 1] = (cy - bh / 2) * h
    boxes_xyxy[:, 2] = (cx + bw / 2) * w
    boxes_xyxy[:, 3] = (cy + bh / 2) * h
    return boxes_xyxy
