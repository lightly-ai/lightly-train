#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import torch
from PIL import Image as PILImageClass
from PIL import ImageDraw, ImageFont
from PIL.Image import Image as PILImage
from torch import Tensor


def draw_label(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw a highlighted label rectangle near a bounding box.

    The label is drawn above the box when there is enough space; otherwise it is
    drawn below. Coordinates are normalized so PIL always receives a valid
    rectangle.
    """
    padding = 4

    # Measure text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Normalize anchor coordinates
    x1 = float(x1)
    y1 = float(y1)

    label_width = text_width + 2 * padding
    label_height = text_height + 2 * padding

    # Prefer drawing above the box; if that would go out of bounds, draw below.
    if y1 >= label_height:
        rect_top = y1 - label_height
        rect_bottom = y1
    else:
        rect_top = y1
        rect_bottom = y1 + label_height

    rect_left = x1
    rect_right = x1 + label_width

    # Ensure valid ordering for PIL
    x0, x1_rect = sorted((rect_left, rect_right))
    y0, y1_rect = sorted((rect_top, rect_bottom))

    draw.rectangle([x0, y0, x1_rect, y1_rect], fill=color, outline=color)
    draw.text((x0 + padding, y0 + padding), text, fill="white", font=font)


def denormalize_image(
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
        Denormalized tensor with values in [0, 1] or [0, 255] range depending on input.
    """
    mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype).view(
        -1, 1, 1
    )
    std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype).view(
        -1, 1, 1
    )

    # Denormalize: x_denorm = x_norm * std + mean
    denormalized = image * std_tensor + mean_tensor

    # Clamp to valid range [0, 1]
    denormalized = torch.clamp(denormalized, 0, 1)

    return denormalized


def get_class_color(class_id: int) -> tuple[int, int, int]:
    """Generate a deterministic RGB color for a class ID.

    Uses HSV color space with varying hue to ensure the same class ID always gets
    the same color, with good visual distinction between different classes.
    This maintains color consistency throughout the training process.

    Args:
        class_id: The class ID to generate a color for.

    Returns:
        RGB tuple with values in range [0, 255].
    """
    # Generate hue based on class_id (0-1 range)
    # Use modulo to cycle through hue values with good distribution
    hue = (
        class_id * 0.618033988749895
    ) % 1.0  # Golden ratio for good color distribution

    # Use high saturation and value for vibrant, distinct colors
    saturation = 0.9
    value = 0.95

    # Convert HSV to RGB
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

    # Convert to 0-255 range
    return (int(r * 255), int(g * 255), int(b * 255))


def create_image_mosaic(images: list[PILImage], images_per_row: int = 2) -> PILImage:
    """Create a mosaic of multiple images arranged in a grid.

    Args:
        images: List of PIL images to arrange in a mosaic.
        images_per_row: Number of images per row in the mosaic (default: 2 for 2x2).

    Returns:
        A single PIL image containing the mosaic.
    """
    if not images:
        raise ValueError("At least one image is required to create a mosaic")

    # Calculate grid dimensions
    num_images = len(images)
    rows = (num_images + images_per_row - 1) // images_per_row
    cols = min(num_images, images_per_row)

    # Get image dimensions (assume all images have the same size)
    first_image = images[0]
    img_width, img_height = first_image.size

    # Create the mosaic image
    mosaic_width = cols * img_width
    mosaic_height = rows * img_height
    mosaic = PILImageClass.new(
        "RGB", (mosaic_width, mosaic_height), color=(255, 255, 255)
    )

    # Paste images into the mosaic
    for idx, image in enumerate(images):
        row = idx // images_per_row
        col = idx % images_per_row
        x = col * img_width
        y = row * img_height

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        mosaic.paste(image, (x, y))

    return mosaic


def load_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a high-quality font with fallbacks.

    Attempts to load a system font, falling back to default if unavailable.

    Args:
        size: Font size in pixels.

    Returns:
        A PIL font object.
    """
    # List of font paths to try on different systems
    font_paths = []

    if sys.platform == "darwin":  # macOS
        font_paths.extend(
            [
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Arial.ttf",
            ]
        )
    elif sys.platform.startswith("linux"):
        font_paths.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            ]
        )
    elif sys.platform == "win32":
        font_paths.extend(
            [
                "C:\\Windows\\Fonts\\arial.ttf",
                "C:\\Windows\\Fonts\\segoeui.ttf",
            ]
        )

    # Try to load the first available font
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                pass

    # Fallback: use default font with size if available
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Size argument was added in Pillow 10.1.0
        return ImageFont.load_default()
