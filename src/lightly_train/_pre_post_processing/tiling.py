#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


def _tile_starts(size: int, tile_size: int, step: int) -> list[int]:
    if size <= tile_size:
        return [0]

    last_start = size - tile_size
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def tile_image(
    image: Tensor,
    overlap: float,
    tile_size: tuple[int, int],
    *,
    padding_mode: Literal["resize", "pad"] = "resize",
) -> tuple[Tensor, Tensor]:
    """
    Split an image tensor into tiles.

    If the input image is smaller than `tile_size` in either spatial dimension, it
    is either upscaled or padded depending on `padding_mode`.

    Args:
        image: Image tensor of shape (C, H, W).
        overlap: Fractional overlap between tiles in [0, 1) (0.0 means no overlap).
        tile_size: (tile_height, tile_width).
        padding_mode: How to handle images smaller than `tile_size`. "resize" keeps
            the historical behavior and upscales the image. "pad" pads the image on
            the bottom and right without changing the original pixels.

    Returns:
        tiles: Tensor of shape (N, C, tile_size[0], tile_size[1]), containing all extracted tiles.
        tiles_coordinates: Tensor of shape (N, 2) with (x, y) = (w_start, h_start) for each tile.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in the range [0.0, 1.0).")
    if padding_mode not in ("resize", "pad"):
        raise ValueError("padding_mode must be either 'resize' or 'pad'.")

    # Current image shape.
    _, h, w = image.shape
    h_tile, w_tile = tile_size
    if h_tile <= 0 or w_tile <= 0:
        raise ValueError("tile_size must contain positive values.")

    # If the image is too small, resize or pad it to fit at least one tile.
    if h < h_tile or w < w_tile:
        if padding_mode == "resize":
            scale = max(h_tile / h, w_tile / w)
            new_h = math.ceil(h * scale)
            new_w = math.ceil(w * scale)
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            pad_h = max(0, h_tile - h)
            pad_w = max(0, w_tile - w)
            image = F.pad(image, pad=[0, pad_w, 0, pad_h])
        _, h, w = image.shape

    # Define the steps.
    h_step = max(1, int((1.0 - overlap) * h_tile))
    w_step = max(1, int((1.0 - overlap) * w_tile))
    h_starts = _tile_starts(size=h, tile_size=h_tile, step=h_step)
    w_starts = _tile_starts(size=w, tile_size=w_tile, step=w_step)

    tiles: list[Tensor] = []
    tiles_coordinates: list[Tensor] = []

    for h_start in h_starts:
        for w_start in w_starts:
            # Extract the tile.
            tile = image[:, h_start : h_start + h_tile, w_start : w_start + w_tile]
            tiles.append(tile)
            tiles_coordinates.append(
                torch.tensor([w_start, h_start], device=tile.device)
            )

    # Stack the tiles and coordinates
    return torch.stack(tiles), torch.stack(tiles_coordinates)
