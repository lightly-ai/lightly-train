#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

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
) -> tuple[Tensor, Tensor]:
    """
    Split an image tensor into tiles.

    Tiles are crops of the image at its native resolution, never resampled, so the
    returned coordinates are always in ORIGINAL-image pixels: a box predicted on tile
    ``i`` is brought into original-image coordinates by adding
    ``tiles_coordinates[i]``, with no scale factor involved. Callers that want the
    tiles at a different resolution (SAHI magnification, for example) resize the
    returned stack themselves and keep `tile_size` as the size the coordinates and box
    scales refer to.

    That unconditional contract is deliberate. An earlier ``padding_mode="resize"``
    upscaled images smaller than a tile and returned coordinates in that upscaled
    frame without recording the scale, which silently multiplied every tile detection
    of such an image by that factor. Upscaling also made the tile count explode on
    extreme aspect ratios: fitting a 1x1000 image to a 640x640 tile produced 1250
    tiles, one model input row each, where padding produces 2.

    Tiles are not necessarily fully inside the image: if the image is smaller than
    `tile_size` in a dimension it is zero padded on the bottom/right, so a tile can
    extend up to ``tile_size - 1`` pixels past the image in that dimension. Padding can
    only happen in a dimension in which the image is smaller than the tile, and in that
    dimension there is exactly one tile position. Callers must account for the padded
    band: clip boxes to the original image or crop pasted masks to
    ``min(tile_size, orig - start)``.

    Args:
        image: Image tensor of shape (C, H, W).
        overlap: Fractional overlap between tiles in [0, 1) (0.0 means no overlap).
        tile_size: (tile_height, tile_width), in original-image pixels.

    Returns:
        tiles: Tensor of shape (N, C, tile_size[0], tile_size[1]), containing all extracted tiles.
        tiles_coordinates: Tensor of shape (N, 2) with (x, y) = (w_start, h_start) for
            each tile, in original-image pixels.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in the range [0.0, 1.0).")

    # Current image shape.
    _, h, w = image.shape
    h_tile, w_tile = tile_size
    if h_tile <= 0 or w_tile <= 0:
        raise ValueError("tile_size must contain positive values.")

    # Pad, never resize: resizing would put the coordinates in a frame the caller has no
    # way to undo.
    if h < h_tile or w < w_tile:
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
