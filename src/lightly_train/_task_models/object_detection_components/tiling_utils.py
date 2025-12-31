#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import box_iou, nms


def tile_image(
    image: Tensor, overlap: float, tile_size: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """
    Split an image tensor into tiles.

    If the input image is smaller than `tile_size` in either spatial dimension,
    it is upscaled so that at least one tile of size `tile_size` fits.

    Args:
        image: Image tensor of shape (C, H, W).
        overlap: Fractional overlap between tiles in [0, 1) (0.0 means no overlap).
        tile_size: (tile_height, tile_width).

    Returns:
        tiles: Tensor of shape (N, C, tile_size[0], tile_size[1]), containing all extracted tiles.
        tiles_coordinates: Tensor of shape (N, 2) with (x, y) = (w_start, h_start) for each tile.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in the range [0.0, 1.0).")

    # Current image shape.
    _, h, w = image.shape
    h_tile, w_tile = tile_size

    # If the image is too small, upscale it to fit at least one tile.
    if h < h_tile or w < w_tile:
        scale = max(h_tile / h, w_tile / w)
        new_h = math.ceil(h * scale)
        new_w = math.ceil(w * scale)
        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        _, h, w = image.shape

    # Define the steps.
    h_step = max(1, int((1.0 - overlap) * h_tile))
    w_step = max(1, int((1.0 - overlap) * w_tile))

    tiles = []
    tiles_coordinates = []

    for h_start in range(0, h, h_step):
        for w_start in range(0, w, w_step):
            # Compute the start and end of the current tile.
            h_end = min(h_start + h_tile, h)
            h_start = h_end - h_tile

            w_end = min(w_start + w_tile, w)
            w_start = w_end - w_tile

            # Extract the tile.
            tile = image[:, h_start:h_end, w_start:w_end]
            tiles.append(tile)
            tiles_coordinates.append(
                torch.tensor([w_start, h_start], device=tile.device)
            )

    # Stack the tiles and coordinates
    tiles = torch.stack(tiles)
    tiles_coordinates = torch.stack(tiles_coordinates)

    return tiles, tiles_coordinates


def combine_predictions_tiles_and_global(
    pred_global: dict[str, Tensor],
    pred_tiles: dict[str, Tensor],
    nms_iou_threshold: float = 0.2,
    global_local_iou_threshold: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Combine predictions from the global view (full image) and local views (image tiles).

    Args:
        pred_global: dict with keys "labels", "bboxes", "scores".
        pred_tiles: dict with keys "labels", "bboxes", "scores".
        nms_iou_threshold: IoU used in NMS of tiles predictions.
        global_local_iou_threshold: IoU above which a tile box is removed if it matches a global box of same label.

    Returns:
        Filtered labels, boxes, scores as a tuple.
    """
    # Get tiles and global predictions.
    labels_global = pred_global["labels"]
    boxes_global = pred_global["bboxes"]
    scores_global = pred_global["scores"]
    labels_tiles = pred_tiles["labels"]
    boxes_tiles = pred_tiles["bboxes"]
    scores_tiles = pred_tiles["scores"]

    # NMS on tiles predictions is needed due overlapping tiles.
    if boxes_tiles.numel() > 0:
        keep = nms(boxes_tiles, scores_tiles, nms_iou_threshold)
        labels_tiles = labels_tiles[keep]
        boxes_tiles = boxes_tiles[keep]
        scores_tiles = scores_tiles[keep]

    # Drop tile boxes that overlap global boxes of same class
    if boxes_global.numel() > 0 and boxes_tiles.numel() > 0:
        # Compute overlap between tiles and global predictions.
        ious = box_iou(boxes_tiles, boxes_global)
        max_iou, argmax = ious.max(dim=1)

        # Only keep tiles predictions that do not overlap with a
        # global prediction of the same class.
        same_label = labels_tiles == labels_global[argmax]
        keep = torch.logical_or(max_iou <= global_local_iou_threshold, ~same_label)
        labels_tiles = labels_tiles[keep]
        boxes_tiles = boxes_tiles[keep]
        scores_tiles = scores_tiles[keep]

    # Concatenate the global and tiles predictions
    labels = torch.cat([labels_global, labels_tiles], dim=0)
    boxes = torch.cat([boxes_global, boxes_tiles], dim=0)
    scores = torch.cat([scores_global, scores_tiles], dim=0)

    return labels, boxes, scores
