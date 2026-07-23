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
from torchvision.ops import batched_nms, box_iou


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

    tiles = []
    tiles_coordinates = []

    for h_start in h_starts:
        for w_start in w_starts:
            # Extract the tile.
            tile = image[:, h_start : h_start + h_tile, w_start : w_start + w_tile]
            tiles.append(tile)
            tiles_coordinates.append(
                torch.tensor([w_start, h_start], device=tile.device)
            )

    # Stack the tiles and coordinates
    tiles = torch.stack(tiles)
    tiles_coordinates = torch.stack(tiles_coordinates)

    return tiles, tiles_coordinates


def _class_aware_mask_nms(
    labels: Tensor,
    masks: Tensor,
    scores: Tensor,
    iou_threshold: float,
) -> Tensor:
    order = scores.argsort(descending=True)
    keep: list[Tensor] = []

    # Flatten and bool-cast the masks once up front so the NMS loop below does not
    # repeat this work (and the associated allocations) on every iteration.
    masks_flat = masks.flatten(1).bool()
    areas = masks_flat.sum(dim=1)

    while order.numel() > 0:
        current = order[0]
        keep.append(current)

        if order.numel() == 1:
            break

        rest = order[1:]
        same_label = labels[rest] == labels[current]
        ious = _mask_iou_flat(
            masks_flat[current : current + 1],
            areas[current : current + 1],
            masks_flat[rest],
            areas[rest],
        ).squeeze(0)
        suppress = same_label & (ious > iou_threshold)
        order = rest[~suppress]

    if len(keep) == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.stack(keep)


def _mask_iou(masks1: Tensor, masks2: Tensor) -> Tensor:
    masks1_flat = masks1.flatten(1).bool()
    masks2_flat = masks2.flatten(1).bool()
    return _mask_iou_flat(
        masks1_flat,
        masks1_flat.sum(dim=1),
        masks2_flat,
        masks2_flat.sum(dim=1),
    )


def _mask_iou_flat(
    masks1_flat: Tensor,
    areas1: Tensor,
    masks2_flat: Tensor,
    areas2: Tensor,
) -> Tensor:
    # Compute IoUs one row of masks1 at a time. Materializing the full
    # (N1, N2, H*W) intersection tensor at once can exhaust memory for many
    # full-resolution masks, so we bound peak memory to a single (N2, H*W) slice.
    ious = torch.zeros(
        masks1_flat.shape[0], masks2_flat.shape[0], device=masks1_flat.device
    )
    for i in range(masks1_flat.shape[0]):
        intersection = (masks1_flat[i] & masks2_flat).sum(dim=1)
        union = areas1[i] + areas2 - intersection
        ious[i] = torch.where(
            union > 0,
            intersection.float() / union.float(),
            torch.zeros_like(union, dtype=torch.float),
        )

    return ious


def combine_object_detection_tiles(
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

    # NMS on tiles predictions is needed due overlapping tiles. Suppression is
    # class-aware so a high-confidence prediction cannot hide another class.
    if boxes_tiles.numel() > 0:
        keep = batched_nms(boxes_tiles, scores_tiles, labels_tiles, nms_iou_threshold)
        labels_tiles = labels_tiles[keep]
        boxes_tiles = boxes_tiles[keep]
        scores_tiles = scores_tiles[keep]

    # Drop tile boxes that overlap global boxes of same class
    if boxes_global.numel() > 0 and boxes_tiles.numel() > 0:
        # Compute overlap between tiles and global predictions.
        ious = box_iou(boxes_tiles, boxes_global)

        # Only keep tiles predictions that do not overlap above the threshold with
        # any global prediction of the same class. The same-label check must be
        # applied before reducing over global predictions: reducing first (e.g.
        # via the single max-IoU global box) would miss a same-label overlap that
        # is not the strongest one.
        same_label = labels_tiles[:, None] == labels_global[None, :]
        overlaps_same_label = (same_label & (ious > global_local_iou_threshold)).any(
            dim=1
        )
        keep = ~overlaps_same_label
        labels_tiles = labels_tiles[keep]
        boxes_tiles = boxes_tiles[keep]
        scores_tiles = scores_tiles[keep]

    # Concatenate the global and tiles predictions
    labels = torch.cat([labels_global, labels_tiles], dim=0)
    boxes = torch.cat([boxes_global, boxes_tiles], dim=0)
    scores = torch.cat([scores_global, scores_tiles], dim=0)

    return labels, boxes, scores


def combine_instance_segmentation_tiles(
    pred_global: dict[str, Tensor],
    pred_tiles: dict[str, Tensor],
    nms_iou_threshold: float = 0.5,
    global_local_iou_threshold: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Combine predictions from global and tiled instance segmentation views.

    Args:
        pred_global: dict with keys "labels", "masks", "scores". Masks must be
            full-image binary masks of shape (N, H, W).
        pred_tiles: dict with keys "labels", "masks", "scores". Masks must be
            stitched into full-image coordinates with shape (N, H, W).
        nms_iou_threshold: Mask IoU used in NMS of tiles predictions.
        global_local_iou_threshold: Mask IoU above which a tile mask is removed if
            it matches a global mask of same label.

    Returns:
        Filtered labels, masks, scores as a tuple.
    """
    # Get tiles and global predictions.
    labels_global = pred_global["labels"]
    masks_global = pred_global["masks"]
    scores_global = pred_global["scores"]
    labels_tiles = pred_tiles["labels"]
    masks_tiles = pred_tiles["masks"]
    scores_tiles = pred_tiles["scores"]

    # NMS on tiles predictions is needed due overlapping tiles.
    if masks_tiles.numel() > 0:
        keep = _class_aware_mask_nms(
            labels=labels_tiles,
            masks=masks_tiles,
            scores=scores_tiles,
            iou_threshold=nms_iou_threshold,
        )
        labels_tiles = labels_tiles[keep]
        masks_tiles = masks_tiles[keep]
        scores_tiles = scores_tiles[keep]

    # Drop tile masks that overlap global masks of same class.
    if masks_global.numel() > 0 and masks_tiles.numel() > 0:
        # Compute overlap between tiles and global predictions.
        ious = _mask_iou(masks_tiles, masks_global)

        # Only keep tiles predictions that do not overlap above the threshold with
        # any global prediction of the same class. The same-label check must be
        # applied before reducing over global predictions: reducing first (e.g.
        # via the single max-IoU global mask) would miss a same-label overlap that
        # is not the strongest one.
        same_label = labels_tiles[:, None] == labels_global[None, :]
        overlaps_same_label = (same_label & (ious > global_local_iou_threshold)).any(
            dim=1
        )
        keep = ~overlaps_same_label
        labels_tiles = labels_tiles[keep]
        masks_tiles = masks_tiles[keep]
        scores_tiles = scores_tiles[keep]

    # Concatenate the global and tiles predictions.
    labels = torch.cat([labels_global, labels_tiles], dim=0)
    masks = torch.cat([masks_global, masks_tiles], dim=0)
    scores = torch.cat([scores_global, scores_tiles], dim=0)

    return labels, masks, scores
