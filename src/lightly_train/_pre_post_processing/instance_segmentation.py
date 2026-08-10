#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from torch import Tensor


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
