#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch

from lightly_train._pre_post_processing import instance_segmentation


def test_combine_instance_segmentation_tiles() -> None:
    masks_global = torch.zeros(1, 8, 8, dtype=torch.bool)
    masks_global[0, :2, :2] = True
    labels_global = torch.tensor([2])
    scores_global = torch.tensor([0.8])

    masks_tiles = torch.zeros(3, 8, 8, dtype=torch.bool)
    masks_tiles[0, 2:4, 2:4] = True
    masks_tiles[1, 4:6, 4:6] = True
    masks_tiles[2, 6:, 6:] = True
    labels_tiles = torch.tensor([0, 0, 1])
    scores_tiles = torch.tensor([0.2, 0.9, 0.5])

    labels_out, masks_out, scores_out = (
        instance_segmentation.combine_instance_segmentation_tiles(
            pred_global={
                "labels": labels_global,
                "masks": masks_global,
                "scores": scores_global,
            },
            pred_tiles={
                "labels": labels_tiles,
                "masks": masks_tiles,
                "scores": scores_tiles,
            },
            nms_iou_threshold=0.5,
            global_local_iou_threshold=0.5,
        )
    )

    torch.testing.assert_close(scores_out, torch.tensor([0.8, 0.9, 0.5, 0.2]))
    torch.testing.assert_close(labels_out, torch.tensor([2, 0, 1, 0]))
    torch.testing.assert_close(
        masks_out, torch.cat([masks_global, masks_tiles[[1, 2, 0]]], dim=0)
    )


def test_combine_instance_segmentation_tiles__suppresses_same_label_overlap() -> None:
    masks_global = torch.zeros(0, 8, 8, dtype=torch.bool)
    labels_global = torch.empty(0, dtype=torch.long)
    scores_global = torch.empty(0)
    masks_tiles = torch.zeros(2, 8, 8, dtype=torch.bool)
    masks_tiles[:, 2:6, 2:6] = True
    labels_tiles = torch.tensor([1, 1])
    scores_tiles = torch.tensor([0.7, 0.9])

    labels_out, masks_out, scores_out = (
        instance_segmentation.combine_instance_segmentation_tiles(
            pred_global={
                "labels": labels_global,
                "masks": masks_global,
                "scores": scores_global,
            },
            pred_tiles={
                "labels": labels_tiles,
                "masks": masks_tiles,
                "scores": scores_tiles,
            },
            nms_iou_threshold=0.5,
            global_local_iou_threshold=0.5,
        )
    )

    torch.testing.assert_close(labels_out, torch.tensor([1]))
    torch.testing.assert_close(scores_out, torch.tensor([0.9]))
    torch.testing.assert_close(masks_out, masks_tiles[1:2])


def test_combine_instance_segmentation_tiles__keeps_different_label_overlap() -> None:
    masks_global = torch.zeros(0, 8, 8, dtype=torch.bool)
    labels_global = torch.empty(0, dtype=torch.long)
    scores_global = torch.empty(0)
    masks_tiles = torch.zeros(2, 8, 8, dtype=torch.bool)
    masks_tiles[:, 2:6, 2:6] = True
    labels_tiles = torch.tensor([1, 2])
    scores_tiles = torch.tensor([0.7, 0.9])

    labels_out, masks_out, scores_out = (
        instance_segmentation.combine_instance_segmentation_tiles(
            pred_global={
                "labels": labels_global,
                "masks": masks_global,
                "scores": scores_global,
            },
            pred_tiles={
                "labels": labels_tiles,
                "masks": masks_tiles,
                "scores": scores_tiles,
            },
            nms_iou_threshold=0.5,
            global_local_iou_threshold=0.5,
        )
    )

    torch.testing.assert_close(labels_out, torch.tensor([2, 1]))
    torch.testing.assert_close(scores_out, torch.tensor([0.9, 0.7]))
    torch.testing.assert_close(masks_out, masks_tiles[[1, 0]])


def test_combine_instance_segmentation_tiles__suppresses_same_label_global_overlap() -> (
    None
):
    masks_global = torch.zeros(1, 8, 8, dtype=torch.bool)
    masks_global[0, 2:6, 2:6] = True
    labels_global = torch.tensor([1])
    scores_global = torch.tensor([0.8])
    masks_tiles = torch.zeros(3, 8, 8, dtype=torch.bool)
    masks_tiles[0, 2:6, 2:6] = True
    masks_tiles[1, 2:6, 2:6] = True
    masks_tiles[2, :2, :2] = True
    labels_tiles = torch.tensor([1, 2, 1])
    scores_tiles = torch.tensor([0.9, 0.7, 0.6])

    labels_out, masks_out, scores_out = (
        instance_segmentation.combine_instance_segmentation_tiles(
            pred_global={
                "labels": labels_global,
                "masks": masks_global,
                "scores": scores_global,
            },
            pred_tiles={
                "labels": labels_tiles,
                "masks": masks_tiles,
                "scores": scores_tiles,
            },
            nms_iou_threshold=1.0,
            global_local_iou_threshold=0.5,
        )
    )

    torch.testing.assert_close(labels_out, torch.tensor([1, 2, 1]))
    torch.testing.assert_close(scores_out, torch.tensor([0.8, 0.7, 0.6]))
    torch.testing.assert_close(
        masks_out, torch.cat([masks_global, masks_tiles[[1, 2]]], dim=0)
    )


def test_combine_instance_segmentation_tiles__suppresses_lower_iou_same_label_global() -> (
    None
):
    # The tile mask overlaps a different-label global mask most strongly, but also
    # overlaps a same-label global mask above the threshold. It must be suppressed
    # based on the same-label overlap, not the single strongest (different-label)
    # match.
    masks_global = torch.zeros(2, 8, 8, dtype=torch.bool)
    masks_global[0, 2:6, 2:6] = True  # different label, IoU == 1.0 with the tile mask
    masks_global[1, 2:6, 2:5] = True  # same label, IoU == 0.75 with the tile mask
    labels_global = torch.tensor([2, 1])
    scores_global = torch.tensor([0.9, 0.8])
    masks_tiles = torch.zeros(1, 8, 8, dtype=torch.bool)
    masks_tiles[0, 2:6, 2:6] = True
    labels_tiles = torch.tensor([1])
    scores_tiles = torch.tensor([0.7])

    labels_out, masks_out, scores_out = (
        instance_segmentation.combine_instance_segmentation_tiles(
            pred_global={
                "labels": labels_global,
                "masks": masks_global,
                "scores": scores_global,
            },
            pred_tiles={
                "labels": labels_tiles,
                "masks": masks_tiles,
                "scores": scores_tiles,
            },
            nms_iou_threshold=0.5,
            global_local_iou_threshold=0.5,
        )
    )

    torch.testing.assert_close(labels_out, labels_global)
    torch.testing.assert_close(masks_out, masks_global)
    torch.testing.assert_close(scores_out, scores_global)


def test_combine_instance_segmentation_tiles__keeps_same_label_low_global_overlap() -> (
    None
):
    masks_global = torch.zeros(1, 8, 8, dtype=torch.bool)
    masks_global[0, 2:6, 2:6] = True
    labels_global = torch.tensor([1])
    scores_global = torch.tensor([0.8])
    masks_tiles = torch.zeros(1, 8, 8, dtype=torch.bool)
    masks_tiles[0, :2, :2] = True
    labels_tiles = torch.tensor([1])
    scores_tiles = torch.tensor([0.9])

    labels_out, masks_out, scores_out = (
        instance_segmentation.combine_instance_segmentation_tiles(
            pred_global={
                "labels": labels_global,
                "masks": masks_global,
                "scores": scores_global,
            },
            pred_tiles={
                "labels": labels_tiles,
                "masks": masks_tiles,
                "scores": scores_tiles,
            },
            nms_iou_threshold=0.5,
            global_local_iou_threshold=0.5,
        )
    )

    torch.testing.assert_close(labels_out, torch.tensor([1, 1]))
    torch.testing.assert_close(scores_out, torch.tensor([0.8, 0.9]))
    torch.testing.assert_close(masks_out, torch.cat([masks_global, masks_tiles], dim=0))


def test_combine_instance_segmentation_tiles__handles_empty_tile_predictions() -> None:
    masks_global = torch.zeros(1, 8, 8, dtype=torch.bool)
    masks_global[0, 2:6, 2:6] = True
    labels_global = torch.tensor([1])
    scores_global = torch.tensor([0.8])
    labels_tiles = torch.empty(0, dtype=torch.long)
    masks_tiles = torch.empty(0, 8, 8, dtype=torch.bool)
    scores_tiles = torch.empty(0)

    labels_out, masks_out, scores_out = (
        instance_segmentation.combine_instance_segmentation_tiles(
            pred_global={
                "labels": labels_global,
                "masks": masks_global,
                "scores": scores_global,
            },
            pred_tiles={
                "labels": labels_tiles,
                "masks": masks_tiles,
                "scores": scores_tiles,
            },
            nms_iou_threshold=0.5,
            global_local_iou_threshold=0.5,
        )
    )

    torch.testing.assert_close(labels_out, labels_global)
    torch.testing.assert_close(masks_out, masks_global)
    torch.testing.assert_close(scores_out, scores_global)


def test_combine_instance_segmentation_tiles__handles_empty_predictions() -> None:
    labels_global = torch.empty(0, dtype=torch.long)
    masks_global = torch.empty(0, 8, 8, dtype=torch.bool)
    scores_global = torch.empty(0)
    labels_tiles = torch.empty(0, dtype=torch.long)
    masks_tiles = torch.empty(0, 8, 8, dtype=torch.bool)
    scores_tiles = torch.empty(0)

    labels_out, masks_out, scores_out = (
        instance_segmentation.combine_instance_segmentation_tiles(
            pred_global={
                "labels": labels_global,
                "masks": masks_global,
                "scores": scores_global,
            },
            pred_tiles={
                "labels": labels_tiles,
                "masks": masks_tiles,
                "scores": scores_tiles,
            },
            nms_iou_threshold=0.5,
            global_local_iou_threshold=0.5,
        )
    )

    assert labels_out.shape == (0,)
    assert masks_out.shape == (0, 8, 8)
    assert scores_out.shape == (0,)
    assert labels_out.dtype == labels_global.dtype
    assert masks_out.dtype == masks_global.dtype
    assert scores_out.dtype == scores_global.dtype
