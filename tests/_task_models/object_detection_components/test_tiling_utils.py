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
import torch.nn.functional as F

from lightly_train._task_models.object_detection_components import tiling_utils


@pytest.fixture
def tile_image() -> torch.Tensor:
    return torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(3, 32, 32)


@pytest.fixture
def small_tile_image() -> torch.Tensor:
    return torch.arange(3 * 8 * 10, dtype=torch.float32).reshape(3, 8, 10)


def test_tile_image(tile_image: torch.Tensor) -> None:
    image = tile_image

    tiles, coordinates = tiling_utils.tile_image(
        image=image, overlap=0.5, tile_size=(16, 16)
    )

    assert tiles.shape == (9, 3, 16, 16)
    torch.testing.assert_close(
        coordinates,
        torch.tensor(
            [
                [0, 0],
                [8, 0],
                [16, 0],
                [0, 8],
                [8, 8],
                [16, 8],
                [0, 16],
                [8, 16],
                [16, 16],
            ],
            device=image.device,
        ),
    )
    torch.testing.assert_close(tiles[0], image[:, :16, :16])
    torch.testing.assert_close(tiles[-1], image[:, 16:32, 16:32])


def test_tile_image__resize_mode_on_small_image(small_tile_image: torch.Tensor) -> None:
    image = small_tile_image

    tiles, coordinates = tiling_utils.tile_image(
        image=image, overlap=0.2, tile_size=(16, 16), padding_mode="resize"
    )

    assert tiles.shape == (2, 3, 16, 16)
    torch.testing.assert_close(
        coordinates,
        torch.tensor([[0, 0], [4, 0]], device=image.device),
    )
    resized_image = F.interpolate(
        image.unsqueeze(0), size=(16, 20), mode="bilinear", align_corners=False
    ).squeeze(0)
    torch.testing.assert_close(tiles[0], resized_image[:, :16, :16])
    torch.testing.assert_close(tiles[-1], resized_image[:, :16, 4:20])


def test_tile_image__pad_mode_on_small_image(small_tile_image: torch.Tensor) -> None:
    image = small_tile_image

    tiles, coordinates = tiling_utils.tile_image(
        image=image, overlap=0.2, tile_size=(16, 16), padding_mode="pad"
    )

    assert tiles.shape == (1, 3, 16, 16)
    torch.testing.assert_close(coordinates, torch.tensor([[0, 0]], device=image.device))
    torch.testing.assert_close(tiles[0, :, :8, :10], image)
    assert torch.all(tiles[0, :, 8:, :] == 0)
    assert torch.all(tiles[0, :, :, 10:] == 0)


def test_tile_image__appends_last_tile_for_non_divisible_size() -> None:
    # 30 is not a multiple of the tile size, so the last tile in each dimension
    # must be snapped back to size - tile_size (14) to stay within the image.
    image = torch.arange(3 * 30 * 30, dtype=torch.float32).reshape(3, 30, 30)

    tiles, coordinates = tiling_utils.tile_image(
        image=image, overlap=0.0, tile_size=(16, 16)
    )

    assert tiles.shape == (4, 3, 16, 16)
    torch.testing.assert_close(
        coordinates,
        torch.tensor([[0, 0], [14, 0], [0, 14], [14, 14]], device=image.device),
    )
    torch.testing.assert_close(tiles[0], image[:, :16, :16])
    torch.testing.assert_close(tiles[-1], image[:, 14:30, 14:30])


@pytest.mark.parametrize("overlap", [-0.1, 1.0])
def test_tile_image__raises_for_invalid_overlap(
    overlap: float, tile_image: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        tiling_utils.tile_image(image=tile_image, overlap=overlap, tile_size=(16, 16))


@pytest.mark.parametrize("tile_size", [(0, 16), (16, 0), (-1, 16), (16, -1)])
def test_tile_image__raises_for_invalid_tile_size(
    tile_size: tuple[int, int], tile_image: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="tile_size"):
        tiling_utils.tile_image(image=tile_image, overlap=0.2, tile_size=tile_size)


def test_tile_image__raises_for_invalid_padding_mode(
    tile_image: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="padding_mode"):
        tiling_utils.tile_image(
            image=tile_image,
            overlap=0.2,
            tile_size=(16, 16),
            padding_mode="invalid",  # type: ignore[arg-type]
        )


def test_combine_object_detection_tiles() -> None:
    labels_global = torch.tensor([1])
    boxes_global = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores_global = torch.tensor([0.8])
    labels_tiles = torch.tensor([2, 3])
    boxes_tiles = torch.tensor(
        [
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],
        ]
    )
    scores_tiles = torch.tensor([0.7, 0.9])

    labels_out, boxes_out, scores_out = tiling_utils.combine_object_detection_tiles(
        pred_global={
            "labels": labels_global,
            "bboxes": boxes_global,
            "scores": scores_global,
        },
        pred_tiles={
            "labels": labels_tiles,
            "bboxes": boxes_tiles,
            "scores": scores_tiles,
        },
        nms_iou_threshold=0.5,
        global_local_iou_threshold=0.1,
    )

    torch.testing.assert_close(labels_out, torch.tensor([1, 3, 2]))
    torch.testing.assert_close(
        boxes_out,
        torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [40.0, 40.0, 50.0, 50.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ),
    )
    torch.testing.assert_close(scores_out, torch.tensor([0.8, 0.9, 0.7]))


def test_combine_object_detection_tiles__suppresses_tile_nms() -> None:
    labels_tiles = torch.tensor([1, 1, 2])
    boxes_tiles = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    scores_tiles = torch.tensor([0.8, 0.9, 0.7])

    labels_out, boxes_out, scores_out = tiling_utils.combine_object_detection_tiles(
        pred_global={
            "labels": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty(0, 4),
            "scores": torch.empty(0),
        },
        pred_tiles={
            "labels": labels_tiles,
            "bboxes": boxes_tiles,
            "scores": scores_tiles,
        },
        nms_iou_threshold=0.5,
        global_local_iou_threshold=0.1,
    )

    torch.testing.assert_close(labels_out, torch.tensor([1, 2]))
    torch.testing.assert_close(boxes_out, boxes_tiles[[1, 2]])
    torch.testing.assert_close(scores_out, torch.tensor([0.9, 0.7]))


def test_combine_object_detection_tiles__keeps_overlapping_different_labels() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
    labels_out, boxes_out, scores_out = tiling_utils.combine_object_detection_tiles(
        pred_global={
            "labels": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty(0, 4),
            "scores": torch.empty(0),
        },
        pred_tiles={
            "labels": torch.tensor([1, 2]),
            "bboxes": boxes,
            "scores": torch.tensor([0.9, 0.8]),
        },
        nms_iou_threshold=0.5,
        global_local_iou_threshold=0.1,
    )

    torch.testing.assert_close(labels_out, torch.tensor([1, 2]))
    torch.testing.assert_close(boxes_out, boxes)
    torch.testing.assert_close(scores_out, torch.tensor([0.9, 0.8]))


def test_combine_object_detection_tiles__suppresses_same_label_global_overlap() -> None:
    labels_global = torch.tensor([1])
    boxes_global = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores_global = torch.tensor([0.8])
    labels_tiles = torch.tensor([1, 2, 1])
    boxes_tiles = torch.tensor(
        [
            [1.0, 1.0, 9.0, 9.0],
            [1.0, 1.0, 9.0, 9.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    scores_tiles = torch.tensor([0.9, 0.7, 0.6])

    labels_out, boxes_out, scores_out = tiling_utils.combine_object_detection_tiles(
        pred_global={
            "labels": labels_global,
            "bboxes": boxes_global,
            "scores": scores_global,
        },
        pred_tiles={
            "labels": labels_tiles,
            "bboxes": boxes_tiles,
            "scores": scores_tiles,
        },
        nms_iou_threshold=1.0,
        global_local_iou_threshold=0.5,
    )

    torch.testing.assert_close(labels_out, torch.tensor([1, 2, 1]))
    torch.testing.assert_close(
        boxes_out,
        torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 9.0, 9.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ),
    )
    torch.testing.assert_close(scores_out, torch.tensor([0.8, 0.7, 0.6]))


def test_combine_object_detection_tiles__suppresses_lower_iou_same_label_global() -> (
    None
):
    # The tile box overlaps a different-label global box most strongly, but also
    # overlaps a same-label global box above the threshold. It must be suppressed
    # based on the same-label overlap, not the single strongest (different-label)
    # match.
    labels_global = torch.tensor([2, 1])
    boxes_global = torch.tensor(
        [
            [1.0, 1.0, 9.0, 9.0],  # different label, IoU == 1.0 with the tile box
            [0.0, 0.0, 10.0, 10.0],  # same label, IoU == 0.64 with the tile box
        ]
    )
    scores_global = torch.tensor([0.9, 0.8])
    labels_tiles = torch.tensor([1])
    boxes_tiles = torch.tensor([[1.0, 1.0, 9.0, 9.0]])
    scores_tiles = torch.tensor([0.7])

    labels_out, boxes_out, scores_out = tiling_utils.combine_object_detection_tiles(
        pred_global={
            "labels": labels_global,
            "bboxes": boxes_global,
            "scores": scores_global,
        },
        pred_tiles={
            "labels": labels_tiles,
            "bboxes": boxes_tiles,
            "scores": scores_tiles,
        },
        nms_iou_threshold=0.5,
        global_local_iou_threshold=0.5,
    )

    torch.testing.assert_close(labels_out, labels_global)
    torch.testing.assert_close(boxes_out, boxes_global)
    torch.testing.assert_close(scores_out, scores_global)


def test_combine_object_detection_tiles__handles_empty_predictions() -> None:
    labels_out, boxes_out, scores_out = tiling_utils.combine_object_detection_tiles(
        pred_global={
            "labels": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty(0, 4),
            "scores": torch.empty(0),
        },
        pred_tiles={
            "labels": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty(0, 4),
            "scores": torch.empty(0),
        },
        nms_iou_threshold=0.5,
        global_local_iou_threshold=0.1,
    )

    assert labels_out.shape == (0,)
    assert boxes_out.shape == (0, 4)
    assert scores_out.shape == (0,)
    assert labels_out.dtype == torch.long
    assert boxes_out.dtype == torch.float32
    assert scores_out.dtype == torch.float32


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
        tiling_utils.combine_instance_segmentation_tiles(
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
        tiling_utils.combine_instance_segmentation_tiles(
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
        tiling_utils.combine_instance_segmentation_tiles(
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
        tiling_utils.combine_instance_segmentation_tiles(
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
        tiling_utils.combine_instance_segmentation_tiles(
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
        tiling_utils.combine_instance_segmentation_tiles(
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
        tiling_utils.combine_instance_segmentation_tiles(
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
        tiling_utils.combine_instance_segmentation_tiles(
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
