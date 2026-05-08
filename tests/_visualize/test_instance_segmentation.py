#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from PIL.Image import Image as PILImage
from torch import Tensor

from lightly_train._visualize import instance_segmentation, utils
from lightly_train.types import BinaryMasksDict, InstanceSegmentationBatch

_BACKGROUND_COLOR: float = 0.0
_WHITE_COLOR: float = 1.0
_BLACK_PIXEL: tuple[int, int, int] = (0, 0, 0)
_WHITE_PIXEL: tuple[int, int, int] = (255, 255, 255)


def _assert_bbox_corners_have_color(
    *,
    image: PILImage,
    xyxy: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = xyxy
    assert image.getpixel((x1, y1)) == color
    assert image.getpixel((x2, y1)) == color
    assert image.getpixel((x1, y2)) == color
    assert image.getpixel((x2, y2)) == color


def _make_batch(
    *,
    image: Tensor,
    binary_masks: list[BinaryMasksDict],
    bboxes: list[Tensor],
) -> InstanceSegmentationBatch:
    batch_size = image.shape[0]
    classes = [bm["labels"] for bm in binary_masks]
    return InstanceSegmentationBatch(
        image_path=[f"img_{i}.png" for i in range(batch_size)],
        image=image,
        binary_masks=binary_masks,
        bboxes=bboxes,
        classes=classes,
    )


def _binary_masks_block(
    *,
    height: int,
    width: int,
    label: int,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
) -> BinaryMasksDict:
    """Build a single-instance BinaryMasksDict whose mask covers [y0:y1, x0:x1]."""
    masks = torch.zeros(1, height, width, dtype=torch.bool)
    masks[0, y0:y1, x0:x1] = True
    return BinaryMasksDict(masks=masks, labels=torch.tensor([label], dtype=torch.long))


def _empty_binary_masks(*, height: int, width: int) -> BinaryMasksDict:
    return BinaryMasksDict(
        masks=torch.zeros(0, height, width, dtype=torch.bool),
        labels=torch.zeros(0, dtype=torch.long),
    )


def test_plot_instance_segmentation_labels__overlay_and_bbox_use_class_color() -> None:
    # 128x128 black canvas. Class 1 mask covers the top-left 64x64 region (rows
    # and cols 0..63). cxcywh [0.25, 0.25, 0.5, 0.5] maps to xyxy [0, 0, 64, 64],
    # tight around the mask. y1=0 is below label_height, so the label rectangle
    # is drawn below the bbox top edge inside the box, near (0, 0). The probe
    # (32, 32) sits below the label rect (~27 px tall) and away from the outline.
    binary_masks = _binary_masks_block(
        height=128, width=128, label=1, y0=0, x0=0, y1=64, x1=64
    )
    bboxes = [torch.tensor([[0.25, 0.25, 0.5, 0.5]])]
    batch = _make_batch(
        image=torch.full((1, 3, 128, 128), _BACKGROUND_COLOR),
        binary_masks=[binary_masks],
        bboxes=bboxes,
    )
    result = instance_segmentation.plot_instance_segmentation_labels(
        batch=batch,
        class_names={1: "dog"},
        max_images=1,
        image_normalize=None,
        alpha=1.0,
    )
    # All four bbox corners are painted in class 1's color.
    _assert_bbox_corners_have_color(
        image=result, xyxy=(0, 0, 64, 64), color=utils._get_class_color(1)
    )
    # Mask interior, below the label rect and off the bbox outline: at alpha=1
    # the overlay alone determines the pixel, so it must be class 1's color.
    # A buggy implementation that skips the overlay would leave this pixel black.
    assert result.getpixel((32, 32)) == utils._get_class_color(1)
    # Pixel far outside both mask and bbox: overlay is black there, and at
    # alpha=1 the blended image equals the overlay, so the pixel stays black.
    assert result.getpixel((100, 100)) == _BLACK_PIXEL


def test_plot_instance_segmentation_labels__alpha_zero_keeps_image_in_mask_region() -> (
    None
):
    # alpha=0 disables the overlay, so even pixels covered by the mask must
    # retain the original image color. The bbox is still drawn afterwards, so
    # we probe inside the bbox but well away from the outline and below the
    # label rectangle.
    binary_masks = _binary_masks_block(
        height=128, width=128, label=1, y0=0, x0=0, y1=64, x1=64
    )
    bboxes = [torch.tensor([[0.25, 0.25, 0.5, 0.5]])]
    batch = _make_batch(
        image=torch.full((1, 3, 128, 128), _WHITE_COLOR),
        binary_masks=[binary_masks],
        bboxes=bboxes,
    )
    result = instance_segmentation.plot_instance_segmentation_labels(
        batch=batch,
        class_names={1: "dog"},
        max_images=1,
        image_normalize=None,
        alpha=0.0,
    )
    assert result.getpixel((32, 32)) == _WHITE_PIXEL


def test_plot_instance_segmentation_labels__image_normalize_denormalizes() -> None:
    # mean=std=0.5 maps a 0.0 tensor to 0.5 -> pixel 127. Without
    # denormalization the input would render as black. The probe pixel is
    # outside the mask and outside the bbox so it isn't overwritten.
    binary_masks = _binary_masks_block(
        height=64, width=64, label=1, y0=0, x0=0, y1=16, x1=16
    )
    bboxes = [torch.tensor([[0.125, 0.125, 0.25, 0.25]])]
    batch = _make_batch(
        image=torch.zeros(1, 3, 64, 64),
        binary_masks=[binary_masks],
        bboxes=bboxes,
    )
    result = instance_segmentation.plot_instance_segmentation_labels(
        batch=batch,
        class_names={1: "dog"},
        max_images=1,
        image_normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
        alpha=0.0,
    )
    assert result.getpixel((63, 63)) == (127, 127, 127)


def test_plot_instance_segmentation_predictions__score_threshold_filters_low_scores() -> (
    None
):
    # Two predictions on a 128x128 black canvas:
    #   instance 0: class 0, mask in rows/cols 0..63, score 0.9 -> kept.
    #   instance 1: class 1, mask in rows/cols 64..127, score 0.2 -> dropped.
    # The dropped instance must produce no overlay AND no bbox, so its bbox
    # corners must remain black. A bug that filters only the overlay but still
    # draws the box (or vice versa) would fail at these probes.
    masks = torch.zeros(2, 128, 128, dtype=torch.bool)
    masks[0, 0:64, 0:64] = True
    masks[1, 64:128, 64:128] = True
    predictions = [
        {
            "masks": masks,
            "labels": torch.tensor([0, 1], dtype=torch.long),
            "scores": torch.tensor([0.9, 0.2]),
        }
    ]
    batch = _make_batch(
        image=torch.full((1, 3, 128, 128), _BACKGROUND_COLOR),
        binary_masks=[_empty_binary_masks(height=128, width=128)],
        bboxes=[torch.zeros(0, 4)],
    )
    result = instance_segmentation.plot_instance_segmentation_predictions(
        batch=batch,
        predictions=predictions,
        class_names={0: "cat", 1: "dog"},
        max_images=1,
        image_normalize=None,
        alpha=1.0,
        score_threshold=0.5,
    )
    # Kept instance: bbox is derived from mask extent -> xyxy [0, 0, 63, 63].
    # All four corners are painted in class 0's color.
    _assert_bbox_corners_have_color(
        image=result, xyxy=(0, 0, 63, 63), color=utils._get_class_color(0)
    )
    # Mask interior of the kept instance: overlay paints class 0's color.
    # Probe well below the label rect and off the bbox outline.
    assert result.getpixel((32, 40)) == utils._get_class_color(0)
    # Dropped instance region: no overlay (class 1 never got painted) and no
    # bbox (filtered before _bboxes_from_masks). All four would-be corners
    # stay black.
    _assert_bbox_corners_have_color(
        image=result, xyxy=(64, 64, 127, 127), color=_BLACK_PIXEL
    )


def test_plot_instance_segmentation_predictions__bbox_derived_from_mask_extent() -> (
    None
):
    # The mask is a 32x32 block at rows/cols 16..47 on a 128x128 canvas.
    # _bboxes_from_masks uses inclusive max coords (xs.max() / ys.max()),
    # so the derived bbox is xyxy [16, 16, 47, 47]. alpha=0 disables the
    # overlay so only the bbox is drawn — that isolates the bbox geometry.
    # The mask sits in the canvas interior (not at origin) so a buggy
    # implementation that always uses a fixed bbox (e.g. the full image, or
    # the top-left corner) would mispaint the canvas corners.
    masks = torch.zeros(1, 128, 128, dtype=torch.bool)
    masks[0, 16:48, 16:48] = True
    predictions = [
        {
            "masks": masks,
            "labels": torch.tensor([0], dtype=torch.long),
            "scores": torch.tensor([0.9]),
        }
    ]
    batch = _make_batch(
        image=torch.full((1, 3, 128, 128), _BACKGROUND_COLOR),
        binary_masks=[_empty_binary_masks(height=128, width=128)],
        bboxes=[torch.zeros(0, 4)],
    )
    result = instance_segmentation.plot_instance_segmentation_predictions(
        batch=batch,
        predictions=predictions,
        class_names={0: "cat"},
        max_images=1,
        image_normalize=None,
        alpha=0.0,
        score_threshold=0.5,
    )
    # Bbox corners are at exactly the mask extent. y1=16 is below
    # label_height (~22), so the label is drawn below the box and its glyphs
    # don't overlap the bottom corners we probe here.
    _assert_bbox_corners_have_color(
        image=result, xyxy=(16, 16, 47, 47), color=utils._get_class_color(0)
    )
    # Canvas corners far outside the mask remain background — a bbox spanning
    # the full image would have painted these.
    assert result.getpixel((0, 0)) == _BLACK_PIXEL
    assert result.getpixel((127, 127)) == _BLACK_PIXEL
    # Just outside the bbox stays untouched — proves the bbox is tight, not
    # padded by some constant.
    assert result.getpixel((48, 48)) == _BLACK_PIXEL
