#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from PIL import Image, ImageChops
from PIL.Image import Image as PILImage
from torch import Tensor

from lightly_train._visualize import panoptic_segmentation
from lightly_train.types import (
    MaskPanopticSegmentationBatch,
    PanopticBinaryMasksDict,
)

_WHITE_COLOR: float = 1.0
_WHITE_PIXEL: tuple[int, int, int] = (255, 255, 255)
_BLACK_PIXEL: tuple[int, int, int] = (0, 0, 0)
# These come from the deterministic golden-ratio palette in utils._get_class_color.
_CLASS_0_COLOR: tuple[int, int, int] = (242, 24, 24)
_CLASS_1_COLOR: tuple[int, int, int] = (24, 87, 242)


def _has_legend(image: PILImage) -> bool:
    """Return True if any pixel differs from pure white (i.e. legend was drawn)."""
    white = Image.new(image.mode, image.size, _WHITE_PIXEL)
    return ImageChops.difference(image, white).getbbox() is not None


def _empty_panoptic_binary_masks(
    *, batch_size: int, height: int, width: int
) -> list[PanopticBinaryMasksDict]:
    return [
        PanopticBinaryMasksDict(
            masks=torch.zeros(0, height, width, dtype=torch.bool),
            labels=torch.zeros(0, dtype=torch.long),
            iscrowd=torch.zeros(0, dtype=torch.long),
        )
        for _ in range(batch_size)
    ]


def _make_batch(
    *,
    batch_size: int = 1,
    height: int = 32,
    width: int = 32,
    image: Tensor | None = None,
    masks: Tensor | None = None,
) -> MaskPanopticSegmentationBatch:
    if image is None:
        image = torch.rand(batch_size, 3, height, width)
    if masks is None:
        masks = torch.zeros(batch_size, height, width, 2, dtype=torch.long)
    return MaskPanopticSegmentationBatch(
        image_path=[f"img_{i}.png" for i in range(batch_size)],
        image=image,
        masks=masks,
        binary_masks=_empty_panoptic_binary_masks(
            batch_size=batch_size, height=height, width=width
        ),
    )


def test_plot_panoptic_segmentation_labels__grid_caps_at_max_images() -> None:
    batch = _make_batch(batch_size=4, height=16, width=16)
    result = panoptic_segmentation.plot_panoptic_segmentation_labels(
        batch=batch,
        class_names={0: "_"},
        max_images=2,
        image_normalize=None,
        alpha=0.0,
    )
    assert result.size == (32, 16)


def test_plot_panoptic_segmentation_labels__no_image_normalize_skips_denormalization() -> (
    None
):
    # alpha=0 disables the mask overlay; a uniform mask draws no contours; an
    # empty included_classes draws no legend. The 0.4 input therefore renders
    # as uniform (102, 102, 102).
    batch = _make_batch(
        image=torch.full((1, 3, 32, 32), 0.4),
        masks=torch.zeros(1, 32, 32, 2, dtype=torch.long),
    )
    result = panoptic_segmentation.plot_panoptic_segmentation_labels(
        batch=batch,
        class_names={},
        max_images=1,
        image_normalize=None,
        alpha=0.0,
    )
    assert result.getpixel((0, 0)) == (102, 102, 102)
    assert result.getpixel((31, 31)) == (102, 102, 102)


def test_plot_panoptic_segmentation_labels__image_normalize_denormalizes() -> None:
    # Input 0.0 with mean=0.5, std=0.5 -> denormalized to 0.5 -> pixel 127.
    # Without denormalization, 0.0 would render as black (0).
    batch = _make_batch(
        image=torch.zeros(1, 3, 32, 32),
        masks=torch.zeros(1, 32, 32, 2, dtype=torch.long),
    )
    result = panoptic_segmentation.plot_panoptic_segmentation_labels(
        batch=batch,
        class_names={},
        max_images=1,
        image_normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
        alpha=0.0,
    )
    assert result.getpixel((0, 0)) == (127, 127, 127)
    assert result.getpixel((31, 31)) == (127, 127, 127)


def test_plot_panoptic_segmentation_labels__alpha_one_replaces_image_with_mask_colors() -> (
    None
):
    # alpha=1.0 makes the blended image equal to the overlay, which paints
    # every class-0 pixel with the class-0 color. The far corner avoids the
    # upper-left legend region.
    batch = _make_batch(
        image=torch.full((1, 3, 64, 64), _WHITE_COLOR),
        masks=torch.zeros(1, 64, 64, 2, dtype=torch.long),
    )
    result = panoptic_segmentation.plot_panoptic_segmentation_labels(
        batch=batch,
        class_names={0: "cat"},
        max_images=1,
        image_normalize=None,
        alpha=1.0,
    )
    assert result.getpixel((63, 63)) == _CLASS_0_COLOR


def test_plot_panoptic_segmentation_labels__contours_drawn_at_segment_boundary_within_same_class() -> (
    None
):
    # KEY panoptic feature: the panoptic visualizer draws contours along
    # *segment* boundaries, not class boundaries. Two adjacent instances of the
    # SAME class must still be separated by a black contour. Top half: class 0,
    # segment 0. Bottom half: class 0, segment 1. The semantic visualizer
    # (which uses class boundaries) would leave this image contour-free.
    masks = torch.zeros(1, 128, 128, 2, dtype=torch.long)
    masks[0, 64:, :, 1] = 1
    batch = _make_batch(
        image=torch.full((1, 3, 128, 128), _WHITE_COLOR),
        masks=masks,
    )
    result = panoptic_segmentation.plot_panoptic_segmentation_labels(
        batch=batch,
        class_names={0: "cat"},
        max_images=1,
        image_normalize=None,
        alpha=1.0,
    )
    # Probe near the right edge to avoid the upper-left legend region. Rows 63
    # and 64 straddle the segment-id boundary and must both be black.
    assert result.getpixel((120, 63)) == _BLACK_PIXEL
    assert result.getpixel((120, 64)) == _BLACK_PIXEL


def test_plot_panoptic_segmentation_labels__different_instances_same_class_get_different_colors() -> (
    None
):
    # Top half: class 0, segment 0 -> first instance, must use the base class
    # color (so the legend swatch matches). Bottom half: class 0, segment 1 ->
    # second instance, must use a hue-shifted color (otherwise the two
    # instances are visually indistinguishable). Probe pixels are taken near
    # the right edge to avoid the upper-left legend, and away from the
    # boundary contour at row 64.
    masks = torch.zeros(1, 128, 128, 2, dtype=torch.long)
    masks[0, 64:, :, 1] = 1
    batch = _make_batch(
        image=torch.full((1, 3, 128, 128), _WHITE_COLOR),
        masks=masks,
    )
    result = panoptic_segmentation.plot_panoptic_segmentation_labels(
        batch=batch,
        class_names={0: "cat"},
        max_images=1,
        image_normalize=None,
        alpha=1.0,
    )
    top_pixel = result.getpixel((120, 30))
    bottom_pixel = result.getpixel((120, 100))
    assert top_pixel == _CLASS_0_COLOR
    assert bottom_pixel != _CLASS_0_COLOR
    assert bottom_pixel != _BLACK_PIXEL


def test_plot_panoptic_segmentation_labels__legend_skips_class_not_in_included_classes() -> (
    None
):
    # With alpha=0 and a uniform mask there is nothing on the canvas except
    # the legend. So a class that isn't in included_classes leaves the image
    # fully white, while a known class adds a legend.
    batch = _make_batch(
        image=torch.full((1, 3, 256, 256), _WHITE_COLOR),
        masks=torch.zeros(1, 256, 256, 2, dtype=torch.long),
    )
    with_known_class = panoptic_segmentation.plot_panoptic_segmentation_labels(
        batch=batch,
        class_names={0: "cat"},
        max_images=1,
        image_normalize=None,
        alpha=0.0,
    )
    with_unknown_class = panoptic_segmentation.plot_panoptic_segmentation_labels(
        batch=batch,
        class_names={1: "dog"},
        max_images=1,
        image_normalize=None,
        alpha=0.0,
    )
    assert _has_legend(with_known_class)
    assert not _has_legend(with_unknown_class)


def test_plot_panoptic_segmentation_predictions__overlay_uses_predicted_class_colors() -> (
    None
):
    # Predicted (H, W, 2) mask with two classes split top/bottom, each with its
    # own segment id (so each is the first instance of its class and therefore
    # uses the base class color). The ground-truth batch is unused for color
    # selection — only pred_masks should drive the overlay. Probe near the
    # right edge to avoid the legend, and away from the contour at row 64.
    pred_mask = torch.zeros(128, 128, 2, dtype=torch.long)
    pred_mask[64:, :, 0] = 1
    pred_mask[64:, :, 1] = 1
    batch = _make_batch(
        image=torch.full((1, 3, 128, 128), _WHITE_COLOR),
        masks=torch.zeros(1, 128, 128, 2, dtype=torch.long),
    )
    result = panoptic_segmentation.plot_panoptic_segmentation_predictions(
        batch=batch,
        pred_masks=[pred_mask],
        class_names={0: "cat", 1: "dog"},
        max_images=1,
        image_normalize=None,
        alpha=1.0,
    )
    assert result.getpixel((120, 30)) == _CLASS_0_COLOR
    assert result.getpixel((120, 100)) == _CLASS_1_COLOR
