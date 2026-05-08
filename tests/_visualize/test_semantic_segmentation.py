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

from lightly_train._visualize import semantic_segmentation
from lightly_train.types import BinaryMasksDict, MaskSemanticSegmentationBatch

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


def _empty_binary_masks(
    *, batch_size: int, height: int, width: int
) -> list[BinaryMasksDict]:
    return [
        BinaryMasksDict(
            masks=torch.zeros(0, height, width, dtype=torch.bool),
            labels=torch.zeros(0, dtype=torch.long),
        )
        for _ in range(batch_size)
    ]


def _make_batch(
    *,
    batch_size: int = 1,
    height: int = 32,
    width: int = 32,
    image: Tensor | None = None,
    mask: Tensor | None = None,
) -> MaskSemanticSegmentationBatch:
    if image is None:
        image = torch.rand(batch_size, 3, height, width)
    if mask is None:
        mask = torch.zeros(batch_size, height, width, dtype=torch.long)
    return MaskSemanticSegmentationBatch(
        image_path=[f"img_{i}.png" for i in range(batch_size)],
        image=image,
        mask=mask,
        binary_masks=_empty_binary_masks(
            batch_size=batch_size, height=height, width=width
        ),
    )


def test_plot_semantic_segmentation_labels__grid_caps_at_max_images() -> None:
    batch = _make_batch(batch_size=4, height=16, width=16)
    result = semantic_segmentation.plot_semantic_segmentation_labels(
        batch=batch,
        class_names={0: "_"},
        max_images=2,
        image_normalize=None,
        alpha=0.0,
    )
    assert result.size == (32, 16)


def test_plot_semantic_segmentation_labels__no_image_normalize_skips_denormalization() -> (
    None
):
    # alpha=0 disables the mask overlay; uniform mask draws no contours; an
    # empty included_classes draws no legend. The 0.4 input therefore renders
    # as uniform (102, 102, 102).
    batch = _make_batch(
        image=torch.full((1, 3, 32, 32), 0.4),
        mask=torch.zeros(1, 32, 32, dtype=torch.long),
    )
    result = semantic_segmentation.plot_semantic_segmentation_labels(
        batch=batch,
        class_names={},
        max_images=1,
        image_normalize=None,
        alpha=0.0,
    )
    assert result.getpixel((0, 0)) == (102, 102, 102)
    assert result.getpixel((31, 31)) == (102, 102, 102)


def test_plot_semantic_segmentation_labels__image_normalize_denormalizes() -> None:
    # Input 0.0 with mean=0.5, std=0.5 -> denormalized to 0.5 -> pixel 127.
    # Without denormalization, 0.0 would render as black (0).
    batch = _make_batch(
        image=torch.zeros(1, 3, 32, 32),
        mask=torch.zeros(1, 32, 32, dtype=torch.long),
    )
    result = semantic_segmentation.plot_semantic_segmentation_labels(
        batch=batch,
        class_names={},
        max_images=1,
        image_normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
        alpha=0.0,
    )
    assert result.getpixel((0, 0)) == (127, 127, 127)
    assert result.getpixel((31, 31)) == (127, 127, 127)


def test_plot_semantic_segmentation_labels__alpha_one_replaces_image_with_mask_colors() -> (
    None
):
    # alpha=1.0 makes the blended image equal to the overlay, which paints
    # every class-0 pixel with the class-0 color. The far corner avoids the
    # upper-left legend region.
    batch = _make_batch(
        image=torch.full((1, 3, 64, 64), _WHITE_COLOR),
        mask=torch.zeros(1, 64, 64, dtype=torch.long),
    )
    result = semantic_segmentation.plot_semantic_segmentation_labels(
        batch=batch,
        class_names={0: "cat"},
        max_images=1,
        image_normalize=None,
        alpha=1.0,
    )
    assert result.getpixel((63, 63)) == _CLASS_0_COLOR


def test_plot_semantic_segmentation_labels__contours_drawn_at_class_boundary() -> None:
    # Top half class 0, bottom half class 1. The two rows straddling the
    # boundary are part of the contour and rendered black, regardless of the
    # underlying overlay color. Probe pixels are taken near the right edge to
    # avoid the upper-left legend region.
    mask = torch.zeros(1, 128, 128, dtype=torch.long)
    mask[0, 64:, :] = 1
    batch = _make_batch(
        image=torch.full((1, 3, 128, 128), _WHITE_COLOR),
        mask=mask,
    )
    result = semantic_segmentation.plot_semantic_segmentation_labels(
        batch=batch,
        class_names={0: "cat", 1: "dog"},
        max_images=1,
        image_normalize=None,
        alpha=1.0,
    )
    # Pixel at (col=120, row=63) and (col=120, row=64) are on the contour.
    assert result.getpixel((120, 63)) == _BLACK_PIXEL
    assert result.getpixel((120, 64)) == _BLACK_PIXEL
    # Pixel inside the top region keeps its class-0 overlay color.
    assert result.getpixel((120, 30)) == _CLASS_0_COLOR
    # Pixel inside the bottom region keeps its class-1 overlay color.
    assert result.getpixel((120, 100)) == _CLASS_1_COLOR


def test_plot_semantic_segmentation_labels__legend_skips_class_not_in_included_classes() -> (
    None
):
    # With alpha=0 and a uniform mask there is nothing on the canvas except
    # the legend. So a class that isn't in included_classes (e.g. ignore_index
    # 255) leaves the image fully white, while a known class adds a legend.
    batch = _make_batch(
        image=torch.full((1, 3, 256, 256), _WHITE_COLOR),
        mask=torch.zeros(1, 256, 256, dtype=torch.long),
    )
    with_known_class = semantic_segmentation.plot_semantic_segmentation_labels(
        batch=batch,
        class_names={0: "cat"},
        max_images=1,
        image_normalize=None,
        alpha=0.0,
    )
    with_unknown_class = semantic_segmentation.plot_semantic_segmentation_labels(
        batch=batch,
        class_names={1: "dog"},
        max_images=1,
        image_normalize=None,
        alpha=0.0,
    )
    assert _has_legend(with_known_class)
    assert not _has_legend(with_unknown_class)


def test_plot_semantic_segmentation_predictions__overlay_matches_argmax_class() -> None:
    # Logits are non-uniform across regions and across classes, so each pixel's
    # argmax depends on which class has the larger logit there. Top half:
    # class 0 wins (2.0 > -2.0). Bottom half: class 1 wins (2.0 > -2.0).
    # alpha=1.0 makes the blended image equal to the overlay, which paints
    # each pixel with its predicted class color. Probe pixels are taken near
    # the right edge to avoid the upper-left legend region, and away from the
    # boundary to avoid the contour.
    logits = torch.zeros(2, 128, 128)
    logits[0, :64, :] = 2.0
    logits[1, :64, :] = -2.0
    logits[0, 64:, :] = -2.0
    logits[1, 64:, :] = 2.0
    batch = _make_batch(
        image=torch.full((1, 3, 128, 128), _WHITE_COLOR),
        mask=torch.zeros(1, 128, 128, dtype=torch.long),
    )
    result = semantic_segmentation.plot_semantic_segmentation_predictions(
        batch=batch,
        logits=[logits],
        class_names={0: "cat", 1: "dog"},
        max_images=1,
        image_normalize=None,
        alpha=1.0,
    )
    # Top region argmax == 0, so overlay must paint the class-0 color.
    assert result.getpixel((120, 30)) == _CLASS_0_COLOR
    # Bottom region argmax == 1, so overlay must paint the class-1 color.
    assert result.getpixel((120, 100)) == _CLASS_1_COLOR
