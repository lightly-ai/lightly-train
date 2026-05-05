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

from lightly_train._visualize import image_classification
from lightly_train.types import ImageClassificationBatch

_WHITE_COLOR: float = 1.0
_WHITE_PIXEL: tuple[int, int, int] = (255, 255, 255)


def _non_white_bbox(image: PILImage) -> tuple[int, int, int, int] | None:
    """Bounding box of pixels that differ from pure white, or None if all white."""
    white = Image.new(image.mode, image.size, _WHITE_PIXEL)
    return ImageChops.difference(image, white).getbbox()


def _has_legend(image: PILImage) -> bool:
    """Return True if the image has any pixel that differs from pure white."""
    return _non_white_bbox(image) is not None


def _make_batch(
    *,
    batch_size: int = 1,
    height: int = 32,
    width: int = 32,
    classes: list[Tensor] | None = None,
) -> ImageClassificationBatch:
    image = torch.rand(batch_size, 3, height, width)
    if classes is None:
        classes = [torch.zeros(0, dtype=torch.long) for _ in range(batch_size)]
    return ImageClassificationBatch(
        image_path=[f"img_{i}.jpg" for i in range(batch_size)],
        image=image,
        classes=classes,
    )


def _make_batch_from_image(
    *,
    image: Tensor,
    classes: list[Tensor] | None = None,
) -> ImageClassificationBatch:
    batch_size = image.shape[0]
    if classes is None:
        classes = [torch.zeros(0, dtype=torch.long) for _ in range(batch_size)]
    return ImageClassificationBatch(
        image_path=[f"img_{i}.jpg" for i in range(batch_size)],
        image=image,
        classes=classes,
    )


def test_plot_image_classification_labels__grid_caps_at_max_images() -> None:
    batch = _make_batch(batch_size=4, height=16, width=16)
    result = image_classification.plot_image_classification_labels(
        batch=batch, included_classes={0: "_"}, max_images=2, image_normalize=None
    )
    assert result.size == (32, 16)


def test_plot_image_classification_labels__empty_classes_produces_clean_image() -> None:
    # No class labels → no overlay drawn; the image passes through unchanged.
    batch = _make_batch_from_image(
        image=torch.full((1, 3, 32, 32), _WHITE_COLOR),
        classes=[torch.zeros(0, dtype=torch.long)],
    )
    result = image_classification.plot_image_classification_labels(
        batch=batch, included_classes={0: "_"}, max_images=1, image_normalize=None
    )
    assert result.getpixel((0, 0)) == _WHITE_PIXEL
    assert result.getpixel((31, 31)) == _WHITE_PIXEL


def test_plot_image_classification_labels__no_image_normalize_skips_denormalization() -> (
    None
):
    # Without image_normalize the image tensor passes through unchanged.
    # Uniform 0.4 -> 102, no labels drawn.
    batch = _make_batch_from_image(
        image=torch.full((1, 3, 32, 32), 0.4),
        classes=[torch.zeros(0, dtype=torch.long)],
    )
    result = image_classification.plot_image_classification_labels(
        batch=batch, included_classes={0: "_"}, max_images=1, image_normalize=None
    )
    assert result.getpixel((0, 0)) == (102, 102, 102)
    assert result.getpixel((31, 31)) == (102, 102, 102)


def test_plot_image_classification_labels__image_normalize_denormalizes() -> None:
    # With image_normalize, the image is denormalized before rendering.
    # Input tensor of 0.0 with mean=0.5, std=0.5 -> denormalized to 0.5 -> pixel 127.
    # Without denormalization, 0.0 would render as black (0).
    batch = _make_batch_from_image(
        image=torch.zeros(1, 3, 32, 32),
        classes=[torch.zeros(0, dtype=torch.long)],
    )
    result = image_classification.plot_image_classification_labels(
        batch=batch,
        included_classes={0: "_"},
        max_images=1,
        image_normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    )
    assert result.getpixel((0, 0)) == (127, 127, 127)
    assert result.getpixel((31, 31)) == (127, 127, 127)


def test_plot_image_classification_labels__multiple_classes_stack_vertically() -> None:
    # Two labels are stacked vertically in the legend. The second label extends
    # the legend further down compared to a single label.
    image = torch.full((1, 3, 256, 256), _WHITE_COLOR)
    included_classes = {0: "cat", 1: "dog"}
    result_one = image_classification.plot_image_classification_labels(
        batch=_make_batch_from_image(
            image=image,
            classes=[torch.tensor([0], dtype=torch.long)],
        ),
        included_classes=included_classes,
        max_images=1,
        image_normalize=None,
    )
    result_two = image_classification.plot_image_classification_labels(
        batch=_make_batch_from_image(
            image=image,
            classes=[torch.tensor([0, 1], dtype=torch.long)],
        ),
        included_classes=included_classes,
        max_images=1,
        image_normalize=None,
    )
    bbox_one = _non_white_bbox(result_one)
    bbox_two = _non_white_bbox(result_two)
    assert bbox_one is not None
    assert bbox_two is not None
    # The two-label legend extends further down than the one-label legend.
    assert bbox_two[3] > bbox_one[3]


def test_plot_image_classification_labels__mixed_empty_nonempty_annotations() -> None:
    # Image 0 has a class label; image 1 has none.
    # Grid is 2×1 (256 wide, 128 tall): image 0 at x=0..127, image 1 at x=128..255.
    batch = _make_batch_from_image(
        image=torch.full((2, 3, 128, 128), _WHITE_COLOR),
        classes=[
            torch.tensor([0], dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        ],
    )
    result = image_classification.plot_image_classification_labels(
        batch=batch, included_classes={0: "cat"}, max_images=2, image_normalize=None
    )
    # Image 0 has a legend; image 1 stays fully white.
    assert _has_legend(result.crop((0, 0, 128, 128)))
    assert not _has_legend(result.crop((128, 0, 256, 128)))


def test_plot_image_classification_predictions__grid_caps_at_max_images() -> None:
    batch = _make_batch(batch_size=4, height=16, width=16)
    logits = torch.zeros(4, 2)
    result = image_classification.plot_image_classification_predictions(
        batch=batch,
        logits=logits,
        included_classes={0: "_", 1: "_"},
        max_images=2,
        top_k=1,
        image_normalize=None,
    )
    assert result.size == (32, 16)


def test_plot_image_classification_predictions__no_image_normalize_skips_denormalization() -> (
    None
):
    # Without image_normalize the image tensor passes through unchanged.
    # Check the far corner to avoid the corner-label overlay area.
    batch = _make_batch_from_image(
        image=torch.full((1, 3, 128, 128), 0.4),
        classes=[torch.zeros(0, dtype=torch.long)],
    )
    logits = torch.zeros(1, 2)
    result = image_classification.plot_image_classification_predictions(
        batch=batch,
        logits=logits,
        included_classes={0: "_", 1: "_"},
        max_images=1,
        top_k=1,
        image_normalize=None,
    )
    assert result.getpixel((127, 127)) == (102, 102, 102)


def test_plot_image_classification_predictions__image_normalize_denormalizes() -> None:
    # With image_normalize, the image is denormalized before rendering.
    # Input tensor of 0.0 with mean=0.5, std=0.5 -> denormalized to 0.5 -> pixel 127.
    # Without denormalization, 0.0 would render as black (0).
    # Check the far corner to avoid the corner-label overlay area.
    batch = _make_batch_from_image(
        image=torch.zeros(1, 3, 128, 128),
        classes=[torch.zeros(0, dtype=torch.long)],
    )
    logits = torch.zeros(1, 2)
    result = image_classification.plot_image_classification_predictions(
        batch=batch,
        logits=logits,
        included_classes={0: "_", 1: "_"},
        max_images=1,
        top_k=1,
        image_normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    )
    assert result.getpixel((127, 127)) == (127, 127, 127)


def test_plot_image_classification_predictions__effective_k_multi_label() -> None:
    # With top_k=1 but two ground-truth labels, effective_k=max(1,2)=2 predictions
    # are drawn. The legend extends further down with two entries than with one.
    image = torch.full((1, 3, 256, 256), _WHITE_COLOR)
    logits = torch.tensor([[10.0, 5.0, 1.0]])
    included_classes = {0: "cat", 1: "dog", 2: "bird"}
    result_one_gt = image_classification.plot_image_classification_predictions(
        batch=_make_batch_from_image(
            image=image,
            classes=[torch.tensor([0], dtype=torch.long)],
        ),
        logits=logits,
        included_classes=included_classes,
        max_images=1,
        top_k=1,
        image_normalize=None,
    )
    result_two_gt = image_classification.plot_image_classification_predictions(
        batch=_make_batch_from_image(
            image=image,
            classes=[torch.tensor([0, 1], dtype=torch.long)],
        ),
        logits=logits,
        included_classes=included_classes,
        max_images=1,
        top_k=1,
        image_normalize=None,
    )
    bbox_one = _non_white_bbox(result_one_gt)
    bbox_two = _non_white_bbox(result_two_gt)
    assert bbox_one is not None
    assert bbox_two is not None
    assert bbox_two[3] > bbox_one[3]


def test_plot_image_classification_predictions__mixed_empty_nonempty_annotations() -> (
    None
):
    # Image 0 has a ground-truth label; image 1 has none. Both still receive
    # top_k=1 predictions (effective_k=max(1,0)=1 for image 1).
    # Grid is 2×1 (256 wide, 128 tall): image 0 at x=0..127, image 1 at x=128..255.
    batch = _make_batch_from_image(
        image=torch.full((2, 3, 128, 128), _WHITE_COLOR),
        classes=[
            torch.tensor([0], dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        ],
    )
    logits = torch.zeros(2, 2)
    result = image_classification.plot_image_classification_predictions(
        batch=batch,
        logits=logits,
        included_classes={0: "cat", 1: "_"},
        max_images=2,
        top_k=1,
        image_normalize=None,
    )
    assert result.size == (256, 128)
    # Both images get a top-1 prediction drawn as a legend.
    assert _has_legend(result.crop((0, 0, 128, 128)))
    assert _has_legend(result.crop((128, 0, 256, 128)))
