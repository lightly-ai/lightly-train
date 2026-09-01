from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from lightly_train._task_models.eomt.instance_segmentation_validation import (
    get_chunked_labels_masks_scores,
)


def _reference_get_labels_masks_scores(
    mask_logits: Tensor, class_logits: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    labels = class_logits.argmax(dim=-1)
    masks = mask_logits.sigmoid()
    scores = class_logits.softmax(dim=-1).amax(dim=-1)
    return labels, masks, scores


def test_get_chunked_labels_masks_scores_matches_unchunked() -> None:
    torch.manual_seed(0)
    mask_logits = torch.randn(1, 4, 3, 5)
    class_logits = torch.randn(1, 4, 6)
    resize_size = (7, 9)
    crop_size = (5, 9)
    output_size = (12, 20)

    chunk_call_count = 0

    def chunked_get_labels_masks_scores(
        mask_logits: Tensor, class_logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        nonlocal chunk_call_count
        chunk_call_count += 1
        return _reference_get_labels_masks_scores(mask_logits, class_logits)

    # Force one query per chunk.
    max_chunk_bytes = mask_logits.element_size() * resize_size[0] * resize_size[1]
    chunked = get_chunked_labels_masks_scores(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=resize_size,
        crop_size=crop_size,
        output_size=output_size,
        get_labels_masks_scores=chunked_get_labels_masks_scores,
        max_chunk_bytes=max_chunk_bytes,
    )
    assert chunk_call_count == mask_logits.shape[1]

    # Reference: resize to padded model input size, crop the padded region,
    # then resize to the original image size before converting to masks.
    reference_logits = F.interpolate(mask_logits, resize_size, mode="bilinear")
    reference_logits = reference_logits[..., : crop_size[0], : crop_size[1]]
    reference_logits = F.interpolate(reference_logits, output_size, mode="bilinear")
    expected = _reference_get_labels_masks_scores(reference_logits, class_logits)

    for actual, expected_part in zip(chunked, expected):
        torch.testing.assert_close(actual, expected_part)


def test_get_chunked_labels_masks_scores_single_chunk_matches_unchunked() -> None:
    """Chunking must not change results when all queries fit into one chunk."""
    torch.manual_seed(0)
    mask_logits = torch.randn(1, 4, 6, 6)
    class_logits = torch.randn(1, 4, 6)
    resize_size = (8, 8)
    crop_size = (8, 6)
    output_size = (16, 14)

    chunked = get_chunked_labels_masks_scores(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=resize_size,
        crop_size=crop_size,
        output_size=output_size,
        get_labels_masks_scores=_reference_get_labels_masks_scores,
    )

    reference_logits = F.interpolate(mask_logits, resize_size, mode="bilinear")
    reference_logits = reference_logits[..., : crop_size[0], : crop_size[1]]
    reference_logits = F.interpolate(reference_logits, output_size, mode="bilinear")
    expected = _reference_get_labels_masks_scores(reference_logits, class_logits)

    for actual, expected_part in zip(chunked, expected):
        torch.testing.assert_close(actual, expected_part)


def test_get_chunked_labels_masks_scores_crops_padded_region() -> None:
    """A query active only in the padded region must be cropped away.

    This mirrors the padded validation case: a small logit patch whose signal
    lies entirely in the padded columns must not leak into the final mask.
    """
    # Single query, single logit pixel that lands in the padded region after
    # the resize to the padded model input size.
    mask_logits = torch.full((1, 1, 2, 2), 10.0)
    class_logits = torch.tensor([[[0.0, 1.0]]])
    resize_size = (4, 4)
    # Only the left half of the padded input is real image content.
    crop_size = (4, 2)
    output_size = (4, 2)

    labels, masks, scores = get_chunked_labels_masks_scores(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=resize_size,
        crop_size=crop_size,
        output_size=output_size,
        get_labels_masks_scores=_reference_get_labels_masks_scores,
    )

    reference_logits = F.interpolate(mask_logits, resize_size, mode="bilinear")
    reference_logits = reference_logits[..., : crop_size[0], : crop_size[1]]
    reference_logits = F.interpolate(reference_logits, output_size, mode="bilinear")
    expected = _reference_get_labels_masks_scores(reference_logits, class_logits)

    for actual, expected_part in zip((labels, masks, scores), expected):
        torch.testing.assert_close(actual, expected_part)


def test_get_chunked_labels_masks_scores_no_crop() -> None:
    """crop_size=None skips cropping (square inputs without padding)."""
    torch.manual_seed(0)
    mask_logits = torch.randn(1, 3, 4, 4)
    class_logits = torch.randn(1, 3, 5)
    resize_size = (6, 6)
    output_size = (10, 10)

    chunked = get_chunked_labels_masks_scores(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=resize_size,
        crop_size=None,
        output_size=output_size,
        get_labels_masks_scores=_reference_get_labels_masks_scores,
    )

    reference_logits = F.interpolate(mask_logits, resize_size, mode="bilinear")
    reference_logits = F.interpolate(reference_logits, output_size, mode="bilinear")
    expected = _reference_get_labels_masks_scores(reference_logits, class_logits)

    for actual, expected_part in zip(chunked, expected):
        torch.testing.assert_close(actual, expected_part)
