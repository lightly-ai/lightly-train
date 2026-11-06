#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from lightly_train._task_models.eomt.instance_segmentation import (
    get_chunk_size,
    get_instance_segmentation_prediction,
)


def _get_labels_masks_scores(
    mask_logits: Tensor, class_logits: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Mirrors EoMTInstanceSegmentation.get_labels_masks_scores."""
    scores = class_logits.softmax(dim=-1, dtype=torch.float32)[..., :-1].type_as(
        class_logits
    )
    scores, labels = torch.max(scores, dim=-1)
    masks = mask_logits > 0
    mask_logits_fp32 = mask_logits.float()
    masks_fp32 = masks.float()
    score_per_query = (
        mask_logits_fp32.sigmoid().flatten(2) * masks_fp32.flatten(2)
    ).sum(2)
    mask_pixels_per_query = masks_fp32.flatten(2).sum(2)
    mask_scores = score_per_query / mask_pixels_per_query
    mask_scores = torch.where(
        mask_pixels_per_query > 0, mask_scores, torch.zeros_like(mask_scores)
    )
    return labels, masks, (scores * mask_scores).type_as(mask_logits)


def _get_reference_prediction(
    *,
    mask_logits: Tensor,
    class_logits: Tensor,
    resize_size: tuple[int, int],
    crop_size: tuple[int, int] | None,
    image_size: tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor]:
    """The unchunked validation path from before the chunking change."""
    logits = mask_logits.unsqueeze(0)
    logits = F.interpolate(logits, resize_size, mode="bilinear")
    if crop_size is not None:
        logits = logits[..., : crop_size[0], : crop_size[1]]
    logits = F.interpolate(logits, image_size, mode="bilinear")
    labels, masks, scores = _get_labels_masks_scores(logits, class_logits.unsqueeze(0))
    return labels[0], masks[0], scores[0]


def test_get_chunk_size__training_parity() -> None:
    # Numbers from the issue: 200 queries, a 2943x1435 image and a 640x640 model
    # input size.
    chunk_size = get_chunk_size(
        num_queries=200, resize_size=(640, 640), image_size=(1435, 2943)
    )
    assert chunk_size == 19
    # A chunk covers no more mask elements than training scores at once.
    assert chunk_size * 1435 * 2943 <= 200 * 640 * 640


def test_get_chunk_size__image_smaller_than_model_input() -> None:
    # Images up to the model input size are scored in a single chunk, which keeps
    # the behaviour unchanged for them.
    chunk_size = get_chunk_size(
        num_queries=200, resize_size=(640, 640), image_size=(320, 320)
    )
    assert chunk_size >= 200


def test_get_chunk_size__clamps_to_one() -> None:
    chunk_size = get_chunk_size(
        num_queries=1, resize_size=(2, 2), image_size=(1000, 1000)
    )
    assert chunk_size == 1


def test_get_instance_segmentation_prediction__matches_reference() -> None:
    generator = torch.Generator().manual_seed(0)
    mask_logits = torch.randn((7, 3, 5), generator=generator)
    class_logits = torch.randn((7, 4), generator=generator)
    resize_size = (6, 8)
    crop_size = (5, 7)
    image_size = (9, 11)
    chunk_sizes: list[int] = []

    def get_labels_masks_scores(
        mask_logits: Tensor, class_logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        chunk_sizes.append(mask_logits.shape[1])
        return _get_labels_masks_scores(mask_logits, class_logits)

    prediction = get_instance_segmentation_prediction(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=resize_size,
        crop_size=crop_size,
        image_size=image_size,
        get_labels_masks_scores=get_labels_masks_scores,
    )

    assert chunk_sizes == [3, 3, 1]
    labels, masks, scores = _get_reference_prediction(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=resize_size,
        crop_size=crop_size,
        image_size=image_size,
    )
    assert torch.equal(prediction["labels"], labels)
    assert torch.equal(prediction["masks"], masks)
    # Scores are summed over the mask pixels. The summation order depends on the
    # chunk size, so they match up to floating point tolerance.
    torch.testing.assert_close(prediction["scores"], scores)


def test_get_instance_segmentation_prediction__single_chunk_matches_reference() -> None:
    generator = torch.Generator().manual_seed(0)
    mask_logits = torch.randn((4, 6, 6), generator=generator)
    class_logits = torch.randn((4, 6), generator=generator)
    resize_size = (8, 8)
    crop_size = (8, 6)
    image_size = (4, 4)
    chunk_sizes: list[int] = []

    def get_labels_masks_scores(
        mask_logits: Tensor, class_logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        chunk_sizes.append(mask_logits.shape[1])
        return _get_labels_masks_scores(mask_logits, class_logits)

    prediction = get_instance_segmentation_prediction(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=resize_size,
        crop_size=crop_size,
        image_size=image_size,
        get_labels_masks_scores=get_labels_masks_scores,
    )

    assert chunk_sizes == [4]
    labels, masks, scores = _get_reference_prediction(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=resize_size,
        crop_size=crop_size,
        image_size=image_size,
    )
    assert torch.equal(prediction["labels"], labels)
    assert torch.equal(prediction["masks"], masks)
    torch.testing.assert_close(prediction["scores"], scores)


def test_get_instance_segmentation_prediction__crops_padded_region() -> None:
    # A query that is only active in the padded columns must not appear in the mask.
    mask_logits = torch.tensor([[[-10.0, 10.0], [-10.0, 10.0]]])
    class_logits = torch.tensor([[0.0, 1.0]])
    resize_size = (4, 4)
    # Only the left half of the padded input is image content.
    crop_size = (4, 2)
    image_size = (4, 2)

    prediction = get_instance_segmentation_prediction(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=resize_size,
        crop_size=crop_size,
        image_size=image_size,
        get_labels_masks_scores=_get_labels_masks_scores,
    )

    _, masks, _ = _get_reference_prediction(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=resize_size,
        crop_size=crop_size,
        image_size=image_size,
    )
    assert torch.equal(prediction["masks"], masks)
    assert not prediction["masks"].any()
    # Without the crop the padded columns leak into the mask.
    _, masks_no_crop, _ = _get_reference_prediction(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=resize_size,
        crop_size=None,
        image_size=image_size,
    )
    assert masks_no_crop.any()


def test_get_instance_segmentation_prediction__shapes() -> None:
    generator = torch.Generator().manual_seed(0)
    mask_logits = torch.randn((7, 3, 5), generator=generator)
    class_logits = torch.randn((7, 4), generator=generator)

    prediction = get_instance_segmentation_prediction(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=(6, 8),
        crop_size=(5, 7),
        image_size=(9, 11),
        get_labels_masks_scores=_get_labels_masks_scores,
    )

    assert prediction["labels"].shape == (7,)
    assert prediction["masks"].shape == (7, 9, 11)
    assert prediction["scores"].shape == (7,)


def test_get_instance_segmentation_prediction__masks_on_cpu() -> None:
    generator = torch.Generator().manual_seed(0)
    mask_logits = torch.randn((7, 3, 5), generator=generator)
    class_logits = torch.randn((7, 4), generator=generator)

    prediction = get_instance_segmentation_prediction(
        mask_logits=mask_logits,
        class_logits=class_logits,
        resize_size=(6, 8),
        crop_size=(5, 7),
        image_size=(9, 11),
        get_labels_masks_scores=_get_labels_masks_scores,
    )

    assert prediction["masks"].device.type == "cpu"
    assert prediction["labels"].device == mask_logits.device
    assert prediction["scores"].device == mask_logits.device
