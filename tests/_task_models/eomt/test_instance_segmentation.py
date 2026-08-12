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

from lightly_train._task_models.eomt.instance_segmentation import (
    _FP32_WORKING_BUFFERS_PER_PIXEL,
    _query_chunk_size,
    get_instance_segmentation_prediction,
)


def _get_labels_masks_scores(
    mask_logits: Tensor, class_logits: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    class_scores = class_logits.softmax(dim=-1, dtype=torch.float32)[..., :-1]
    scores, labels = torch.max(class_scores, dim=-1)
    masks = mask_logits > 0
    masks_fp32 = masks.float()
    mask_logits_fp32 = mask_logits.float()
    pixels = masks_fp32.flatten(2).sum(2)
    mask_scores = (
        mask_logits_fp32.sigmoid().flatten(2) * masks_fp32.flatten(2)
    ).sum(2) / pixels
    mask_scores = torch.where(pixels > 0, mask_scores, torch.zeros_like(mask_scores))
    return labels, masks, (scores * mask_scores).type_as(mask_logits)


def test_query_chunk_size_uses_byte_budget() -> None:
    image_size = (4, 5)
    bytes_per_query = (
        4
        * 5
        * torch.tensor([], dtype=torch.float32).element_size()
        * _FP32_WORKING_BUFFERS_PER_PIXEL
    )

    assert _query_chunk_size(
        image_size=image_size, memory_budget_bytes=2 * bytes_per_query
    ) == 2
    assert _query_chunk_size(
        image_size=image_size, memory_budget_bytes=bytes_per_query - 1
    ) == 1


def test_chunked_prediction_matches_single_chunk() -> None:
    generator = torch.Generator().manual_seed(0)
    mask_logits = torch.randn((7, 3, 5), generator=generator)
    class_logits = torch.randn((7, 4), generator=generator)
    kwargs = {
        "mask_logits": mask_logits,
        "class_logits": class_logits,
        "model_image_size": (6, 8),
        "crop_size": (5, 7),
        "image_size": (9, 11),
        "get_labels_masks_scores": _get_labels_masks_scores,
    }

    unchunked = get_instance_segmentation_prediction(
        **kwargs,
        memory_budget_bytes=10**9,
    )
    bytes_per_query = (
        9
        * 11
        * torch.tensor([], dtype=torch.float32).element_size()
        * _FP32_WORKING_BUFFERS_PER_PIXEL
    )
    chunked = get_instance_segmentation_prediction(
        **kwargs,
        memory_budget_bytes=2 * bytes_per_query,
    )

    torch.testing.assert_close(chunked["labels"], unchunked["labels"])
    torch.testing.assert_close(chunked["masks"], unchunked["masks"])
    torch.testing.assert_close(chunked["scores"], unchunked["scores"])
    assert chunked["masks"].device.type == "cpu"
