#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

# Approximate peak working set for resized logits and the FP32 mask-score
# intermediates in get_labels_masks_scores. The calculation uses six FP32-sized
# buffers per output pixel to stay conservative across input dtypes.
_INSTANCE_PREDICTION_MEMORY_BUDGET_BYTES = 512 * 1024**2
_FP32_WORKING_BUFFERS_PER_PIXEL = 6


def _query_chunk_size(
    *,
    image_size: tuple[int, int],
    memory_budget_bytes: int = _INSTANCE_PREDICTION_MEMORY_BUDGET_BYTES,
) -> int:
    image_h, image_w = image_size
    bytes_per_query = (
        image_h
        * image_w
        * torch.tensor([], dtype=torch.float32).element_size()
        * _FP32_WORKING_BUFFERS_PER_PIXEL
    )
    return max(1, memory_budget_bytes // bytes_per_query)


def get_instance_segmentation_prediction(
    *,
    mask_logits: Tensor,
    class_logits: Tensor,
    model_image_size: tuple[int, int],
    crop_size: tuple[int, int],
    image_size: tuple[int, int],
    get_labels_masks_scores: Callable[[Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
    memory_budget_bytes: int = _INSTANCE_PREDICTION_MEMORY_BUDGET_BYTES,
) -> dict[str, Tensor]:
    """Build one prediction while bounding resized-mask working memory.

    Queries are independent for bilinear interpolation and mask-score reduction,
    so processing them in chunks preserves the result. Masks are moved to CPU as
    soon as a chunk is complete because the metric and visualization store them
    there. Labels and scores stay on their original device for distributed sync.
    """
    if mask_logits.shape[0] == 0:
        raise ValueError("mask_logits must contain at least one query")

    crop_h, crop_w = crop_size
    # The first resize can be larger than the final image resize. Size chunks
    # from the largest materialized mask surface so the intermediate logits stay
    # within the same working-set budget.
    peak_image_size = max(
        (model_image_size, image_size), key=lambda size: size[0] * size[1]
    )
    chunk_size = _query_chunk_size(
        image_size=peak_image_size,
        memory_budget_bytes=memory_budget_bytes,
    )
    labels_chunks: list[Tensor] = []
    masks_chunks: list[Tensor] = []
    scores_chunks: list[Tensor] = []

    for start in range(0, mask_logits.shape[0], chunk_size):
        end = start + chunk_size
        logits_chunk = mask_logits[start:end].unsqueeze(0)
        class_logits_chunk = class_logits[start:end].unsqueeze(0)

        logits_chunk = F.interpolate(logits_chunk, model_image_size, mode="bilinear")
        logits_chunk = logits_chunk[..., :crop_h, :crop_w]
        logits_chunk = F.interpolate(logits_chunk, image_size, mode="bilinear")

        labels, masks, scores = get_labels_masks_scores(
            logits_chunk, class_logits_chunk
        )
        labels_chunks.append(labels[0])
        masks_chunks.append(masks[0].cpu())
        scores_chunks.append(scores[0])

    return {
        "labels": torch.cat(labels_chunks),
        "masks": torch.cat(masks_chunks),
        "scores": torch.cat(scores_chunks),
    }
