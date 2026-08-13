from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

GetLabelsMasksScoresFn = Callable[[Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]


def get_chunked_labels_masks_scores(
    *,
    mask_logits: Tensor,
    class_logits: Tensor,
    output_size: tuple[int, int],
    get_labels_masks_scores: GetLabelsMasksScoresFn,
    max_chunk_bytes: int = 128 * 1024 * 1024,
) -> tuple[Tensor, Tensor, Tensor]:
    """Resize query masks in chunks to reduce peak validation memory."""
    if mask_logits.ndim != 4:
        raise ValueError(f"Expected mask_logits with 4 dimensions, got {mask_logits.ndim}.")
    if class_logits.ndim != 3:
        raise ValueError(
            f"Expected class_logits with 3 dimensions, got {class_logits.ndim}."
        )
    if mask_logits.shape[:2] != class_logits.shape[:2]:
        raise ValueError(
            "mask_logits and class_logits must agree on batch size and query count."
        )

    num_queries = mask_logits.shape[1]
    if num_queries == 0:
        resized_mask_logits = F.interpolate(mask_logits, output_size, mode="bilinear")
        return get_labels_masks_scores(resized_mask_logits, class_logits)

    bytes_per_query = (
        mask_logits.element_size()
        * mask_logits.shape[0]
        * output_size[0]
        * output_size[1]
    )
    chunk_size = max(1, max_chunk_bytes // bytes_per_query)

    if chunk_size >= num_queries:
        resized_mask_logits = F.interpolate(mask_logits, output_size, mode="bilinear")
        return get_labels_masks_scores(resized_mask_logits, class_logits)

    labels_chunks: list[Tensor] = []
    masks_chunks: list[Tensor] = []
    scores_chunks: list[Tensor] = []
    for start in range(0, num_queries, chunk_size):
        end = min(start + chunk_size, num_queries)
        resized_mask_logits = F.interpolate(
            mask_logits[:, start:end], output_size, mode="bilinear"
        )
        labels, masks, scores = get_labels_masks_scores(
            resized_mask_logits,
            class_logits[:, start:end],
        )
        labels_chunks.append(labels)
        masks_chunks.append(masks)
        scores_chunks.append(scores)

    return (
        torch.cat(labels_chunks, dim=1),
        torch.cat(masks_chunks, dim=1),
        torch.cat(scores_chunks, dim=1),
    )
