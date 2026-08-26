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
    resize_size: tuple[int, int],
    crop_size: tuple[int, int] | None,
    output_size: tuple[int, int],
    get_labels_masks_scores: GetLabelsMasksScoresFn,
    max_chunk_bytes: int = 128 * 1024 * 1024,
) -> tuple[Tensor, Tensor, Tensor]:
    """Resize query masks in chunks to reduce peak validation memory.

    The mask logits are resized per query chunk to ``resize_size`` (the padded
    model input size). Afterwards the padded region is cropped with
    ``crop_size`` and the remaining logits are resized to ``output_size``
    before ``get_labels_masks_scores`` is called on each chunk. This matches
    the unchunked validation path while bounding the temporary memory of the
    resized mask logits.

    Args:
        mask_logits:
            Mask logits of shape (1, Q, H', W') at the patch grid resolution.
        class_logits:
            Class logits of shape (1, Q, num_classes).
        resize_size:
            The padded model input size the logits are resized to, e.g.
            ``self.model.image_size``.
        crop_size:
            The ``(crop_h, crop_w)`` size of the non-padded image region in
            ``resize_size`` coordinates, as returned by
            ``resize_and_pad``. Pass ``None`` to skip cropping.
        output_size:
            The original image size ``(image_h, image_w)`` the cropped logits
            are resized to.
        get_labels_masks_scores:
            Function returning ``(labels, masks, scores)`` for a chunk of
            resized mask logits and class logits.
        max_chunk_bytes:
            Soft byte budget for the resized mask logits of a single chunk.
    """
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

    # The peak memory is dominated by the resize to the padded model input
    # size, which is usually much larger than the cropped and final sizes.
    resize_h, resize_w = resize_size
    if crop_size is not None:
        resize_h = max(resize_h, crop_size[0])
        resize_w = max(resize_w, crop_size[1])
    resize_h = max(resize_h, output_size[0])
    resize_w = max(resize_w, output_size[1])

    num_queries = mask_logits.shape[1]
    bytes_per_query = mask_logits.element_size() * mask_logits.shape[0] * resize_h * resize_w
    chunk_size = max(1, max_chunk_bytes // bytes_per_query)

    labels_chunks: list[Tensor] = []
    masks_chunks: list[Tensor] = []
    scores_chunks: list[Tensor] = []
    for start in range(0, num_queries, chunk_size):
        end = min(start + chunk_size, num_queries)
        # Resize to the padded model input size.
        resized_mask_logits = F.interpolate(
            mask_logits[:, start:end], resize_size, mode="bilinear"
        )
        # Remove the padded region.
        if crop_size is not None:
            resized_mask_logits = resized_mask_logits[
                ..., : crop_size[0], : crop_size[1]
            ]
        # Resize to the original image size.
        resized_mask_logits = F.interpolate(
            resized_mask_logits, output_size, mode="bilinear"
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
