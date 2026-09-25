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


def get_chunk_size(
    *,
    num_queries: int,
    resize_size: tuple[int, int],
    image_size: tuple[int, int],
) -> int:
    """Number of queries to score at once.

    Training scores all queries at the model input size. The chunk covers the same
    number of mask elements at the original image size.
    """
    resize_pixels = resize_size[0] * resize_size[1]
    image_pixels = image_size[0] * image_size[1]
    return max(1, num_queries * resize_pixels // image_pixels)


def get_instance_segmentation_prediction(
    *,
    mask_logits: Tensor,
    class_logits: Tensor,
    resize_size: tuple[int, int],
    crop_size: tuple[int, int],
    image_size: tuple[int, int],
    get_labels_masks_scores: Callable[[Tensor, Tensor], tuple[Tensor, Tensor, Tensor]],
) -> dict[str, Tensor]:
    """Get the prediction for a single image, scoring the queries in chunks.

    Bilinear interpolation and the mask score reduction act on every query alone, so
    a chunk boundary does not change the result.

    Args:
        mask_logits:
            Mask logits of shape (Q, H', W').
        class_logits:
            Class logits of shape (Q, num_classes).
        resize_size:
            Size the logits are resized to before cropping, usually
            self.model.image_size.
        crop_size:
            Size of the non-padded region, as returned by resize_and_pad.
        image_size:
            Original image size.
        get_labels_masks_scores:
            Called per chunk with batched mask and class logits.

    Returns:
        A dict with labels of shape (Q,), masks of shape (Q, H, W) on the CPU and
        scores of shape (Q,).
    """
    crop_h, crop_w = crop_size
    chunk_size = get_chunk_size(
        num_queries=mask_logits.shape[0],
        resize_size=resize_size,
        image_size=image_size,
    )
    labels_chunks: list[Tensor] = []
    masks_chunks: list[Tensor] = []
    scores_chunks: list[Tensor] = []

    for start in range(0, mask_logits.shape[0], chunk_size):
        end = start + chunk_size
        logits_chunk = mask_logits[start:end].unsqueeze(0)  # (1, C, H', W')
        class_logits_chunk = class_logits[start:end].unsqueeze(0)  # (1, C, num_classes)
        # Resize to same size as before passing through the model. This is usually
        # (1, C, 640, 640) and depends on self.model.image_size.
        logits_chunk = F.interpolate(logits_chunk, resize_size, mode="bilinear")
        # Revert resize and pad from self.model.resize_and_pad.
        logits_chunk = logits_chunk[..., :crop_h, :crop_w]  # (1, C, crop_h, crop_w)
        logits_chunk = F.interpolate(logits_chunk, image_size, mode="bilinear")
        # (1, C), (1, C, H, W), (1, C)
        labels, masks, scores = get_labels_masks_scores(
            logits_chunk, class_logits_chunk
        )
        labels_chunks.append(labels[0])
        # The metric and the visualization move the masks to the CPU anyway. Only
        # labels and scores must stay on the device for the distributed metric sync.
        masks_chunks.append(masks[0].cpu())
        scores_chunks.append(scores[0])

    return {
        "labels": torch.cat(labels_chunks),
        "masks": torch.cat(masks_chunks),
        "scores": torch.cat(scores_chunks),
    }
