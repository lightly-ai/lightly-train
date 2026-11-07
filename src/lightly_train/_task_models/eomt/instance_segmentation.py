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


@torch.no_grad()
def get_labels_masks_scores(
    mask_logits: Tensor, class_logits: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    # Get score and label for each query.
    # NOTE(Guarin, 01/26): Calculate softmax always in fp32 for numerical
    # stability. Otherwise converting a fp32 model to fp16 results in NaNs.
    scores = class_logits.softmax(dim=-1, dtype=torch.float32)[..., :-1].type_as(
        class_logits
    )  # (1, Q, K)
    scores, labels = torch.max(scores, dim=-1)  # (1, Q), (1, Q)

    # Multiply scores with mask scores.
    masks = mask_logits > 0  # (1, Q, H, W)

    #### WARNING
    # Be careful when modifying anything here. These operations
    # must be kept in FP32 to avoid numerical issues in FP16 models. Test on
    # TensorRT FP16 when making any changes. You have to force TensorRT to use FP32.
    # See export_tensorrt in the EoMT instance segmentation task models for details.
    mask_logits_fp32 = mask_logits.float()
    masks_fp32 = masks.float()

    # (1, Q)
    score_per_query = (
        mask_logits_fp32.sigmoid().flatten(2) * masks_fp32.flatten(2)
    ).sum(2)
    mask_pixels_per_query = masks_fp32.flatten(2).sum(2)
    mask_scores = score_per_query / mask_pixels_per_query
    mask_scores = torch.where(
        mask_pixels_per_query > 0, mask_scores, torch.zeros_like(mask_scores)
    )

    scores = scores * mask_scores  # (1, Q)
    scores = scores.type_as(mask_logits)
    #####

    # (1, Q), (1, Q, H, W), (1, Q)
    return labels, masks, scores


def get_chunk_size(
    *,
    num_queries: int,
    resize_size: tuple[int, int],
    image_size: tuple[int, int],
) -> int:
    """Number of queries to score at once."""
    resize_pixels = resize_size[0] * resize_size[1]
    image_pixels = image_size[0] * image_size[1]
    # A chunk covers as many mask elements as training scores at once.
    return max(1, num_queries * resize_pixels // image_pixels)


def get_instance_segmentation_prediction(
    *,
    mask_logits: Tensor,
    class_logits: Tensor,
    resize_size: tuple[int, int],
    crop_size: tuple[int, int],
    image_size: tuple[int, int],
) -> dict[str, Tensor]:
    """Get the prediction for a single image.

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

    Returns:
        A dict with labels of shape (Q,), masks of shape (Q, H, W) on the CPU and
        scores of shape (Q,).
    """
    num_queries = mask_logits.shape[0]
    crop_h, crop_w = crop_size
    chunk_size = get_chunk_size(
        num_queries=num_queries,
        resize_size=resize_size,
        image_size=image_size,
    )
    labels_chunks: list[Tensor] = []
    scores_chunks: list[Tensor] = []
    # The metric and the visualization move the masks to the CPU anyway. Only
    # labels and scores must stay on the device for the distributed metric sync.
    masks = torch.empty((num_queries, *image_size), dtype=torch.bool)

    # Interpolation and scoring act on every query alone, so chunking does not
    # change the result.
    for start in range(0, num_queries, chunk_size):
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
        labels, masks_chunk, scores = get_labels_masks_scores(
            mask_logits=logits_chunk, class_logits=class_logits_chunk
        )
        labels_chunks.append(labels[0])
        masks[start:end].copy_(masks_chunk[0])
        scores_chunks.append(scores[0])

    return {
        "labels": torch.cat(labels_chunks),
        "masks": masks,
        "scores": torch.cat(scores_chunks),
    }
