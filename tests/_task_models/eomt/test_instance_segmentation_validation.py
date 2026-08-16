from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from lightly_train._task_models.eomt.instance_segmentation_validation import (
    get_chunked_labels_masks_scores,
)


def test_get_chunked_labels_masks_scores_matches_unchunked() -> None:
    torch.manual_seed(0)
    mask_logits = torch.randn(1, 4, 3, 5)
    class_logits = torch.randn(1, 4, 6)
    output_size = (7, 9)

    chunk_call_count = 0

    def chunked_get_labels_masks_scores(
        mask_logits: Tensor, class_logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        nonlocal chunk_call_count
        chunk_call_count += 1
        labels = class_logits.argmax(dim=-1)
        masks = mask_logits.sigmoid()
        scores = class_logits.softmax(dim=-1).amax(dim=-1)
        return labels, masks, scores

    def reference_get_labels_masks_scores(
        mask_logits: Tensor, class_logits: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        labels = class_logits.argmax(dim=-1)
        masks = mask_logits.sigmoid()
        scores = class_logits.softmax(dim=-1).amax(dim=-1)
        return labels, masks, scores

    max_chunk_bytes = mask_logits.element_size() * output_size[0] * output_size[1]
    chunked = get_chunked_labels_masks_scores(
        mask_logits=mask_logits,
        class_logits=class_logits,
        output_size=output_size,
        get_labels_masks_scores=chunked_get_labels_masks_scores,
        max_chunk_bytes=max_chunk_bytes,
    )
    assert chunk_call_count == mask_logits.shape[1]

    expected = reference_get_labels_masks_scores(
        F.interpolate(mask_logits, output_size, mode="bilinear"),
        class_logits,
    )

    for actual, expected_part in zip(chunked, expected):
        torch.testing.assert_close(actual, expected_part)
