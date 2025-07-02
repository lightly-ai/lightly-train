#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class DINOv2SemanticSegmentationCrossEntropyLoss(Module):
    """
    Computes pixel-wise cross-entropy loss for semantic segmentation.

    Expects model logits of shape (B, num_classes, H, W) and ground-truth
    masks of shape (B, H, W) containing integer class indices.

    The loss ignores the specified ignore_index pixels.

    Args:
        ignore_index: Specifies a target value that is ignored and does
        not contribute to the input gradient.
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: Tensor of shape (B, num_classes, H, W)
                with raw model logits (before softmax).
            targets: Tensor of shape (B, H, W) with class indices
                in the range [0, num_classes - 1].

        Returns:
            Scalar tensor representing the pixel-wise cross-entropy loss.
        """
        return F.cross_entropy(
            logits,
            targets.long(),
            ignore_index=self.ignore_index,
        )
