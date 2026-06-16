#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch import Tensor
from torch.nn import Module


class SemanticSegmentationSoftMaxPostProcessor(Module):
    """Post processor for semantic segmentation that applies a softmax to the output of the model.

    This is useful for models that output logits, and we want to convert them to probabilities.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Applies a softmax to the input tensor.

        Args:
            x: The input tensor of shape (batch_size, num_classes, height, width).

        Returns:
            The output tensor of shape (batch_size, num_classes, height, width) with softmax applied.
        """
        return x.softmax(dim=1)
