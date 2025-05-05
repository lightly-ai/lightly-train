#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch import Tensor
from torch.nn import Module, MSELoss


class DistillationV2Loss(Module):
    """
    Computes the Mean Squared Error (MSE) loss used for DistillationV2.

    The loss directly compares the student and teacher representations using the
    MSE loss function. The student and teacher features are not normalized.

    """

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.mse_loss = MSELoss()

    def forward(self, teacher_features: Tensor, student_features: Tensor) -> Tensor:
        """Computes the MSE loss between the student and teacher features.

        Args:
            teacher_features: Tensor containing teacher representations from the current batch.
                The expected shape is (batch_size, n_features, feature_dim).
                n_features is the number of tokens per sequence in the teacher model.
            student_features: Tensor containing student representations from the current batch.
                The expected shape is (batch_size, n_features, feature_dim).

        Returns:
            MSE loss as a scalar tensor.
        """
        # Compute the loss.
        loss: Tensor = self.mse_loss(teacher_features, student_features)
        return loss
