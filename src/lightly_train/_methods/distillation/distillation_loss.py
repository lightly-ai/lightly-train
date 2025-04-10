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
from torch.nn import KLDivLoss, Module
from torch.nn import functional as F


class DistillationLoss(Module):
    """
    Computes the distillation loss based on Kullback-Leibler (KL) divergence.

    This loss function utilizes pseudo classification weights obtained by collecting
    L2-normalized teacher features in a queue. Both teacher and student features are
    L2-normalized and projected onto the queue to generate logits. The KL divergence
    between the student and teacher distributions is then computed by applying a softmax.

    A temperature parameter is used to control the sharpness of the probability
    distribution.
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.kl_divergence = KLDivLoss(reduction="batchmean", log_target=False)

    def forward(
        self, teacher_features: Tensor, student_features: Tensor, queue: Tensor
    ) -> Tensor:
        """Computes the KL divergence between the student and teacher distributions.
        All inputs are expected to be L2-normalized.

        Args:
            teacher_features: Tensor containing teacher representations from the current batch.
                The expected shape is (batch_size, feature_dim).
            student_features: Tensor containing student representations from the current batch.
                The expected shape is (batch_size, feature_dim).
            queue: Tensor containing teacher representations from the current and previous batches.
                The expected shape is (queue_size, feature_dim).

        Returns:
            KL divergence loss as a scalar tensor.
        """
        # Compute the teacher-student similarity.
        student_queue_similarity = torch.einsum(
            "b d, c d -> b c", student_features, queue
        )

        # Compute the teacher-teacher similarity.
        teacher_queue_similarity = torch.einsum(
            "b d, c d -> b c", teacher_features, queue
        )

        # Compute the teacher distribution
        teacher_distribution = F.softmax(
            teacher_queue_similarity / self.temperature, dim=-1
        )

        # Compute the student log-distribution
        student_log_distribution = F.log_softmax(
            student_queue_similarity / self.temperature, dim=-1
        )

        # Compute the loss.
        loss: Tensor = self.kl_divergence(
            student_log_distribution, teacher_distribution
        )
        return loss
