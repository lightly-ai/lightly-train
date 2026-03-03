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


class DistillationV3Loss(Module):
    """
    Computes the DistillationV3 loss based on Kullback-Leibler (KL) divergence.

    This loss function utilizes pseudo classification weights obtained by collecting
    L2-normalized teacher features in a queue. Both teacher and student features are
    L2-normalized and projected onto the queue to generate logits. The KL divergence
    between the student and teacher distributions is then computed by applying a softmax.

    A temperature parameter is used to control the sharpness of the probability
    distribution.
    """

    def __init__(self, temperature_global: float, temperature_local: float):
        super().__init__()
        self.temperature_global = temperature_global
        self.temperature_local = temperature_local
        self.kl_divergence = KLDivLoss(reduction="batchmean", log_target=False)

    def forward(
        self,
        teacher_features_global: Tensor,
        teacher_features_local: Tensor,
        student_features_global: Tensor,
        student_features_local: Tensor,
        queue: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Computes the KL divergence between the student and teacher distributions.
        All inputs are expected to be L2-normalized.

        Args:
            teacher_features_global: Tensor containing teacher global representations from the current batch.
                The expected shape is (batch_size, feature_dim).
            teacher_features_local: Tensor containing teacher local representations from the current batch.
                The expected shape is (batch_size, n_tokens, feature_dim).
            student_features_global: Tensor containing student global representations from the current batch.
                The expected shape is (batch_size, feature_dim).
            student_features_local: Tensor containing student local representations from the current batch.
                The expected shape is (batch_size, n_tokens, feature_dim).
            queue: Tensor containing teacher representations from the current and previous batches.
                The expected shape is (queue_size, feature_dim).

        Returns:
            KL divergence global and local losses as scalar tensors.
        """
        # Compute the teacher-student similarity.
        student_queue_similarity = torch.einsum(
            "b d, c d -> b c", student_features_global, queue
        )

        # Compute the teacher-teacher inter-image similarity.
        teacher_queue_similarity = torch.einsum(
            "b d, c d -> b c", teacher_features_global, queue
        )

        # Compute the teacher inter-image distribution.
        teacher_distribution_inter_image = F.softmax(
            teacher_queue_similarity / self.temperature_global, dim=-1
        )

        # Compute the student log-distribution
        student_log_distribution_inter_image = F.log_softmax(
            student_queue_similarity / self.temperature_global, dim=-1
        )

        # Compute the global loss.
        global_loss: Tensor = self.kl_divergence(
            student_log_distribution_inter_image, teacher_distribution_inter_image
        )

        # Compute the teacher-teacher intra-image similarity.
        teacher_teacher_similarity = torch.einsum(
            "b m d, b n d -> b m n", teacher_features_local, teacher_features_local
        )
        teacher_teacher_similarity = teacher_teacher_similarity.flatten(
            start_dim=0, end_dim=1
        )

        # Compute the student-student intra-image similarity.
        student_student_similarity = torch.einsum(
            "b m d, b n d -> b m n", student_features_local, student_features_local
        )
        student_student_similarity = student_student_similarity.flatten(
            start_dim=0, end_dim=1
        )

        # Compute the teacher intra-image distribution
        teacher_distribution_intra_image = F.softmax(
            teacher_teacher_similarity / self.temperature_local, dim=-1
        )

        # Compute the student log-distribution
        student_log_distribution_intra_image = F.log_softmax(
            student_student_similarity / self.temperature_local, dim=-1
        )

        # Compute the local loss.
        local_loss: Tensor = self.kl_divergence(
            student_log_distribution_intra_image, teacher_distribution_intra_image
        )

        return global_loss, local_loss
