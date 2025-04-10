#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# test_distillation_loss.py

import torch
import torch.nn.functional as F
from torch import Tensor

from lightly_train._methods.distillation.distillation_loss import DistillationLoss


class TestDistillationLoss:
    def test_loss_returns_scalar(self) -> None:
        """Test that the loss function returns a scalar tensor."""
        # Instantiate the loss function.
        temperature = 0.07
        loss_fn = DistillationLoss(temperature=temperature)

        # Create some dummy inputs.
        batch_size, feature_dim, queue_size = 2, 4, 8
        teacher_features = torch.randn(batch_size, feature_dim)
        student_features = torch.randn(batch_size, feature_dim)
        queue = torch.randn(queue_size, feature_dim)

        # Normalize the inputs.
        teacher_features = F.normalize(teacher_features, p=2, dim=1)
        student_features = F.normalize(student_features, p=2, dim=1)
        queue = F.normalize(queue, p=2, dim=1)

        # Compute the loss.
        loss = loss_fn(teacher_features, student_features, queue)

        # Check that loss is a Tensor.
        assert isinstance(loss, Tensor)

        # Check that loss is a scalar (0-dim).
        assert loss.dim() == 0

    def test_loss_zero_when_identical(self) -> None:
        """Test that loss is nearly zero when teacher and student features are identical."""
        # Instantiate the loss function.
        temperature = 0.07
        loss_fn = DistillationLoss(temperature=temperature)

        # Create some dummy inputs with identical teacher and student features.
        batch_size, feature_dim, queue_size = 2, 4, 8
        teacher_features = torch.randn(batch_size, feature_dim)
        student_features = teacher_features.clone()
        queue = torch.randn(queue_size, feature_dim)

        # Normalize the inputs.
        teacher_features = F.normalize(teacher_features, p=2, dim=1)
        student_features = F.normalize(student_features, p=2, dim=1)
        queue = F.normalize(queue, p=2, dim=1)

        # Compute the loss.
        loss = loss_fn(teacher_features, student_features, queue)

        # When the features are identical, the KL divergence loss should be close to 0.
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_loss_non_negative(self) -> None:
        """Test that the computed loss is non-negative."""
        # Instantiate the loss function.
        temperature = 0.07
        loss_fn = DistillationLoss(temperature=temperature)

        # Create some dummy inputs.
        batch_size, feature_dim, queue_size = 2, 4, 8
        teacher_features = torch.randn(batch_size, feature_dim)
        student_features = torch.randn(batch_size, feature_dim)
        queue = torch.randn(queue_size, feature_dim)

        # Normalize the inputs.
        teacher_features = F.normalize(teacher_features, p=2, dim=1)
        student_features = F.normalize(student_features, p=2, dim=1)
        queue = F.normalize(queue, p=2, dim=1)

        # Compute the loss.
        loss = loss_fn(teacher_features, student_features, queue)

        # KL divergence loss should be non-negative.
        assert loss >= 0
