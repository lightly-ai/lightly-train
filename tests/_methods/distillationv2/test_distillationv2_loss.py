#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from torch import Tensor

from lightly_train._methods.distillationv2.distillationv2_loss import DistillationV2Loss


class TestDistillationV2Loss:
    def test_loss_returns_scalar(self) -> None:
        """Test that the loss function returns a scalar tensor."""
        # Instantiate the loss function.
        loss_fn = DistillationV2Loss()

        # Create some dummy inputs.
        batch_size, n_features, feature_dim = 2, 4, 8
        teacher_features = torch.randn(batch_size, n_features, feature_dim)
        student_features = torch.randn(batch_size, n_features, feature_dim)

        # Compute the loss.
        loss = loss_fn(teacher_features, student_features)

        # Check that loss is a Tensor.
        assert isinstance(loss, Tensor)

        # Check that loss is a scalar (0-dim).
        assert loss.dim() == 0

    def test_loss_zero_when_identical(self) -> None:
        """Test that loss is nearly zero when teacher and student features are identical."""
        # Instantiate the loss function.
        loss_fn = DistillationV2Loss()

        # Create some dummy inputs with identical teacher and student features.
        batch_size, n_features, feature_dim = 2, 4, 8
        teacher_features = torch.randn(batch_size, n_features, feature_dim)
        student_features = teacher_features.clone()

        # Compute the loss.
        loss = loss_fn(teacher_features, student_features)

        # When the features are identical, the MSE loss should be close to 0.
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_loss_non_negative(self) -> None:
        """Test that the computed loss is non-negative."""
        # Instantiate the loss function.
        loss_fn = DistillationV2Loss()

        # Create some dummy inputs.
        batch_size, n_features, feature_dim = 2, 4, 8
        teacher_features = torch.randn(batch_size, n_features, feature_dim)
        student_features = torch.randn(batch_size, n_features, feature_dim)

        # Compute the loss.
        loss = loss_fn(teacher_features, student_features)

        # MSE loss should be non-negative.
        assert loss >= 0
