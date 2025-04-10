#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

import pytest
import torch
import torch.nn.functional as F
from pytest_mock import MockerFixture

from lightly_train._methods.distillation import distillation
from lightly_train._methods.distillation.distillation import (
    Distillation,
    DistillationArgs,
    DistillationLARSArgs,
)
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._scaling import ScalingInfo

from ... import helpers


class TestDistillationArgs:
    def test_resolve_auto_queue_size(self) -> None:
        """Test `resolve_auto` assigne correct queue size for a dataset of size 400."""
        args = DistillationArgs()

        # Set the dataset size to 400.
        scaling_info = ScalingInfo(dataset_size=400, epochs=1)

        # Infer the queue size.
        args.resolve_auto(
            scaling_info=scaling_info, optimizer_args=DistillationLARSArgs()
        )

        # The expected queue size is 128 and it is expected to be an int.
        assert args.queue_size == 128
        assert not isinstance(args.queue_size, str)
        assert not args.has_auto()

    def test_resolve_auto_queue_size_large_dataset(self) -> None:
        """Test `resolve_auto` handles large dataset sizes properly."""
        args = DistillationArgs()

        # Set the dataset size to 1e8.
        scaling_info = ScalingInfo(dataset_size=int(1e8), epochs=1)

        # Infer the queue size.
        args.resolve_auto(
            scaling_info=scaling_info, optimizer_args=DistillationLARSArgs()
        )

        # The expected queue size is 8192 and it is expected to be an int.
        assert args.queue_size == 8192
        assert not isinstance(args.queue_size, str)
        assert not args.has_auto()

    def test_too_large_queue_size(self) -> None:
        """Ensure an an error is raised when the queue size is manually set larger than dataset size."""
        args = DistillationArgs()

        # Manually set queue size larger than dataset size.
        args.queue_size = 1000
        scaling_info = ScalingInfo(dataset_size=500, epochs=1)

        # Check that an error is raised.
        with pytest.raises(ValueError, match="cannot be larger than the dataset size"):
            args.resolve_auto(
                scaling_info=scaling_info, optimizer_args=DistillationLARSArgs()
            )

    def test_resolve_auto_does_not_change_explicit_queue_size(self) -> None:
        """Ensure manually set queue size is not changed by `resolve_auto`."""
        args = DistillationArgs()

        # Manually set a queue size.
        args.queue_size = 512
        scaling_info = ScalingInfo(dataset_size=10_000, epochs=1)

        # Resolve auto values.
        args.resolve_auto(
            scaling_info=scaling_info, optimizer_args=DistillationLARSArgs()
        )

        # Verify that the queue size is unchanged.
        assert args.queue_size == 512
        assert not args.has_auto()


class TestDistillation:
    @pytest.mark.parametrize(
        "optim_type, expected",
        [
            ("auto", DistillationLARSArgs),
            (OptimizerType.LARS, DistillationLARSArgs),
        ],
    )
    def test_optimizer_args_cls(
        self, optim_type: OptimizerType | Literal["auto"], expected: type[OptimizerArgs]
    ) -> None:
        """Test optimizer argument class resolution."""

        assert Distillation.optimizer_args_cls(optim_type=optim_type) == expected

    def test_mixup_data_preserves_shape(self) -> None:
        """Test that mixup does not change the shape of the input tensor."""
        # Create dummy input images.
        x = torch.rand(2, 3, 16, 16)

        # Mix the images.
        mixed_x = Distillation._mixup_data(x)

        # Check that the images still have the same shape.
        assert mixed_x.shape == x.shape, (
            "Mixup should not change the shape of the tensor."
        )

    def test_mixup_data_with_fixed_seed(self) -> None:
        """Test that mixup is deterministic when using a fixed random seed."""
        # Create dummy input images.
        x = torch.rand(2, 3, 16, 16)

        # Mix the images a first time with a fixed seed.
        torch.manual_seed(42)
        mixed_x_1 = Distillation._mixup_data(x)

        # Mix the images a second time with the same seed.
        torch.manual_seed(42)
        mixed_x_2 = Distillation._mixup_data(x)

        # Verify that the result is the same.
        torch.testing.assert_close(mixed_x_1, mixed_x_2, atol=1e-6, rtol=1e-6)

    def test_mixup_with_binary_images(self) -> None:
        """Test that mixup correctly interpolates between binary images of all zeros and all ones."""
        batch_size = 8
        x = torch.cat(
            [
                torch.zeros(batch_size // 2, 3, 16, 16),
                torch.ones(batch_size // 2, 3, 16, 16),
            ],
            dim=0,
        )

        # Mix the images with a fixed seed.
        torch.manual_seed(42)
        mixed_x = Distillation._mixup_data(x)

        # Get the mixing value.
        torch.manual_seed(42)
        lambda_ = torch.empty(1).uniform_(0.0, 1.0).item()

        # Infer the expected values.
        expected_values = {0.0, lambda_, 1.0 - lambda_, 1.0}

        # Get the produced values.
        unique_values = set(mixed_x.unique().tolist())  # type: ignore

        # Verify that the produced values are correct.
        assert expected_values == unique_values, (
            "Mixup should only produce 0, 1, lambda and 1 - lambda when fed with binary images."
        )

    def test_queue_update(self, mocker: MockerFixture) -> None:
        """Test that the queue updates correctly when adding new teacher features."""
        # Set the queue and batch attributes.
        teacher_embed_dim = 16
        queue_size = 10
        batch_size = 2

        # Instantiate the distillation method.
        mock_automodel = mocker.patch.object(distillation, "AutoModel")
        mock_automodel.from_pretrained.return_value.layernorm.weight.shape = (
            teacher_embed_dim,
        )
        distill = Distillation(
            method_args=DistillationArgs(queue_size=queue_size),
            optimizer_args=DistillationLARSArgs(),
            embedding_model=helpers.get_embedding_model(),
            global_batch_size=batch_size,
        )
        mock_automodel.from_pretrained.assert_called_once()

        # Set the queue to be full of ones at the end.
        distill.teacher_queue[-batch_size:, :] = 1.0

        # Create a dummy batch.
        x_teacher = torch.randn(batch_size, teacher_embed_dim)
        x_teacher = F.normalize(x_teacher, dim=-1, p=2)

        # Update the queue with the latest batch.
        distill._update_queue(x_teacher)

        # Check that the first entries in the queue are identical to the batch.
        assert torch.allclose(
            distill.teacher_queue[:batch_size], x_teacher, atol=1e-6
        ), "Queue should be updated with new values at the beginning."

        # Check that the last entries in the queue are now zeroes.
        assert torch.allclose(
            distill.teacher_queue[-batch_size:],
            torch.zeros_like(x_teacher),
            atol=1e-6,
        ), "Queue should be updated with new values at the beginning."

        # Check that the number of non-zero rows is equal to the batch size and that the rows are normalized.
        assert distill.teacher_queue.norm(dim=-1).sum() == batch_size

    def test_teacher_queue_never_exceeds_capacity(self, mocker: MockerFixture) -> None:
        """Test that the teacher queue can handle large batches."""
        # Set the queue and batch attributes.
        teacher_embed_dim = 16
        queue_size = 10
        batch_size = 12

        # Instantiate the distillation method.
        mock_automodel = mocker.patch.object(distillation, "AutoModel")
        mock_automodel.from_pretrained.return_value.layernorm.weight.shape = (
            teacher_embed_dim,
        )
        distill = Distillation(
            method_args=DistillationArgs(queue_size=queue_size),
            optimizer_args=DistillationLARSArgs(),
            embedding_model=helpers.get_embedding_model(),
            global_batch_size=batch_size,
        )
        mock_automodel.from_pretrained.assert_called_once()

        # Initialize the queue with zeros except ones at the end.
        distill.teacher_queue = torch.zeros([queue_size, teacher_embed_dim])

        # Create a dummy batch.
        x_teacher = torch.randn(batch_size, teacher_embed_dim)
        x_teacher = F.normalize(x_teacher, dim=-1, p=2)

        # Update the queue with the latest batch.
        distill._update_queue(x_teacher)

        # Ensure queue size remains consistent.
        assert distill.teacher_queue.shape[0] == queue_size, (
            "Queue size should remain fixed and not exceed its predefined limit."
        )

        # Verify that the queue is filled with the first elements from the batch.
        assert torch.allclose(
            distill.teacher_queue, x_teacher[:queue_size], atol=1e-6
        ), "Queue shoud contain the first element from the batch."
