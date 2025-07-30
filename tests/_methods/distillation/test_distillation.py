#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import pytest
import torch
import torch.nn.functional as F
from pytest_mock import MockerFixture

import lightly_train
from lightly_train._methods.distillation.distillation import (
    Distillation,
    DistillationArgs,
    DistillationLARSArgs,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._scaling import ScalingInfo

from ...helpers import DummyCustomModel, create_images, dummy_vit_model


class TestDistillationArgs:
    def test_resolve_auto_queue_size(self) -> None:
        """Test `resolve_auto` assigne correct queue size for a dataset of size 400."""
        args = DistillationArgs()

        # Set the dataset size to 400.
        scaling_info = ScalingInfo(dataset_size=400, epochs=1)

        # Infer the queue size.
        args.resolve_auto(
            scaling_info=scaling_info,
            optimizer_args=DistillationLARSArgs(),
            wrapped_model=DummyCustomModel(),
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
            scaling_info=scaling_info,
            optimizer_args=DistillationLARSArgs(),
            wrapped_model=DummyCustomModel(),
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
                scaling_info=scaling_info,
                optimizer_args=DistillationLARSArgs(),
                wrapped_model=DummyCustomModel(),
            )

    def test_resolve_auto_does_not_change_explicit_queue_size(self) -> None:
        """Ensure manually set queue size is not changed by `resolve_auto`."""
        args = DistillationArgs()

        # Manually set a queue size.
        args.queue_size = 512
        scaling_info = ScalingInfo(dataset_size=10_000, epochs=1)

        # Resolve auto values.
        args.resolve_auto(
            scaling_info=scaling_info,
            optimizer_args=DistillationLARSArgs(),
            wrapped_model=DummyCustomModel(),
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

        teacher_model = EmbeddingModel(
            wrapped_model=dummy_vit_model(), embed_dim=teacher_embed_dim
        ).eval()

        # Mock the getter of the teacher model.
        mock_get_teacher_model = mocker.patch(
            "lightly_train._methods.distillation.distillation.get_teacher"
        )
        mock_get_teacher_model.return_value = teacher_model

        # Instantiate the distillation method.
        distill = Distillation(
            method_args=DistillationArgs(queue_size=queue_size),
            optimizer_args=DistillationLARSArgs(),
            embedding_model=EmbeddingModel(wrapped_model=DummyCustomModel()),
            global_batch_size=batch_size,
        )
        mock_get_teacher_model.assert_called_once()

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

        teacher_model = EmbeddingModel(
            wrapped_model=dummy_vit_model(), embed_dim=teacher_embed_dim
        ).eval()

        # Mock the getter of the teacher model.
        mock_get_teacher_model = mocker.patch(
            "lightly_train._methods.distillation.distillation.get_teacher"
        )
        mock_get_teacher_model.return_value = teacher_model

        # Instantiate the distillation method.
        distill = Distillation(
            method_args=DistillationArgs(queue_size=queue_size),
            optimizer_args=DistillationLARSArgs(),
            embedding_model=EmbeddingModel(wrapped_model=DummyCustomModel()),
            global_batch_size=batch_size,
        )
        mock_get_teacher_model.assert_called_once()

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

    def test_load_state_dict_from_pretrained_teacher(
        self, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """Test that the distillation method can load a state dict from a pretrained teacher model from DINOv2."""

        # Create a temporary directory for the test.
        out_path = tmp_path / "out"
        data_path = tmp_path / "data"
        create_images(data_path, files=4, height=224, width=224)

        # export the pretrained teacher model from DINOv2.
        lightly_train.train(
            out=out_path,
            data=data_path,
            method="dinov2",
            model="dinov2/_vittest14",
            transform_args={"image_size": (224, 224)},
            epochs=0,
            batch_size=4,
            accelerator="cpu",
        )

        # Setup constants.
        batch_size = 2
        student_embed_dim = 32
        queue_size = 10

        # Dummy student model with real params.
        student_model = EmbeddingModel(
            wrapped_model=DummyCustomModel(student_embed_dim)
        )

        # Instantiate the distillation method.
        _ = Distillation(
            method_args=DistillationArgs(
                queue_size=queue_size,
                teacher="dinov2/_vittest14",
                teacher_weights=f"{out_path}/exported_models/exported_last.pt",
            ),
            optimizer_args=DistillationLARSArgs(),
            embedding_model=student_model,
            global_batch_size=batch_size,
        )

    def test_load_state_dict_ignores_missing_teacher_keys(
        self, mocker: MockerFixture
    ) -> None:
        """Test that missing teacher keys in the state dict do not raise an error."""

        # Setup constants
        batch_size = 2
        student_embed_dim = 32
        teacher_embed_dim = 48
        queue_size = 10

        # Dummy student model with real params.
        student_model = EmbeddingModel(
            wrapped_model=DummyCustomModel(student_embed_dim)
        )

        # Dummy teacher model with real params.
        teacher_model = EmbeddingModel(
            wrapped_model=dummy_vit_model(), embed_dim=teacher_embed_dim
        ).eval()

        # Patch get_teacher.
        mock_get_teacher = mocker.patch(
            "lightly_train._methods.distillation.distillation.get_teacher"
        )
        mock_get_teacher.return_value = teacher_model

        # Instantiate the distillation method.
        distill = Distillation(
            method_args=DistillationArgs(queue_size=queue_size),
            optimizer_args=DistillationLARSArgs(),
            embedding_model=student_model,
            global_batch_size=batch_size,
        )
        mock_get_teacher.assert_called_once()

        # Check that teacher keys are present in the statedict.
        full_state_dict = distill.state_dict()
        assert any(k.startswith("teacher_embedding_model.") for k in full_state_dict)

        # Simulate a checkpoint with teacher keys removed.
        filtered_state_dict = {
            k: v
            for k, v in full_state_dict.items()
            if not k.startswith("teacher_embedding_model.")
        }

        # Try loading the statedict without teacher keys.
        distill.load_state_dict(filtered_state_dict, strict=True)

    def test_load_state_dict_raises_on_non_teacher_missing_key(
        self, mocker: MockerFixture
    ) -> None:
        """Test that load_state_dict raises an error if non-teacher keys are missing."""

        # Setup constants.
        batch_size = 2
        student_embed_dim = 32
        teacher_embed_dim = 48
        queue_size = 10

        # Dummy student model with real params.
        student_model = EmbeddingModel(
            wrapped_model=DummyCustomModel(student_embed_dim)
        )

        # Dummy teacher model with real params.
        teacher_model = EmbeddingModel(
            wrapped_model=dummy_vit_model(), embed_dim=teacher_embed_dim
        ).eval()

        # Patch get_teacher.
        mock_get_teacher = mocker.patch(
            "lightly_train._methods.distillation.distillation.get_teacher"
        )
        mock_get_teacher.return_value = teacher_model

        # Create distillation instance
        distill = Distillation(
            method_args=DistillationArgs(queue_size=queue_size),
            optimizer_args=DistillationLARSArgs(),
            embedding_model=student_model,
            global_batch_size=batch_size,
        )
        mock_get_teacher.assert_called_once()

        # Obtain the full statedict.
        state_dict = distill.state_dict()

        # Ensure non-teacher keys are present.
        non_teacher_keys = [
            k for k in state_dict if not k.startswith("teacher_embedding_model.")
        ]
        assert non_teacher_keys, "No non-teacher keys found in state dict for testing."

        # Remove a non-teacher key
        key_to_remove = non_teacher_keys[0]
        state_dict.pop(key_to_remove)

        # Assert that a RuntimeError is raised due to unexpected missing keys
        with pytest.raises(RuntimeError, match="Missing keys in state_dict"):
            distill.load_state_dict(state_dict, strict=True)

    def test_teacher_not_saved_in_checkpoint(self, mocker: MockerFixture) -> None:
        """Test that the teacher model is not saved in the checkpoint."""

        # Setup constants.
        batch_size = 2
        student_embed_dim = 32
        teacher_embed_dim = 48
        queue_size = 10

        # Dummy student model with real params.
        student_model = EmbeddingModel(
            wrapped_model=DummyCustomModel(student_embed_dim)
        )

        # Dummy teacher model with real params.
        teacher_model = EmbeddingModel(
            wrapped_model=dummy_vit_model(), embed_dim=teacher_embed_dim
        ).eval()

        # Patch get_teacher.
        mock_get_teacher = mocker.patch(
            "lightly_train._methods.distillation.distillation.get_teacher"
        )
        mock_get_teacher.return_value = teacher_model

        # Create distillation instance
        distill = Distillation(
            method_args=DistillationArgs(queue_size=queue_size),
            optimizer_args=DistillationLARSArgs(),
            embedding_model=student_model,
            global_batch_size=batch_size,
        )
        mock_get_teacher.assert_called_once()

        # Simulate saving the checkpoint
        checkpoint = {"state_dict": distill.state_dict()}

        # Check that teacher keys are initially present in the checkpoint.
        teacher_keys = [
            k
            for k in checkpoint["state_dict"]
            if k.startswith("teacher_embedding_model.")
        ]
        assert len(teacher_keys) > 0, (
            "Teacher weights should initially be in the checkpoint."
        )

        # Strip teacher keys from the checkpoint.
        distill.on_save_checkpoint(checkpoint)

        # Assert that no teacher weights are in the checkpoint
        teacher_keys = [
            k
            for k in checkpoint["state_dict"]
            if k.startswith("teacher_embedding_model.")
        ]
        assert len(teacher_keys) == 0, (
            "Teacher weights should not be saved in the final checkpoint."
        )

    @pytest.mark.parametrize(
        "global_batch_size, expected_lr",
        [
            (3072, 1.8 * math.sqrt(3072 / 1536)),  # scaling = sqrt(2)
            (1536, 1.8),  # scaling = 1.0
            (768, 1.8 * math.sqrt(768 / 1536)),  # scaling = sqrt(0.5)
            (384, 1.8 * math.sqrt(384 / 1536)),  # scaling = sqrt(0.25)
            (128, 1.8 * math.sqrt(128 / 1536)),  # scaling = sqrt(1/12)
        ],
    )
    def test_distillation_configure_optimizers_lr_scaling(
        self,
        mocker: MockerFixture,
        global_batch_size: int,
        expected_lr: float,
    ) -> None:
        """Test that the effective learning rate scales correctly with global_batch_size."""

        # Constants.
        student_embed_dim = 32
        teacher_embed_dim = 48
        queue_size = 10
        base_lr = 0.3 * 1536 / 256

        # Dummy student model.
        student_model = EmbeddingModel(
            wrapped_model=DummyCustomModel(student_embed_dim)
        )

        # Dummy teacher model.
        teacher_model = EmbeddingModel(
            wrapped_model=DummyCustomModel(teacher_embed_dim)
        )

        # Patch get_teacher.
        mock_get_teacher = mocker.patch(
            "lightly_train._methods.distillation.distillation.get_teacher"
        )
        mock_get_teacher.return_value = teacher_model

        # Instantiate distillation method.
        distill = Distillation(
            method_args=DistillationArgs(queue_size=queue_size),
            optimizer_args=DistillationLARSArgs(lr=base_lr),
            embedding_model=student_model,
            global_batch_size=global_batch_size,
        )

        # Mock trainer attributes needed by configure_optimizers.
        mock_trainer = mocker.Mock()
        mock_trainer.max_epochs = 100
        mock_trainer.estimated_stepping_batches = 1000
        distill.trainer = mock_trainer

        # Call configure_optimizers.
        optimizers_schedulers = distill.configure_optimizers()

        # Check we got a tuple (optimizers, schedulers).
        assert isinstance(optimizers_schedulers, tuple), (
            f"Expected tuple from configure_optimizers, got {type(optimizers_schedulers)}"
        )

        optimizers, _ = optimizers_schedulers

        # Check that optimizers is a list.
        assert isinstance(optimizers, list), (
            f"Expected list of optimizers, got {type(optimizers)}"
        )

        # There should be exactly one optimizer.
        assert len(optimizers) == 1
        optimizer = optimizers[0]

        # Verify that all parameter groups have the expected scaled learning rate.
        for param_group in optimizer.param_groups:
            assert param_group["initial_lr"] == pytest.approx(expected_lr, rel=1e-6), (
                f"Expected learning rate {expected_lr}, but got {param_group['initial_lr']}."
            )
