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
from pytest_mock import MockerFixture

from lightly_train._methods.distillationv2.distillationv2 import (
    DistillationV2,
    DistillationV2Args,
    DistillationV2LARSArgs,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType

from ...helpers import DummyCustomModel


class TestDistillationV2:
    @pytest.mark.parametrize(
        "optim_type, expected",
        [
            ("auto", DistillationV2LARSArgs),
            (OptimizerType.LARS, DistillationV2LARSArgs),
        ],
    )
    def test_optimizer_args_cls(
        self, optim_type: OptimizerType | Literal["auto"], expected: type[OptimizerArgs]
    ) -> None:
        """Test optimizer argument class resolution."""

        assert DistillationV2.optimizer_args_cls(optim_type=optim_type) == expected

    def test_mixup_data_preserves_shape(self) -> None:
        """Test that mixup does not change the shape of the input tensor."""
        # Create dummy input images.
        x = torch.rand(2, 3, 16, 16)

        # Mix the images.
        mixed_x = DistillationV2._mixup_data(x)

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
        mixed_x_1 = DistillationV2._mixup_data(x)

        # Mix the images a second time with the same seed.
        torch.manual_seed(42)
        mixed_x_2 = DistillationV2._mixup_data(x)

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
        mixed_x = DistillationV2._mixup_data(x)

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

    def test_forward_student_output_shape(self, mocker: MockerFixture) -> None:
        """Test that _forward_student returns expected shape."""
        # Set constants.
        batch_size, channels, height, width = 2, 3, 224, 224
        student_embed_dim = 32
        teacher_embed_dim = 48
        n_blocks = 2
        patch_size = 14
        n_tokens = (height // patch_size) * (width // patch_size)

        # Create dummy images.
        x = torch.randn(batch_size, channels, height, width)

        # Patch the teacher model.
        mock_teacher_model = mocker.Mock()
        mock_teacher_model.embed_dim = teacher_embed_dim
        mock_teacher_model.patch_size = patch_size
        mock_teacher_model.get_intermediate_layers.return_value = [
            torch.randn(
                batch_size,
                n_tokens,
                teacher_embed_dim,
            )
            for _ in range(n_blocks)
        ]
        mock_get_teacher = mocker.patch(
            "lightly_train._methods.distillationv2.distillationv2.get_teacher"
        )
        mock_get_teacher.return_value = mock_teacher_model

        # Patch the student embedding model.
        mock_student_model = mocker.Mock()
        mock_student_model.embed_dim = student_embed_dim
        mock_student_model.return_value = torch.randn(
            batch_size, student_embed_dim, 7, 7
        )

        # Init distillation method.
        distill = DistillationV2(
            method_args=DistillationV2Args(n_teacher_blocks=n_blocks),
            optimizer_args=DistillationV2LARSArgs(),
            embedding_model=mock_student_model,
            global_batch_size=batch_size,
        )
        mock_get_teacher.assert_called_once()

        # Run _forward_student.
        out = distill._forward_student(x)

        # Expected shape: (batch_size, n_tokens, teacher_embedding_dim).
        assert out.shape == (batch_size, n_tokens, distill.teacher_embedding_dim)

    def test_load_state_dict_ignores_missing_teacher_keys(
        self, mocker: MockerFixture
    ) -> None:
        """Test that missing teacher keys in the state dict do not raise an error."""

        # Setup constants
        batch_size = 2
        student_embed_dim = 32
        teacher_embed_dim = 48

        # Dummy student model with real params.
        student_model = EmbeddingModel(
            wrapped_model=DummyCustomModel(student_embed_dim)
        )

        # Dummy teacher model with real params.
        teacher_model = EmbeddingModel(
            wrapped_model=DummyCustomModel(teacher_embed_dim)
        )

        # Patch get_teacher.
        mock_get_teacher = mocker.patch(
            "lightly_train._methods.distillationv2.distillationv2.get_teacher"
        )
        mock_get_teacher.return_value = teacher_model

        # Instantiate the distillation method.
        distill = DistillationV2(
            method_args=DistillationV2Args(),
            optimizer_args=DistillationV2LARSArgs(),
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

        # Dummy student model with real params.
        student_model = EmbeddingModel(
            wrapped_model=DummyCustomModel(student_embed_dim)
        )

        # Dummy teacher model with real params.
        teacher_model = EmbeddingModel(
            wrapped_model=DummyCustomModel(teacher_embed_dim)
        )

        # Patch get_teacher.
        mock_get_teacher = mocker.patch(
            "lightly_train._methods.distillationv2.distillationv2.get_teacher"
        )
        mock_get_teacher.return_value = teacher_model

        # Create distillation instance
        distill = DistillationV2(
            method_args=DistillationV2Args(),
            optimizer_args=DistillationV2LARSArgs(),
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
