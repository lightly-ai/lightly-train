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

from lightly_train._methods.distillationv2.distillationv2 import (
    DistillationV2,
    DistillationV2LARSArgs,
)
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType


class TestDistillation:
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
