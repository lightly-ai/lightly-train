#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from typing import Literal

import pytest
from pytest_mock import MockerFixture

from lightly_train._methods.method_args import MethodArgs
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.sgd_args import SGDArgs

from ..helpers import DummyCustomModel, DummyMethod


class TestMethod:
    @pytest.mark.parametrize("lr_scale_method", ["linear", "sqrt"])
    @pytest.mark.parametrize("global_batch_size", [256, 512, 1024])
    @pytest.mark.parametrize("reference_batch_size", [256, 1024, 1536])
    def test_method_lr_scaling_generic(
        self,
        mocker: MockerFixture,
        lr_scale_method: Literal["linear", "sqrt"],
        global_batch_size: int,
        reference_batch_size: int,
    ) -> None:
        """Generic test that Method.configure_optimizers applies correct learning rate scaling."""
        # Set constants.
        embed_dim = 32

        # Set the method arguments.
        method_args = MethodArgs(
            reference_batch_size=reference_batch_size,
            lr_scale_method=lr_scale_method,
        )

        # Set the optimizer arguments.
        optimizer_args = SGDArgs()

        # Set the embedding model.
        embedding_model = EmbeddingModel(wrapped_model=DummyCustomModel(embed_dim))

        # Compute expected learning rate.
        base_lr = optimizer_args.lr
        expected_scale = global_batch_size / reference_batch_size
        if lr_scale_method == "sqrt":
            expected_scale = math.sqrt(expected_scale)
        expected_lr = base_lr * expected_scale

        # Instantiate dummy method.
        method = DummyMethod(
            method_args=method_args,
            optimizer_args=optimizer_args,
            embedding_model=embedding_model,
            global_batch_size=global_batch_size,
        )

        # Mock trainer attributes needed by configure_optimizers.
        mock_trainer = mocker.Mock()
        mock_trainer.max_epochs = 100
        mock_trainer.estimated_stepping_batches = 1000
        method.trainer = mock_trainer

        # Call configure_optimizers.
        optimizers_schedulers = method.configure_optimizers()

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
