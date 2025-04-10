#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
)
from torch import Tensor
from torch.nn import Module

from lightly_train._callbacks import callback_helpers
from lightly_train._callbacks.callback_args import (
    CallbackArgs,
)
from lightly_train._callbacks.checkpoint import ModelCheckpoint, ModelCheckpointArgs
from lightly_train._transforms.transform import NormalizeArgs
from lightly_train.errors import ConfigValidationError

from .. import helpers
from ..helpers import DummyCustomModel


@pytest.mark.parametrize(
    "callback_args, expected_result",
    [
        # Test case for default empty dictionary
        ({}, CallbackArgs()),
        # Test case for None input
        (None, CallbackArgs()),
        # Test case for user config
        (
            {"model_checkpoint": {"every_n_epochs": 5}},
            CallbackArgs(model_checkpoint=ModelCheckpointArgs(every_n_epochs=5)),
        ),
        # Test case for passing CallbackArgs object
        (
            CallbackArgs(model_checkpoint=ModelCheckpointArgs(every_n_epochs=42)),
            CallbackArgs(model_checkpoint=ModelCheckpointArgs(every_n_epochs=42)),
        ),
    ],
)
def test_get_callback_args__success(
    callback_args: CallbackArgs | dict[str, Any] | None, expected_result: CallbackArgs
) -> None:
    callback_args = callback_helpers.get_callback_args(callback_args)
    assert callback_args == expected_result


def test_get_callback_args__failure() -> None:
    with pytest.raises(ConfigValidationError):
        callback_helpers.get_callback_args({"nonexisting_arg": 1})


def test_get_callbacks__default(tmp_path: Path) -> None:
    model = helpers.get_model()
    embedding_model = helpers.get_embedding_model(model=model)
    callback_args = CallbackArgs()
    callbacks = callback_helpers.get_callbacks(
        callback_args=callback_args,
        out=tmp_path,
        model=model,
        embedding_model=embedding_model,
        normalize_args=NormalizeArgs(),
    )
    assert len(callbacks) == 5
    early_stopping = next(c for c in callbacks if isinstance(c, EarlyStopping))
    model_checkpoint = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
    assert early_stopping.monitor == "train_loss"
    assert early_stopping.patience == int(1e12)
    assert model_checkpoint.save_last
    assert str(model_checkpoint.dirpath) == str(tmp_path / "checkpoints")


def test_get_callbacks__disable(tmp_path: Path) -> None:
    model = helpers.get_model()
    embedding_model = helpers.get_embedding_model(model=model)
    callback_args = CallbackArgs(
        learning_rate_monitor=None,
        early_stopping=None,
    )
    callbacks = callback_helpers.get_callbacks(
        callback_args=callback_args,
        out=tmp_path,
        model=model,
        embedding_model=embedding_model,
        normalize_args=NormalizeArgs(),
    )
    assert len(callbacks) == 3
    assert any(isinstance(c, DeviceStatsMonitor) for c in callbacks)
    assert any(isinstance(c, ModelCheckpoint) for c in callbacks)


def test_get_callbacks__user_config(tmp_path: Path) -> None:
    model = helpers.get_model()
    embedding_model = helpers.get_embedding_model(model=model)
    callback_args = CallbackArgs(
        model_checkpoint=ModelCheckpointArgs(every_n_epochs=5),
    )
    callbacks = callback_helpers.get_callbacks(
        callback_args=callback_args,
        out=tmp_path,
        model=model,
        embedding_model=embedding_model,
        normalize_args=NormalizeArgs(),
    )
    model_checkpoint = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
    assert str(model_checkpoint.dirpath) == str(tmp_path / "checkpoints")
    assert model_checkpoint.every_n_epochs == 5


def test_get_checkpoint_model() -> None:
    model = helpers.get_model()
    checkpoint_model = callback_helpers.get_checkpoint_model(model=model)
    assert not isinstance(checkpoint_model, DummyCustomModel)


def test_get_checkpoint_model__no_model_getter() -> None:
    class DummyCustomModel(Module):
        def feature_dim(self) -> int:
            return 1

        def forward_features(self, x: Tensor) -> Tensor:
            return torch.zeros(1)

        def forward_pool(self, x: Tensor) -> Tensor:
            return torch.zeros(1)

    model = DummyCustomModel()
    checkpoint_model = callback_helpers.get_checkpoint_model(model=model)
    assert model == checkpoint_model
    assert isinstance(checkpoint_model, DummyCustomModel)
