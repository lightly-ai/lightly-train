#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from omegaconf import OmegaConf
from pytest import LogCaptureFixture
from pytest_mock import MockerFixture
from pytorch_lightning.accelerators.cpu import CPUAccelerator

from lightly_train._checkpoint import Checkpoint
from lightly_train._commands import train
from lightly_train._commands.train import (
    CLITrainConfig,
    FunctionTrainConfig,
    TrainConfig,
)
from lightly_train._loggers.jsonl import JSONLLogger
from lightly_train._methods import method_helpers
from lightly_train._methods.dino.dino import DINOAdamWArgs, DINOArgs
from lightly_train._scaling import ScalingInfo

from .. import helpers
from ..helpers import DummyCustomModel


def test_train__cpu(tmp_path: Path) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    helpers.create_images(image_dir=data, files=10)

    train.train(
        out=out,
        data=data,
        model="torchvision/resnet18",
        method="simclr",
        batch_size=4,
        num_workers=2,
        epochs=1,
        accelerator="cpu",
    )

    # Check that the correct files were created.
    filepaths = {fp.relative_to(out) for fp in out.rglob("*")}
    expected_filepaths = {
        Path("checkpoints"),
        Path("checkpoints") / "epoch=0-step=2.ckpt",
        Path("checkpoints") / "last.ckpt",
        Path("exported_models"),
        Path("exported_models") / "exported_last.pt",
        Path("metrics.jsonl"),
        Path("train.log"),
        # Tensorboard filename is not deterministic, so we need to find it.
        next(fp for fp in filepaths if fp.name.startswith("events.out.tfevents")),
    }
    assert filepaths == expected_filepaths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
@pytest.mark.parametrize("num_workers", [0, 2, "auto"])
def test_train(
    tmp_path: Path, caplog: LogCaptureFixture, num_workers: int | Literal["auto"]
) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    helpers.create_images(image_dir=data, files=10)

    train.train(
        out=out,
        data=data,
        model="torchvision/resnet18",
        method="simclr",
        batch_size=4,
        num_workers=num_workers,
        epochs=1,
        devices=1,
    )

    # Check that we can resume training
    last_ckpt_path = out / "checkpoints" / "last.ckpt"
    first_ckpt = Checkpoint.from_path(checkpoint=last_ckpt_path)

    with caplog.at_level(logging.INFO):
        train.train(
            out=out,
            data=data,
            model="torchvision/resnet18",
            method="simclr",
            batch_size=4,
            num_workers=2,
            epochs=2,
            devices=1,
            resume_interrupted=True,
        )
    assert (
        f"Restoring states from the checkpoint path at {last_ckpt_path}" in caplog.text
    )
    # Epochs in checkpoint are 0-indexed. Epoch 1 is therefore the second epoch.
    # weights_only=True does not work here.
    assert torch.load(last_ckpt_path, weights_only=False)["epoch"] == 1

    # Check that exported checkpoint weights changed between first and second run.
    second_ckpt = Checkpoint.from_path(checkpoint=last_ckpt_path)
    first_state_dict = first_ckpt.lightly_train.models.model.state_dict()
    second_state_dict = second_ckpt.lightly_train.models.model.state_dict()
    assert first_state_dict.keys() == second_state_dict.keys()
    for key in first_state_dict.keys():
        if key.startswith("fc."):
            # Skip the last layer as it is not pretrained.
            continue
        assert not torch.equal(first_state_dict[key], second_state_dict[key])

    # Check that last.ckpt and exported_model.pt contain same information. If this fails
    # it means that checkpoint loading is not working correctly.
    exported_state_dict = torch.load(
        out / "exported_models" / "exported_last.pt", weights_only=True
    )
    assert second_state_dict.keys() == exported_state_dict.keys()
    for key in second_state_dict.keys():
        if key.startswith("fc."):
            # Skip the last layer as it is not pretrained.
            continue
        assert torch.equal(second_state_dict[key], exported_state_dict[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_train__overwrite_true(tmp_path: Path) -> None:
    """Test that overwrite=True allows training with an existing output directory that
    contains files."""
    out = tmp_path / "out"
    data = tmp_path / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "file.txt").touch()
    helpers.create_images(image_dir=data, files=10)

    train.train(
        out=out,
        data=data,
        model="torchvision/resnet18",
        method="simclr",
        batch_size=4,
        num_workers=0,
        epochs=1,
        devices=1,
        overwrite=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_train__overwrite_false(tmp_path: Path) -> None:
    (tmp_path / "file.txt").touch()

    with pytest.raises(ValueError):
        train.train(
            out=tmp_path,
            data=tmp_path,
            model="torchvision/resnet18",
            method="simclr",
            batch_size=4,
            num_workers=0,
            epochs=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_train__embed_dim(tmp_path: Path) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    helpers.create_images(image_dir=data, files=10)

    train.train(
        out=out,
        data=data,
        model="torchvision/resnet18",
        method="simclr",
        batch_size=4,
        num_workers=0,
        epochs=1,
        devices=1,
        embed_dim=64,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_train__custom_model(tmp_path: Path) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    helpers.create_images(image_dir=data, files=10)

    train.train(
        out=out,
        data=data,
        model=helpers.DummyCustomModel(),
        method="simclr",
        batch_size=4,
        num_workers=0,
        devices=1,
        epochs=1,
    )


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Requires Python 3.10 or higher for typing."
)
def test_train__parameters() -> None:
    """Tests that train function and TrainConfig have the same parameters and default
    values.

    This test is here to make sure we don't forget to update train/TrainConfig when
    we change parameters in one of the two.
    """
    helpers.assert_same_params(a=FunctionTrainConfig, b=train.train)
    helpers.assert_same_params(a=TrainConfig, b=FunctionTrainConfig, assert_type=False)
    helpers.assert_same_params(a=TrainConfig, b=CLITrainConfig, assert_type=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_train__zero_epochs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    helpers.create_images(image_dir=data, files=10)
    train.train(
        out=out,
        data=data,
        model="torchvision/resnet18",
        method="simclr",
        batch_size=4,
        num_workers=0,
        devices=1,
        epochs=0,
    )
    assert (out / "checkpoints" / "last.ckpt").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_train_from_dictconfig(tmp_path: Path) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    helpers.create_images(image_dir=data, files=10)
    config = OmegaConf.create(
        dict(
            out=str(out),
            data=str(data),
            model="torchvision/resnet18",
            method="simclr",
            batch_size=4,
            num_workers=0,
            epochs=1,
            devices=1,
            optim_args={"lr": 0.1},
            loader_args={"shuffle": True},
            trainer_args={"min_epochs": 1},
            model_args={"num_classes": 42},
            callbacks={"model_checkpoint": {"every_n_epochs": 5}},
            loggers={"jsonl": {"flush_logs_every_n_steps": 5}},
        )
    )
    train.train_from_dictconfig(config=config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
@pytest.mark.parametrize("method", method_helpers._list_methods())
@pytest.mark.parametrize(
    "devices", [1]
)  # TODO(Philipp, 09/24): Add test with 2 devices back.
def test_train__method(tmp_path: Path, method: str, devices: int) -> None:
    if torch.cuda.device_count() < devices:
        pytest.skip("Test requires more GPUs than available.")

    out = tmp_path / "out"
    data = tmp_path / "data"
    helpers.create_images(image_dir=data, files=10)

    # DINOv2 needs special model
    model = {
        "dinov2": "dinov2/_vittest14",
    }.get(method, "torchvision/resnet18")

    # Use smaller teacher for unit tests.
    method_args = {
        "distillation": {"teacher": "dinov2/_vittest14"},
        "distillationv1": {"teacher": "dinov2/_vittest14"},
        "distillationv2": {"teacher": "dinov2/_vittest14"},
    }.get(method, {})

    train.train(
        out=out,
        data=data,
        model=model,
        devices=devices,
        method=method,
        method_args=method_args,
        batch_size=4,
        num_workers=0,
        epochs=1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_train__checkpoint_gradients(tmp_path: Path) -> None:
    """Test that checkpoints saved during training do not have disabled gradients.

    This is especially a problem for methods with momentum encoders (e.g. DINO) where
    the momentum encoder does not receive gradients during training. As the momentum
    encoder is used for finetuning, we want to make sure that it doesn't have gradients
    disabled in the checkpoint as this can result in subtle bugs where users don't
    realize that the model is frozen while finetuning.
    """
    out = tmp_path / "out"
    data = tmp_path / "data"
    helpers.create_images(image_dir=data, files=10)

    train.train(
        out=out,
        data=data,
        model="torchvision/resnet18",
        method="dino",
        batch_size=4,
        num_workers=0,
        epochs=1,
        devices=1,
    )
    ckpt_path = out / "checkpoints" / "last.ckpt"
    ckpt = Checkpoint.from_path(checkpoint=ckpt_path)
    for param in ckpt.lightly_train.models.wrapped_model.get_model().parameters():
        assert param.requires_grad


def test_train__TrainConfig__model_dump(tmp_path: Path) -> None:
    """
    Test that TrainConfig is dumped correctly even if some of its attributes are
    subclasses of the types specified in the TrainConfig class.
    """
    out = tmp_path / "out"
    data = tmp_path / "data"
    method_args = DINOArgs()
    optim_args = DINOAdamWArgs()
    method_args.resolve_auto(
        scaling_info=ScalingInfo(dataset_size=20_000, epochs=100),
        optimizer_args=optim_args,
        wrapped_model=DummyCustomModel(),
    )
    config = TrainConfig(
        out=out,
        data=data,
        model="torchvision/resnet18",
        method="simclr",
        optim_args=optim_args,
        method_args=method_args,
    )
    dumped_config_direct = config.model_dump()

    # Assert that the indirect dump is the same as the direct dump.
    dumped_cofig_indirect = {
        key: value.model_dump() if hasattr(value, "model_dump") else value
        for key, value in config.__dict__.items()
    }
    assert dumped_config_direct == dumped_cofig_indirect

    # Check for some specific attributes.
    assert dumped_config_direct["optim_args"]["betas"] == (0.9, 0.999)
    assert dumped_config_direct["method_args"]["warmup_teacher_temp_epochs"] == 30


def test_train__log_resolved_config(caplog: LogCaptureFixture, tmp_path: Path) -> None:
    out = tmp_path / "out"
    data = tmp_path / "data"
    config = TrainConfig(
        out=out,
        data=data,
        accelerator=CPUAccelerator(),
        batch_size=4,
        model="torchvision/resnet18",
    )

    class MemoryLogger(JSONLLogger):
        def __init__(self) -> None:
            self.logs: list[dict[str, Any]] = []

        # Type ignore because JSONLLogger.log_hyperparams has a more complicated
        # signature but we only require part of it for the thest.
        def log_hyperparams(self, params: dict[str, Any]) -> None:  # type: ignore[override]
            self.logs.append(params)

    logger = MemoryLogger()

    assert len(logger.logs) == 0
    with caplog.at_level(logging.INFO):
        train.log_resolved_config(config=config, loggers=[logger])
        expected = (
            "Resolved configuration:\n"
            "{\n"
            '    "accelerator": "CPUAccelerator",\n'
            '    "batch_size": 4,\n'
        )
        assert expected in caplog.text

    assert len(logger.logs) == 1
    assert logger.logs[0]["accelerator"] == "CPUAccelerator"
    assert logger.logs[0]["batch_size"] == 4


def test_train__checkpoint(mocker: MockerFixture, tmp_path: Path) -> None:
    """
    Assert that train_helpers.load_state_dict is called when a checkpoint is provided.
    """
    out = tmp_path / "out"
    data = tmp_path / "data"
    # Use 12 images to make sure that we have at least 3 batches. We need 3 batches for
    # DINO to show updates in the model due to the teacher/student setup and momentum
    # updates. The following happens:
    # After step 1: Student batch norm has not yet changed.
    # After step 2: Student batch norm has changed, but teacher is still the same.
    # After step 3: Teacher gets EMA update from student.
    helpers.create_images(image_dir=data, files=12)

    # Part 1: Generate a checkpoint.
    train.train(
        out=out,
        data=data,
        model="torchvision/resnet18",
        method="dino",
        batch_size=4,
        num_workers=0,
        epochs=0,
        accelerator="cpu",
        devices=1,
    )
    last_ckpt_path = out / "checkpoints" / "last.ckpt"
    first_ckpt = Checkpoint.from_path(checkpoint=last_ckpt_path)

    # Part 2: Load the checkpoint
    spy_load_state_dict = mocker.spy(train.train_helpers, "load_state_dict")  # type: ignore[attr-defined]
    train.train(
        out=out,
        data=data,
        model="torchvision/resnet18",
        method="dino",
        batch_size=4,
        num_workers=0,
        epochs=1,
        overwrite=True,
        checkpoint=last_ckpt_path,
        accelerator="cpu",
        devices=1,
        optim_args={"lr": 10},  # Make sure that parameters change meaningfully.
    )
    spy_load_state_dict.assert_called_once()
    call_args = spy_load_state_dict.call_args_list[0]
    args, kwargs = call_args
    assert kwargs["checkpoint"] == last_ckpt_path

    # Check that exported checkpoint weights changed between first and second run.
    second_ckpt = Checkpoint.from_path(checkpoint=last_ckpt_path)
    first_state_dict = first_ckpt.lightly_train.models.model.state_dict()
    second_state_dict = second_ckpt.lightly_train.models.model.state_dict()
    assert first_state_dict.keys() == second_state_dict.keys()
    for key in first_state_dict.keys():
        if key.startswith("fc."):
            # Skip the last layer as it is not pretrained.
            continue
        assert not torch.equal(first_state_dict[key], second_state_dict[key])

    # Check that last.ckpt and exported_model.pt contain same information. If this fails
    # it means that checkpoint loading is not working correctly.
    exported_state_dict = torch.load(
        out / "exported_models" / "exported_last.pt", weights_only=True
    )
    assert second_state_dict.keys() == exported_state_dict.keys()
    for key in second_state_dict.keys():
        if key.startswith("fc."):
            # Skip the last layer as it is not pretrained.
            continue
        assert torch.equal(second_state_dict[key], exported_state_dict[key])
