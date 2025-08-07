#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from pytest_mock import MockerFixture
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch import Tensor
from torch.testing import assert_close
from torch.utils.data import Dataset
from torchvision.datasets import FakeData

from lightly_train._commands import train_helpers
from lightly_train._loggers.jsonl import JSONLLogger
from lightly_train._methods.simclr.simclr import (
    SimCLR,
    SimCLRArgs,
    SimCLRSGDArgs,
)
from lightly_train._methods.simclr.simclr_transform import (
    SimCLRColorJitterArgs,
    SimCLRTransform,
    SimCLRTransformArgs,
)
from lightly_train._models import package_helpers
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._scaling import IMAGENET_SIZE, ScalingInfo
from lightly_train._transforms.transform import (
    MethodTransformArgs,
    NormalizeArgs,
    RandomRotationArgs,
)
from lightly_train.errors import ConfigValidationError
from lightly_train.types import DatasetItem

from .. import helpers
from ..helpers import DummyCustomModel

REPRESENTATIVE_MODEL_NAMES = [
    "timm/vit_tiny_patch16_224",
    "timm/resnet18",
    "torchvision/convnext_tiny",
    "torchvision/resnet18",
]


class MockDataset(Dataset[DatasetItem]):
    """Typed implementation of torch's TensorDataset."""

    def __init__(self, *tensors: Tensor) -> None:
        self.tensors = tensors

    def __getitem__(self, index: int) -> DatasetItem:
        item: DatasetItem = {
            "filename": "dummy_filename.jpg",
            "views": [tensor[index] for tensor in self.tensors],
        }
        return item

    def __len__(self) -> int:
        return self.tensors[0].size(0)


def test_get_transform__method() -> None:
    assert isinstance(
        train_helpers.get_transform(
            method="simclr", transform_args_resolved=SimCLRTransformArgs()
        ),
        SimCLRTransform,
    )


def test_get_transform__method_and_transform_dict() -> None:
    transform = train_helpers.get_transform(
        method="simclr",
        transform_args_resolved=SimCLRTransformArgs(random_gray_scale=0.42),
    )
    assert isinstance(transform, SimCLRTransform)
    assert transform.transform_args.random_gray_scale == 0.42


@pytest.mark.parametrize(
    "num_nodes, num_devices, expected_total_num_devices",
    [
        (1, 1, 1),
        (1, 2, 2),
        (2, 1, 2),
        (2, 2, 4),
    ],
)
def test_get_total_num_devices(
    num_nodes: int, num_devices: int, expected_total_num_devices: int
) -> None:
    total_num_devices = train_helpers.get_total_num_devices(
        num_nodes=num_nodes,
        num_devices=num_devices,
    )
    assert total_num_devices == expected_total_num_devices


@pytest.mark.parametrize(
    "batch_size, loader_args, expected_batch_size",
    [
        (2, None, 2),
        (2, {"batch_size": 4}, 2),  # loader_args["batch_size"] is ignored
    ],
)
def test_get_dataloader(
    batch_size: int,
    loader_args: dict[str, Any] | None,
    expected_batch_size: int,
) -> None:
    dataset = MockDataset(torch.rand(16, 3, 32, 32))
    dataloader = train_helpers.get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        loader_args=loader_args,
    )
    assert len(dataloader) == 16 // expected_batch_size
    batches = list(dataloader)
    assert len(batches) == 16 // expected_batch_size
    assert all(
        view.shape == (expected_batch_size, 3, 32, 32) for view in batches[0]["views"]
    )


@pytest.mark.parametrize("model_name", REPRESENTATIVE_MODEL_NAMES)
@pytest.mark.parametrize("embed_dim", [None, 64])
@pytest.mark.parametrize("model_args", [None, {"num_classes": 42}])
def test_get_embedding_model(
    model_name: str, embed_dim: int | None, model_args: dict[str, Any] | None
) -> None:
    if model_name.startswith("timm/"):
        pytest.importorskip("timm")
    x = torch.rand(1, 3, 224, 224)
    model = package_helpers.get_wrapped_model(model_name, model_args=model_args)
    embedding_model = train_helpers.get_embedding_model(model, embed_dim=embed_dim)
    embedding = embedding_model.forward(x)
    assert embedding.shape == (1, embedding_model.embed_dim, 1, 1)
    if (
        model_args is not None
        and "num_classes" in model_args
        and not model_name.startswith("torchvision/")
    ):
        assert model.get_model().num_classes == model_args["num_classes"]


@pytest.mark.parametrize("embed_dim", [None, 64])
def test_get_embedding_model__custom(embed_dim: int | None) -> None:
    model = DummyCustomModel()
    x = torch.rand(1, 3, 224, 224)
    embedding_model = train_helpers.get_embedding_model(model, embed_dim=embed_dim)
    assert isinstance(embedding_model.wrapped_model, ModelWrapper)
    embedding = embedding_model.forward(x)
    assert embedding.shape == (1, embedding_model.embed_dim, 1, 1)


def test_get_trainer(tmp_path: Path) -> None:
    trainer = train_helpers.get_trainer(
        out=tmp_path,
        epochs=1,
        accelerator="cpu",
        strategy="auto",
        devices="auto",
        num_nodes=1,
        precision=32,
        log_every_n_steps=1,
        loggers=[JSONLLogger(save_dir="logs")],
        callbacks=[],
        trainer_args=None,
    )
    assert len(trainer.loggers) == 1
    assert trainer.loggers[0].__class__.__name__ == "JSONLLogger"
    assert trainer.max_epochs == 1


def test_get_lightning_logging_interval() -> None:
    assert (
        train_helpers.get_lightning_logging_interval(dataset_size=10, batch_size=10)
        == 1
    )
    assert (
        train_helpers.get_lightning_logging_interval(dataset_size=10, batch_size=2) == 5
    )
    assert (
        train_helpers.get_lightning_logging_interval(dataset_size=1000, batch_size=10)
        == 50
    )
    assert (
        train_helpers.get_lightning_logging_interval(dataset_size=1, batch_size=10) == 1
    )


def test_get_lightning_logging_interval__fails_for_nonpositive_inputs() -> None:
    with pytest.raises(
        ValueError,
        match=r"Dataset size \(0\) and batch size \(10\) must be positive integers.",
    ):
        train_helpers.get_lightning_logging_interval(dataset_size=0, batch_size=10)
    with pytest.raises(
        ValueError,
        match=r"Dataset size \(10\) and batch size \(0\) must be positive integers.",
    ):
        train_helpers.get_lightning_logging_interval(dataset_size=10, batch_size=0)
    with pytest.raises(
        ValueError,
        match=r"Dataset size \(-10\) and batch size \(10\) must be positive integers.",
    ):
        train_helpers.get_lightning_logging_interval(dataset_size=-10, batch_size=10)
    with pytest.raises(
        ValueError,
        match=r"Dataset size \(10\) and batch size \(-10\) must be positive integers.",
    ):
        train_helpers.get_lightning_logging_interval(dataset_size=10, batch_size=-10)


_DDP_STRATEGY = DDPStrategy()


@pytest.mark.parametrize(
    "strategy, accelerator, devices, expected",
    [
        ("ddp", "auto", "auto", "ddp"),
        (_DDP_STRATEGY, "auto", "auto", _DDP_STRATEGY),
        ("auto", "cpu", "auto", "auto"),  # CPU should not use DDP by default
        ("auto", "cpu", 1, "auto"),
        ("auto", "cpu", 2, "ddp_find_unused_parameters_true"),
        ("auto", "gpu", 1, "auto"),
        ("auto", "gpu", 2, "ddp_find_unused_parameters_true"),
    ],
)
def test_get_strategy(
    strategy: str | DDPStrategy,
    accelerator: str,
    devices: str | int,
    expected: str | DDPStrategy,
) -> None:
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            pytest.skip("No GPU available.")
        assert isinstance(devices, int)
        if devices > torch.cuda.device_count():
            pytest.skip("Not enough GPUs available.")

    assert (
        train_helpers.get_strategy(
            strategy=strategy, accelerator=accelerator, devices=devices
        )
        == expected
    )


@pytest.mark.parametrize(
    "optim_type, expected",
    [
        ("auto", "auto"),
        ("adamw", OptimizerType.ADAMW),
        (OptimizerType.ADAMW, OptimizerType.ADAMW),
        ("sgd", OptimizerType.SGD),
    ],
)
def test_get_optimizer_type(
    optim_type: str | OptimizerType, expected: OptimizerType | Literal["auto"]
) -> None:
    assert train_helpers.get_optimizer_type(optim_type=optim_type) == expected


@pytest.mark.parametrize(
    "optim_type, optim_args, expected",
    [
        (OptimizerType.ADAMW, None, AdamWArgs()),
        (OptimizerType.ADAMW, {}, AdamWArgs()),
        (
            OptimizerType.ADAMW,
            {"lr": 0.1, "betas": [0.2, 0.3]},
            AdamWArgs(lr=0.1, betas=(0.2, 0.3)),
        ),
        (OptimizerType.ADAMW, AdamWArgs(), AdamWArgs()),
        (OptimizerType.SGD, None, SimCLRSGDArgs()),
    ],
)
def test_get_optimizer_args(
    optim_type: OptimizerType, optim_args: dict[str, Any] | None, expected: AdamWArgs
) -> None:
    assert (
        train_helpers.get_optimizer_args(
            optim_type=optim_type, optim_args=optim_args, method_cls=SimCLR
        )
        == expected
    )


@pytest.mark.parametrize(
    "dataset, expected",
    [
        (FakeData(size=2), 2),
        (iter(FakeData(size=2)), IMAGENET_SIZE),
    ],
)
def test_dataset_size(dataset: Dataset[DatasetItem], expected: int) -> None:
    assert train_helpers.get_dataset_size(dataset=dataset) == expected


def test_get_scaling_info() -> None:
    expected = ScalingInfo(dataset_size=100, epochs=10)
    assert train_helpers.get_scaling_info(dataset_size=100, epochs=10) == expected


@pytest.mark.parametrize(
    "args,expected",
    [
        (None, SimCLRArgs()),
        ({}, SimCLRArgs()),
        (
            {
                "hidden_dim": 42,
            },
            SimCLRArgs(hidden_dim=42),
        ),
        (SimCLRArgs(hidden_dim=42), SimCLRArgs(hidden_dim=42)),
    ],
)
def test_get_method_args(
    args: dict[str, Any] | SimCLRArgs | None, expected: SimCLRArgs
) -> None:
    scaling_info = ScalingInfo(dataset_size=2, epochs=100)
    resolved_args = train_helpers.get_method_args(
        method_cls=SimCLR,
        method_args=args,
        scaling_info=scaling_info,
        optimizer_args=AdamWArgs(),
        wrapped_model=DummyCustomModel(),
    )
    assert resolved_args == expected


def test_get_method() -> None:
    embedding_model = EmbeddingModel(wrapped_model=DummyCustomModel())
    method = train_helpers.get_method(
        method_cls=SimCLR,
        method_args=SimCLRArgs(temperature=0.2),
        optimizer_args=AdamWArgs(),
        embedding_model=embedding_model,
        global_batch_size=1,
    )
    assert isinstance(method, SimCLR)
    assert method.method_args.temperature == 0.2
    assert method.global_batch_size == 1
    assert method.embedding_model == embedding_model
    assert method.optimizer_args == AdamWArgs()


# Test for SimCLR and DINOv2 methods.
@pytest.mark.parametrize(
    "method,passed_epochs,dataset_size,passed_batch_size,expected_epochs",
    [
        ("simclr", 10, 32, 100_000, 10),
        ("simclr", "auto", 256, 100_000, 100),
        (
            "dinov2",
            "auto",
            1_000_000,
            32,
            4,  # 125_000 * 32 / 1_000_000 = 4
        ),
        (
            "dinov2",
            "auto",
            10_000,
            1024,
            12800,  # 125_000 * 1024 / 10_000 = 12800
        ),
        ("dinov2", 10, 1, 100_000, 10),
    ],
)
def test_get_epochs(
    method: str,
    passed_epochs: int | Literal["auto"],
    dataset_size: int,
    passed_batch_size: int,
    expected_epochs: int,
) -> None:
    epochs = train_helpers.get_epochs(
        method=method,
        epochs=passed_epochs,
        batch_size=passed_batch_size,
        dataset_size=dataset_size,
    )
    assert epochs == expected_epochs


@pytest.mark.parametrize(
    "transform_dict, expected_result",
    [
        # Test case for default empty dictionary
        ({}, SimCLRTransformArgs()),
        # Test case for None input
        (None, SimCLRTransformArgs()),
        # Test case for user config
        (
            {
                "normalize": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
                "random_rotation": {"prob": 0.5, "degrees": 30},
                "color_jitter": {"brightness": 0.1},
            },
            SimCLRTransformArgs(
                normalize=NormalizeArgs(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                random_rotation=RandomRotationArgs(prob=0.5, degrees=30),
                color_jitter=SimCLRColorJitterArgs(brightness=0.1),
            ),
        ),
        # Test case of SimCLRTransformArgs input
        (
            SimCLRTransformArgs(
                normalize=NormalizeArgs(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ),
            SimCLRTransformArgs(
                normalize=NormalizeArgs(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ),
        ),
    ],
)
def test_get_transform_args__success(
    transform_dict: MethodTransformArgs | dict[str, Any] | None,
    expected_result: MethodTransformArgs,
) -> None:
    transform_args = train_helpers.get_transform_args(
        method="simclr", transform_args=transform_dict
    )
    assert transform_args == expected_result


def test_get_transform_args__failure() -> None:
    with pytest.raises(ConfigValidationError):
        train_helpers.get_transform_args(
            method="simclr",
            transform_args={"nonexisting_arg": 1},
        )


def test_load_checkpoint(tmp_path: Path, mocker: MockerFixture) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint = helpers.get_checkpoint()
    checkpoint.save(checkpoint_path)
    wrapped_model = DummyCustomModel()
    embedding_model = EmbeddingModel(wrapped_model=wrapped_model)
    method = helpers.get_method(wrapped_model=wrapped_model)
    spy_load_state_dict = mocker.spy(train_helpers, "load_state_dict")

    train_helpers.load_checkpoint(
        checkpoint=checkpoint_path,
        resume_interrupted=False,
        wrapped_model=wrapped_model,
        embedding_model=embedding_model,
        method=method,
    )

    spy_load_state_dict.assert_called_once_with(
        wrapped_model=wrapped_model,
        embedding_model=embedding_model,
        method=method,
        checkpoint=checkpoint_path,
    )


def test_load_checkpoint__no_checkpoint(mocker: MockerFixture) -> None:
    spy_load_state_dict = mocker.spy(train_helpers, "load_state_dict")
    train_helpers.load_checkpoint(
        checkpoint=None,
        resume_interrupted=False,
        wrapped_model=mocker.MagicMock(),
        embedding_model=mocker.MagicMock(),
        method=mocker.MagicMock(),
    )
    spy_load_state_dict.assert_not_called()


def test_load_checkpoint__checkpoint_and_resume(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    with pytest.raises(
        ValueError,
        match=r"resume_interrupted=True and checkpoint='.*' cannot be set at the same time!",
    ):
        train_helpers.load_checkpoint(
            checkpoint=tmp_path / "checkpoint.pth",
            resume_interrupted=True,
            wrapped_model=mocker.MagicMock(),
            embedding_model=mocker.MagicMock(),
            method=mocker.MagicMock(),
        )


def test_load_state_dict(tmp_path: Path) -> None:
    # Generate a checkpoint and save it on disk.
    # This checkpoint contains a model, a wrapped mode, an embedding model,
    # and a method.
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint = helpers.get_checkpoint()
    checkpoint.save(checkpoint_path)

    # Generate a new model and make it different from the model in the checkpoint.
    # Use .get_model() to get weights that are certainly in all 3 different models.
    model_2 = DummyCustomModel()
    next(model_2.get_model().parameters()).data += 1
    embedding_model_2 = EmbeddingModel(wrapped_model=model_2)
    method_2 = helpers.get_method(wrapped_model=model_2)

    train_helpers.load_state_dict(
        wrapped_model=model_2,
        embedding_model=embedding_model_2,
        method=method_2,
        checkpoint=checkpoint_path,
    )

    # Assert that the model, wrapped model, embedding model, and method have updated
    # the weights.
    assert_close(
        list(model_2.get_model().parameters()),
        list(checkpoint.lightly_train.models.model.parameters()),
    )
    assert_close(
        list(model_2.parameters()),
        list(checkpoint.lightly_train.models.wrapped_model.parameters()),
    )
    assert torch.allclose(
        next(embedding_model_2.parameters()),
        next(checkpoint.lightly_train.models.embedding_model.parameters()),
    )
    assert torch.allclose(
        next(method_2.parameters()),
        next(checkpoint.lightly_train.models.wrapped_model.get_model().parameters()),
    )

    # Assert that the model, embedding model, and method share the same parameter objects.
    assert next(model_2.get_model().parameters()) is next(
        embedding_model_2.parameters()
    )
    assert next(model_2.get_model().parameters()) is next(method_2.parameters())
