#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from albumentations.pytorch.transforms import ToTensorV2
from pytest_mock import MockerFixture
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.nn import Module
from torchvision import models

from lightly_train import _distributed
from lightly_train._commands import common_helpers
from lightly_train._data import cache
from tests._commands.test_train_helpers import MockDataset


@pytest.mark.parametrize(
    "resume_interrupted, resume, expected",
    [
        (True, None, True),
        (False, None, False),
        (True, True, True),
        (False, False, False),
    ],
)
def test_get_resume_interrupted(
    resume_interrupted: bool, resume: bool | None, expected: bool
) -> None:
    assert (
        common_helpers.get_resume_interrupted(
            resume_interrupted=resume_interrupted,
            resume=resume,
        )
        == expected
    )


@pytest.mark.parametrize(
    "resume_interrupted, resume",
    [
        (True, False),
        (False, True),
    ],
)
def test_get_resume_interrupted__error(resume_interrupted: bool, resume: bool) -> None:
    with pytest.raises(
        ValueError,
        match="resume_interrupted=.* and resume=.* cannot be set at the same time!",
    ):
        common_helpers.get_resume_interrupted(
            resume_interrupted=resume_interrupted,
            resume=resume,
        )


def test_get_checkpoint_path(tmp_path: Path) -> None:
    out_file = tmp_path / "file.ckpt"
    out_file.touch()
    assert common_helpers.get_checkpoint_path(checkpoint=out_file) == out_file


def test_get_checkpoint_path__non_existing(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        common_helpers.get_checkpoint_path(checkpoint=out_dir)


def test_get_checkpoint_path__non_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ValueError):
        common_helpers.get_checkpoint_path(checkpoint=out_dir)


def test_get_out_path__nonexisting(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    assert common_helpers.get_out_path(out=out_dir, overwrite=False) == out_dir


def test_get_out_path__existing__no_overwrite(tmp_path: Path) -> None:
    out_file = tmp_path / "file.txt"
    out_file.touch()
    with pytest.raises(ValueError):
        common_helpers.get_out_path(out=out_file, overwrite=False)


def test_get_out_path__existing_file__overwrite(tmp_path: Path) -> None:
    out_file = tmp_path / "file.txt"
    out_file.touch()
    assert common_helpers.get_out_path(out=out_file, overwrite=True) == out_file


def test_get_out_path__existing_dir__overwrite(tmp_path: Path) -> None:
    out_dir = tmp_path / "dir"
    out_dir.mkdir()
    with pytest.raises(ValueError):
        common_helpers.get_out_path(out=out_dir, overwrite=True)


def test_get_accelerator__set() -> None:
    """Test that same accelerator is returned if it is set."""
    assert common_helpers.get_accelerator(accelerator="cpu") == "cpu"
    accelerator = CPUAccelerator()
    assert common_helpers.get_accelerator(accelerator=accelerator) == accelerator


def test_get_out_dir(tmp_path: Path) -> None:
    assert (
        common_helpers.get_out_dir(
            out=tmp_path, resume_interrupted=False, overwrite=False
        )
        == tmp_path
    )


def test_get_out_dir_nonexisting(tmp_path: Path) -> None:
    out_dir = tmp_path / "nonexisting"
    assert (
        common_helpers.get_out_dir(
            out=out_dir, resume_interrupted=False, overwrite=False
        )
        == out_dir
    )


def test_get_out_dir__nondir(tmp_path: Path) -> None:
    out_dir = tmp_path / "file.txt"
    out_dir.touch()
    with pytest.raises(ValueError):
        common_helpers.get_out_dir(
            out=out_dir, resume_interrupted=False, overwrite=False
        )


@pytest.mark.parametrize("resume_interrupted", [True, False])
@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize("rank_zero", [True, False])
def test_get_out_dir__nonempty(
    mocker: MockerFixture,
    tmp_path: Path,
    resume_interrupted: bool,
    overwrite: bool,
    rank_zero: bool,
) -> None:
    (tmp_path / "some_file.txt").touch()
    mocker.patch.object(_distributed, "is_global_rank_zero", return_value=rank_zero)
    if resume_interrupted or overwrite or (not rank_zero):
        assert (
            common_helpers.get_out_dir(
                out=tmp_path, resume_interrupted=resume_interrupted, overwrite=overwrite
            )
            == tmp_path
        )
    else:
        with pytest.raises(ValueError):
            common_helpers.get_out_dir(
                out=tmp_path, resume_interrupted=resume_interrupted, overwrite=overwrite
            )


def test_get_tmp_dir__default() -> None:
    # This is by default the same as the data cache directory.
    assert common_helpers.get_tmp_dir() == cache.get_data_cache_dir()


def test_get_tmp_dir__custom(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_TMP_DIR": str(tmp_path)})
    assert common_helpers.get_tmp_dir() == tmp_path


def test_verify_out_dir_equal_on_all_local_ranks(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    # Use clean temporary directory for the test.
    tmp_dir = tmp_path / "tmp"
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_TMP_DIR": str(tmp_dir)})

    out_dir = tmp_path / "out"
    # Simulate calling the function from rank 0
    mocker.patch.dict(os.environ, {"LOCAL_RANK": "0"})
    with common_helpers.verify_out_dir_equal_on_all_local_ranks(out_dir):
        # Simulate calling the function from rank 1
        mocker.patch.dict(os.environ, {"LOCAL_RANK": "1"})
        with common_helpers.verify_out_dir_equal_on_all_local_ranks(out_dir):
            pass

    # Make sure that no files are left in the temporary directory.
    assert not tmp_dir.exists() or not any(f for f in tmp_dir.iterdir() if f.is_file())


def test_verify_out_dir_equal_on_all_local_ranks__different(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    # Use clean temporary directory for the test.
    tmp_dir = tmp_path / "tmp"
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_TMP_DIR": str(tmp_dir)})

    out_dir_rank0 = tmp_path / "rank0"
    out_dir_rank1 = tmp_path / "rank1"

    # Simulate calling the function from rank 0
    mocker.patch.dict(os.environ, {"LOCAL_RANK": "0"})
    with common_helpers.verify_out_dir_equal_on_all_local_ranks(out_dir_rank0):
        # Simulate calling the function from rank 1
        mocker.patch.dict(
            os.environ,
            {"LOCAL_RANK": "1", "LIGHTLY_TRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC": "0.1"},
        )
        with pytest.raises(RuntimeError, match="Rank 1: Timeout after 0.1 seconds"):
            with common_helpers.verify_out_dir_equal_on_all_local_ranks(out_dir_rank1):
                pass

    # Make sure that no files are left in the temporary directory.
    assert not tmp_dir.exists() or not any(f for f in tmp_dir.iterdir() if f.is_file())


def test_verify_out_dir_equal_on_all_local_ranks__no_rank0(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    # Use clean temporary directory for the test.
    tmp_dir = tmp_path / "tmp"
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_TMP_DIR": str(tmp_dir)})

    out_dir = tmp_path / "rank1"

    mocker.patch.dict(
        os.environ,
        {"LOCAL_RANK": "1", "LIGHTLY_TRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC": "0.1"},
    )
    with pytest.raises(RuntimeError, match="Rank 1: Timeout after 0.1 seconds"):
        with common_helpers.verify_out_dir_equal_on_all_local_ranks(out_dir):
            pass

    # Make sure that no files are left in the temporary directory.
    assert not tmp_dir.exists() or not any(f for f in tmp_dir.iterdir() if f.is_file())


@pytest.mark.parametrize(
    "input_args, expected_output",
    [
        (
            {
                "model": Module(),
                "accelerator": CPUAccelerator(),
                "strategy": DDPStrategy(),
            },
            {
                "model": "Module",
                "accelerator": "CPUAccelerator",
                "strategy": "DDPStrategy",
            },
        ),
        (
            {"model": None, "accelerator": None, "strategy": None},
            {"model": None, "accelerator": None, "strategy": None},
        ),
        (
            {"model": Module(), "accelerator": None, "strategy": DDPStrategy()},
            {"model": "Module", "accelerator": None, "strategy": "DDPStrategy"},
        ),
    ],
)
def test_sanitize_config_dict(
    input_args: dict[str, Any], expected_output: dict[str, Any]
) -> None:
    assert common_helpers.sanitize_config_dict(input_args) == expected_output


def test_pretty_format_args() -> None:
    args = {
        "model_args": None,
        "num_nodes": 1,
        "num_workers": 8,
        "optim_args": {"lr": 0.0001},
        "out": "my_output_dir",
        "overwrite": False,
        "precision": "16-mixed",
        "resume_interrupted": False,
        "seed": 0,
        "strategy": "auto",
        "trainer_args": None,
        "callbacks": None,
        "transform_args": None,
        "accelerator": "auto",
        "batch_size": 128,
        "data": "my_data_dir",
        "devices": "auto",
        "embed_dim": None,
        "epochs": 100,
        "loader_args": None,
        "method": "simclr",
        "method_args": {"temperature": 0.1},
        "model": "torchvision/resnet18",
        "resume": None,
    }
    # Assert that the args are ordered alphabetically.
    expected_str = (
        "{\n"
        '    "accelerator": "auto",\n'
        '    "batch_size": 128,\n'
        '    "callbacks": null,\n'
        '    "data": "my_data_dir",\n'
        '    "devices": "auto",\n'
        '    "embed_dim": null,\n'
        '    "epochs": 100,\n'
        '    "loader_args": null,\n'
        '    "method": "simclr",\n'
        '    "method_args": {\n'
        '        "temperature": 0.1\n'
        "    },\n"
        '    "model": "torchvision/resnet18",\n'
        '    "model_args": null,\n'
        '    "num_nodes": 1,\n'
        '    "num_workers": 8,\n'
        '    "optim_args": {\n'
        '        "lr": 0.0001\n'
        "    },\n"
        '    "out": "my_output_dir",\n'
        '    "overwrite": false,\n'
        '    "precision": "16-mixed",\n'
        '    "resume": null,\n'
        '    "resume_interrupted": false,\n'
        '    "seed": 0,\n'
        '    "strategy": "auto",\n'
        '    "trainer_args": null,\n'
        '    "transform_args": null\n'
        "}"
    )
    assert common_helpers.pretty_format_args(args=args) == expected_str


def test_pretty_format_args__custom_model() -> None:
    assert common_helpers.pretty_format_args(
        args={
            "model": models.resnet18(),
            "batch_size": 128,
            "epochs": 100,
        }
    ) == ('{\n    "batch_size": 128,\n    "epochs": 100,\n    "model": "ResNet"\n}')

    class MyModel(Module):
        pass

    assert common_helpers.pretty_format_args(
        args={
            "model": MyModel(),
            "batch_size": 128,
            "epochs": 100,
        }
    ) == ('{\n    "batch_size": 128,\n    "epochs": 100,\n    "model": "MyModel"\n}')


@pytest.mark.parametrize(
    "args, expected",
    [
        (
            {
                "model": "torchvision/resnet18",
                "batch_size": 128,
                "data": ["my_data_dir", "my_data_dir_2"],
                "devices": [0, 1],
            },
            {
                "model": "torchvision/resnet18",
                "batch_size": 128,
                "data": ["my_data_dir", "my_data_dir_2"],
                "devices": [0, 1],
            },
        ),
        (
            {
                "model": "torchvision/resnet18",
                "batch_size": 128,
                "data": [
                    "my_data_dir",
                    "my_data_dir_2",
                    "my_data_dir_3",
                    "my_data_dir_4",
                    "my_data_dir_5",
                    "my_data_dir_6",
                ],
                "devices": [0, 1],
            },
            {
                "model": "torchvision/resnet18",
                "batch_size": 128,
                "data": [
                    "my_data_dir",
                    "my_data_dir_2",
                    "my_data_dir_3",
                    "...",
                    "my_data_dir_6",
                ],
                "devices": [0, 1],
            },
        ),
        (
            {
                "model": "torchvision/resnet18",
                "batch_size": 128,
                "data": [
                    "my_data_dir",
                    "my_data_dir_2",
                    "my_data_dir_3",
                    "my_data_dir_4",
                    "my_data_dir_5",
                    "my_data_dir_6",
                ],
                "devices": [0, 1, 2, 3, 4, 5, 6, 7],
            },
            {
                "model": "torchvision/resnet18",
                "batch_size": 128,
                "data": [
                    "my_data_dir",
                    "my_data_dir_2",
                    "my_data_dir_3",
                    "...",
                    "my_data_dir_6",
                ],
                "devices": [0, 1, 2, "...", 7],
            },
        ),
    ],
)
def test_remove_excessive_args__all_keys(
    args: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert common_helpers.remove_excessive_args(args=args) == expected


def test_remove_excessive_args__specific_key() -> None:
    args = {
        "model": "torchvision/resnet18",
        "batch_size": 128,
        "data": [
            "my_data_dir",
            "my_data_dir_2",
            "my_data_dir_3",
            "my_data_dir_4",
            "my_data_dir_5",
            "my_data_dir_6",
        ],
        "devices": [0, 1, 2, 3, 4, 5, 6, 7],
    }
    expected = {
        "model": "torchvision/resnet18",
        "batch_size": 128,
        "data": [
            "my_data_dir",
            "my_data_dir_2",
            "my_data_dir_3",
            "...",
            "my_data_dir_6",
        ],
        "devices": [0, 1, 2, 3, 4, 5, 6, 7],
    }
    assert (
        common_helpers.remove_excessive_args(args=args, limit_keys={"data"}) == expected
    )


@pytest.mark.parametrize(
    "num_workers,os_cpu_count,num_devices_per_node,expected_result",
    [
        (0, None, 1, 0),
        (8, None, 1, 8),
        (8, None, 3, 8),
        (64, None, 1, 64),
        (8, 64, 1, 8),
        ("auto", None, 1, 8),
        ("auto", 4, 1, 3),
        ("auto", 4, 2, 1),
        ("auto", 4, 3, 0),
        ("auto", 4, 4, 0),
        ("auto", 4, 8, 0),
        ("auto", 8, 1, 7),
        ("auto", 8, 3, 1),
        ("auto", 16, 1, 8),  # Capped by LIGHTLY_TRAIN_MAX_NUM_WORKERS_AUTO
        ("auto", 64, 7, 8),
    ],
)
def test_get_num_workers(
    mocker: MockerFixture,
    num_workers: int | Literal["auto"],
    os_cpu_count: int | None,
    num_devices_per_node: int,
    expected_result: int,
) -> None:
    mocker.patch.object(common_helpers.os, "cpu_count", return_value=os_cpu_count)  # type: ignore[attr-defined]
    assert (
        common_helpers.get_num_workers(
            num_workers=num_workers, num_devices_per_node=num_devices_per_node
        )
        == expected_result
    )


@pytest.mark.parametrize(
    "num_workers,num_devices_per_node,slurm_cpus_per_task,expected_result",
    [
        (0, 1, "8", 0),
        (1, 1, "8", 1),
        ("auto", 1, "8", 7),
        ("auto", 2, "8", 7),  # num_devices_per_node is ignored
        ("auto", 1, "", 8),  # fallback to default value of 8 workers
        # SLURM_CPUS_PER_TASK overrules LIGHTLY_TRAIN_MAX_NUM_WORKERS_AUTO
        ("auto", 1, "16", 15),
    ],
)
def test_get_num_workers__slurm(
    num_workers: int | Literal["auto"],
    num_devices_per_node: int,
    slurm_cpus_per_task: str,
    expected_result: int,
    mocker: MockerFixture,
) -> None:
    mocker.patch.dict(
        os.environ, {"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": slurm_cpus_per_task}
    )
    assert (
        common_helpers.get_num_workers(
            num_workers=num_workers, num_devices_per_node=num_devices_per_node
        )
        == expected_result
    )


def test_get_dataset_temp_mmap_path(tmp_path: Path) -> None:
    with common_helpers.get_dataset_temp_mmap_path(data=tmp_path) as mmap_path:
        mmap_path.touch()
    # Make sure file is deleted after exiting the context manager.
    assert not mmap_path.exists()


def test_get_dataset_temp_mmap_path__rank(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    # Simulate calling the function from rank 0
    mocker.patch.dict(os.environ, {"LOCAL_RANK": "0"})
    with common_helpers.get_dataset_temp_mmap_path(data=tmp_path) as mmap_path_rank0:
        pass

    # Simulate calling the function from rank 1
    mocker.patch.dict(os.environ, {"LOCAL_RANK": "1"})
    with common_helpers.get_dataset_temp_mmap_path(data=tmp_path) as mmap_path_rank1:
        pass

    assert mmap_path_rank0 == mmap_path_rank1


def test_get_dataset_mmap_filenames__rank0(tmp_path: Path) -> None:
    filenames = ["file1.jpg", "file2.jpg", "file3.jpg"]
    mmap_filepath = tmp_path / "test.mmap"
    mmap_filenames = common_helpers.get_dataset_mmap_filenames(
        filenames=filenames,
        mmap_filepath=mmap_filepath,
    )
    assert list(mmap_filenames) == filenames


def test_get_dataset_mmap_filenames__rank(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    filenames = ["file1.jpg", "file2.jpg", "file3.jpg"]
    mmap_filepath = tmp_path / "test.mmap"

    # Simulate calling the function from rank 0
    mocker.patch.dict(os.environ, {"LOCAL_RANK": "0"})
    mmap_filenames_rank0 = common_helpers.get_dataset_mmap_filenames(
        filenames=filenames,
        mmap_filepath=mmap_filepath,
    )

    # Simulate calling the function from rank 1
    mocker.patch.dict(os.environ, {"LOCAL_RANK": "1"})
    mmap_filenames_rank1 = common_helpers.get_dataset_mmap_filenames(
        filenames=filenames,
        mmap_filepath=mmap_filepath,
    )
    assert list(mmap_filenames_rank0) == filenames
    assert list(mmap_filenames_rank1) == filenames


def test_get_dataset_mmap_filenames__rank_error(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    # Test that the function raises an error if it is called with different paths
    # from different ranks.
    filenames = ["file1.jpg", "file2.jpg", "file3.jpg"]
    mmap_filepath_rank0 = tmp_path / "rank0.mmap"
    mmap_filepath_rank1 = tmp_path / "rank1.mmap"

    # Simulate calling the function from rank 0.
    mocker.patch.dict(os.environ, {"LOCAL_RANK": "0"})
    common_helpers.get_dataset_mmap_filenames(
        filenames=filenames,
        mmap_filepath=mmap_filepath_rank0,
    )

    # Simulate calling the function from rank 1.
    mocker.patch.dict(
        os.environ, {"LOCAL_RANK": "1", "LIGHTLY_TRAIN_MMAP_TIMEOUT_SEC": "0.1"}
    )
    with pytest.raises(RuntimeError, match="Rank 1: Timeout after 0.1 seconds"):
        common_helpers.get_dataset_mmap_filenames(
            filenames=filenames,
            mmap_filepath=mmap_filepath_rank1,
        )


def test_get_dataset_mmap_filenames__reuse(
    tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    filenames = ["file1.jpg", "file2.jpg", "file3.jpg"]
    mmap_filepath = tmp_path / "test.mmap"

    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_MMAP_REUSE_FILE": "1"})
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_TMP_DIR": str(tmp_path)})
    mmap_filenames_first = common_helpers.get_dataset_mmap_filenames(
        filenames=filenames,
        mmap_filepath=mmap_filepath,
    )

    # make sure warning is raised if the file already exists
    mocker.patch.dict(os.environ, {"LOCAL_RANK": "1"})
    with caplog.at_level(logging.WARNING):
        mmap_filenames_reused = common_helpers.get_dataset_mmap_filenames(
            filenames=filenames,
            mmap_filepath=mmap_filepath,
        )
    assert "Reusing existing memory-mapped file " in caplog.text
    assert list(mmap_filenames_first) == filenames
    assert list(mmap_filenames_reused) == filenames


def test_get_dataset__path(tmp_path: Path) -> None:
    (tmp_path / "img.jpg").touch()
    mmap_filepath = tmp_path / "test.pyarrow"
    _ = common_helpers.get_dataset(
        data=tmp_path,
        transform=ToTensorV2(),
        mmap_filepath=mmap_filepath,
    )


def test_get_dataset__path__nonexisting(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        common_helpers.get_dataset(
            data=tmp_path / "nonexisting",
            transform=ToTensorV2(),
            mmap_filepath=None,
        )


def test_get_dataset__path__nondir(tmp_path: Path) -> None:
    file = tmp_path / "img.jpg"
    file.touch()
    with pytest.raises(ValueError):
        common_helpers.get_dataset(
            data=file,
            transform=ToTensorV2(),
            mmap_filepath=None,
        )


def test_get_dataset__path__empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        common_helpers.get_dataset(
            data=tmp_path,
            transform=ToTensorV2(),
            mmap_filepath=None,
        )


def test_get_dataset__dirs_and_files(tmp_path: Path) -> None:
    single_img1 = tmp_path / "img.jpg"
    single_img2 = tmp_path / "img_2.jpg"
    single_img1.touch()
    single_img2.touch()
    img_dir = tmp_path / "dir"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "img_1.jpg").touch()
    (img_dir / "img_3.jpg").touch()
    assert os.path.isdir(str(img_dir))
    mmap_filepath = Path(tmp_path / "test.pyarrow")
    _ = common_helpers.get_dataset(
        data=[
            single_img1,
            single_img2,
            img_dir,
        ],
        transform=ToTensorV2(),
        mmap_filepath=mmap_filepath,
    )


def test_get_dataset__dataset() -> None:
    dataset = MockDataset(torch.rand(10, 3, 224, 224))
    dataset_1 = common_helpers.get_dataset(
        data=dataset,
        transform=ToTensorV2(),
        mmap_filepath=None,
    )
    assert dataset == dataset_1


@pytest.mark.parametrize(
    "total_num_devices, global_batch_size, expected_batch_size",
    [
        (1, 8, 8),
        (2, 8, 8),
        (4, 8, 8),
    ],
)
def test_get_global_batch_size(
    total_num_devices: int, global_batch_size: int, expected_batch_size: int
) -> None:
    dataset = MockDataset(torch.rand(16, 3, 32, 32))
    result_global_batch_size = common_helpers.get_global_batch_size(
        global_batch_size=global_batch_size,
        dataset=dataset,
        total_num_devices=total_num_devices,
        loader_args=None,
    )
    assert result_global_batch_size == expected_batch_size


def test_get_global_batch_size__dataset() -> None:
    dataset = MockDataset(torch.rand(2, 2, 32, 32))
    global_batch_size = common_helpers.get_global_batch_size(
        global_batch_size=8,
        dataset=dataset,
        total_num_devices=1,
        loader_args=None,
    )
    assert global_batch_size == 2  # Size of the dataset


@pytest.mark.parametrize(
    "total_num_devices, expected_batch_size",
    [
        (1, 4),
        (2, 8),
    ],
)
def test_get_global_batch_size__loader_args(
    total_num_devices: int, expected_batch_size: int
) -> None:
    dataset = MockDataset(torch.rand(16, 3, 32, 32))
    global_batch_size = common_helpers.get_global_batch_size(
        global_batch_size=8,
        dataset=dataset,
        total_num_devices=total_num_devices,
        loader_args={"batch_size": 4},
    )
    # Expect: total_num_devices * loader_args["batch_size"]
    assert global_batch_size == expected_batch_size


def test_get_global_batch_size__error() -> None:
    dataset = MockDataset(torch.rand(16, 3, 32, 32))
    with pytest.raises(
        ValueError,
        match=re.escape("Batch size 8 must be divisible by (num_nodes * devices) = 3."),
    ):
        common_helpers.get_global_batch_size(
            global_batch_size=8,
            dataset=dataset,
            total_num_devices=3,
            loader_args=None,
        )
