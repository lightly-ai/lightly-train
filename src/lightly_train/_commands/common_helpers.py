#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import warnings
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Generator, Iterable, Literal

import torch
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator
from pytorch_lightning.plugins.environments import (  # type: ignore[attr-defined]
    SLURMEnvironment,
)
from pytorch_lightning.strategies.strategy import Strategy
from torch.nn import Module
from torch.utils.data import Dataset

from lightly_train._commands import common_helpers
from lightly_train._data import image_dataset
from lightly_train._data._serialize import memory_mapped_sequence
from lightly_train._data._serialize.memory_mapped_sequence import MemoryMappedSequence
from lightly_train._data.image_dataset import ImageDataset
from lightly_train._embedding.embedding_format import EmbeddingFormat
from lightly_train._models import package_helpers
from lightly_train.types import DatasetItem, PathLike, Transform

logger = logging.getLogger(__name__)


LIGHTLY_TRAIN_MASK_DIR = os.environ.get("LIGHTLY_TRAIN_MASK_DIR", None)


def get_checkpoint_path(checkpoint: PathLike) -> Path:
    checkpoint_path = Path(checkpoint).resolve()
    logger.debug(f"Making sure checkpoint '{checkpoint_path}' exists.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist!")
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint '{checkpoint_path}' is not a file!")
    return checkpoint_path


def get_out_path(out: PathLike, overwrite: bool) -> Path:
    out_path = Path(out).resolve()
    logger.debug(f"Checking if output path '{out_path}' exists.")
    if out_path.exists():
        if not overwrite:
            raise ValueError(
                f"Output '{out_path}' already exists! Set overwrite=True to overwrite "
                "the file."
            )
        if not out_path.is_file():
            raise ValueError(f"Output '{out_path}' is not a file!")
    return out_path


def get_accelerator(
    accelerator: str | Accelerator,
) -> str | Accelerator:
    logger.debug(f"Getting accelerator for '{accelerator}'.")
    if accelerator != "auto":
        # User specified an accelerator, return it.
        return accelerator

    # Default to CUDA if available.
    if CUDAAccelerator.is_available():
        logger.debug("CUDA is available, defaulting to CUDA.")
        return CUDAAccelerator()
    elif MPSAccelerator.is_available():
        logger.debug("MPS is available, defaulting to MPS.")
        return MPSAccelerator()
    else:
        logger.debug("CUDA and MPS are not available, defaulting to CPU.")
        return CPUAccelerator()


def _get_rank() -> int | None:
    """Get the rank of the current process.

    Copied from https://github.com/Lightning-AI/pytorch-lightning/blob/06a8d5bf33faf0a4f9a24207ae77b439354350af/src/lightning/fabric/utilities/rank_zero.py#L39-L49
    """
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    # None to differentiate whether an environment variable was set at all
    return None


def is_rank_zero() -> bool:
    """Check if the current process is running on the first device."""
    local_rank = _get_rank()
    return local_rank == 0 or local_rank is None


def get_out_dir(out: PathLike, resume: bool, overwrite: bool) -> Path:
    out_dir = Path(out).resolve()
    logger.debug(f"Checking if output directory '{out_dir}' exists.")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"Output '{out_dir}' is not a directory!")

        dir_not_empty = any(out_dir.iterdir())

        if dir_not_empty and (not (resume or overwrite)) and is_rank_zero():
            raise ValueError(
                f"Output '{out_dir}' is not empty! Set overwrite=True to overwrite the "
                "directory or resume=True to resume training."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def pretty_format_args(args: dict[str, Any], indent: int = 4) -> str:
    args = sanitize_config_dict(args)

    return json.dumps(args, indent=indent, sort_keys=True)


def sanitize_config_dict(args: dict[str, Any]) -> dict[str, Any]:
    """Replace classes with their names in the train config dictionary."""
    if isinstance(args.get("model"), Module):
        args["model"] = args["model"].__class__.__name__
    if isinstance(args.get("accelerator"), Accelerator):
        args["accelerator"] = args["accelerator"].__class__.__name__
    if isinstance(args.get("strategy"), Strategy):
        args["strategy"] = args["strategy"].__class__.__name__
    if isinstance(args.get("format"), EmbeddingFormat):
        args["format"] = args["format"].value
    for key, value in args.items():
        if isinstance(value, Path):
            args[key] = str(value)
    return args


def get_num_workers(
    num_workers: int | Literal["auto"], num_devices_per_node: int
) -> int:
    """Returns the number of workers for the dataloader.

    The number of workers are per dataloader. Every device has its own dataloader.
    """
    if num_workers == "auto":
        num_cpus_per_device = _get_num_cpus_per_device(
            num_devices_per_node=num_devices_per_node
        )
        if num_cpus_per_device is None:
            num_workers_auto = 8
        else:
            # Leave 1 CPU for the main process on every device
            num_workers_auto = max(num_cpus_per_device - 1, 0)

        return num_workers_auto
    else:
        return num_workers


def _get_num_cpus_per_device(num_devices_per_node: int) -> int | None:
    """Returns the number of available CPUs per device."""
    if _is_slurm():
        cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK")
        logger.debug(f"SLURM_CPUS_PER_TASK: {cpus_per_task}")
        if cpus_per_task and isinstance(cpus_per_task, str):
            cpu_count = int(cpus_per_task)
        else:
            cpu_count = None
    else:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            cpu_count = cpu_count // num_devices_per_node
    return cpu_count


def _is_slurm() -> bool:
    return "SLURM_JOB_ID" in os.environ


class ModelPart(Enum):
    MODEL = "model"
    EMBEDDING_MODEL = "embedding_model"


class ModelFormat(Enum):
    PACKAGE_DEFAULT = "package_default"
    TORCH_MODEL = "torch_model"
    TORCH_STATE_DICT = "torch_state_dict"

    @classmethod
    def _missing_(cls, value: object) -> None | ModelFormat:
        if str(value) == "ultralytics":
            warnings.warn(
                "The 'ultralytics' format is deprecated and will be removed in version "
                "0.5.0., instead the format can be omitted since it is mapped to the "
                "default format.",
                FutureWarning,
            )
            return cls.PACKAGE_DEFAULT
        raise ValueError(f"{value} is not a valid {cls.__name__}")


def export_model(model: Module, format: ModelFormat, out: Path) -> None:
    if not is_rank_zero():
        return
    logger.debug(f"Exporting model to '{out}' in format '{format}'.")
    out.parent.mkdir(parents=True, exist_ok=True)
    if format == ModelFormat.TORCH_MODEL:
        torch.save(model, out)
    elif format == ModelFormat.TORCH_STATE_DICT:
        torch.save(model.state_dict(), out)
    elif format == ModelFormat.PACKAGE_DEFAULT:
        package = package_helpers.get_package_from_model(model=model)
        package.export_model(model=model, out=out)
    else:
        raise ValueError(f"Invalid format: '{format.value}' is not supported ")


@contextlib.contextmanager
def get_dataset_temp_mmap_path() -> Generator[Path, Any, Any]:
    """Generate file in temporary directory to be used for memory-mapping the dataset.

    In case the Lightning Trainer is used to spawn the processes, rank 0 will run first
    and create the memory-mapped file and share this via the env variable
    LIGHTLY_TRAIN_DATASET_MMAP_PATH. This allows all processes to use the same file.

    In case training runs on SLURM with the `srun` command, the control over spawning
    the processes is given to SLURM and `srun` immediately runs the script `--ntasks-per-node`
    many times in parallel. Therefore a common location of the temporary file can't be
    communicated between the processes and every process has to create its own memory-mapped
    file.
    On Windows, each process creates its own temporary file due to file handling restrictions.
    """
    if platform.system() == "Windows" or SLURMEnvironment.detect():
        # On Windows or SLURM, every rank creates its own temporary file
        with NamedTemporaryFile(delete=True) as mmap_file:
            mmap_filepath = Path(mmap_file.name)
            # Close the file here to prevent failures on Windows, as it will be accessed
            # by another process.
            mmap_file.close()
            yield mmap_filepath
    else:
        # Unix-like systems can share the same file between processes
        LIGHTLY_TRAIN_DATASET_MMAP_PATH = "LIGHTLY_TRAIN_DATASET_MMAP_PATH"
        if common_helpers.is_rank_zero():
            with NamedTemporaryFile(delete=True) as mmap_file:
                mmap_filepath = Path(mmap_file.name)
                os.environ[LIGHTLY_TRAIN_DATASET_MMAP_PATH] = str(mmap_filepath)
                yield mmap_filepath
        else:
            # Make sure that all ranks use the same memory-mapped file.
            mmap_filepath = Path(os.environ[LIGHTLY_TRAIN_DATASET_MMAP_PATH])
            yield mmap_filepath


def get_dataset_mmap_filenames(
    filenames: Iterable[str],
    mmap_filepath: Path,
) -> MemoryMappedSequence[str]:
    """Returns memory-mapped filenames shared across all ranks.

    In case training runs on SLURM with the `srun` command, the control over spawning
    the processes is given to SLURM and `srun` immediately runs the script `--ntasks-per-node`
    many times in parallel. Therefore a common location of the temporary file can't be
    communicated between the processes and every process has to create its own memory-
    mapped file.
    """
    if not SLURMEnvironment.detect():
        if common_helpers.is_rank_zero():
            # Save filenames to memory mapped file and return them.
            return memory_mapped_sequence.memory_mapped_sequence_from_filenames(
                filenames=filenames,
                mmap_filepath=mmap_filepath,
            )
        else:
            # Return memory-mapped filenames from file.
            return memory_mapped_sequence.memory_mapped_sequence_from_file(
                mmap_filepath=mmap_filepath
            )
    else:
        # every rank creates its own memory-mapped sequence
        return memory_mapped_sequence.memory_mapped_sequence_from_filenames(
            filenames=filenames,
            mmap_filepath=mmap_filepath,
        )


def get_dataset(
    data: PathLike | Dataset[DatasetItem],
    transform: Transform,
    mmap_filepath: Path | None,
) -> Dataset[DatasetItem]:
    if isinstance(data, Dataset):
        logger.debug("Using provided dataset.")
        return data

    data = Path(data).resolve()
    logger.debug(f"Making sure data directory '{data}' exists and is not empty.")
    if not data.exists():
        raise ValueError(f"Data directory '{data}' does not exist!")
    elif not data.is_dir():
        raise ValueError(f"Data path '{data}' is not a directory!")
    elif data.is_dir() and not any(data.iterdir()):
        raise ValueError(f"Data directory '{data}' is empty!")
    if mmap_filepath is None:
        raise ValueError("Memory-mapped file path must be provided.")

    logger.info(f"Initializing dataset from '{data}'.")
    # NOTE(Guarin, 01/25): The bottleneck for dataset initialization is filename
    # listing and not the memory mapping. Listing the train set from ImageNet takes
    # about 30 seconds. This is mostly because os.walk is not parallelized.
    filenames = image_dataset.list_image_filenames(image_dir=data)
    return ImageDataset(
        image_dir=data,
        image_filenames=get_dataset_mmap_filenames(
            filenames=filenames,
            mmap_filepath=mmap_filepath,
        ),
        transform=transform,
        mask_dir=Path(LIGHTLY_TRAIN_MASK_DIR) if LIGHTLY_TRAIN_MASK_DIR else None,
    )
