#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import tempfile
import time
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Iterable, Literal

import torch
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator
from pytorch_lightning.strategies.strategy import Strategy
from torch.nn import Module
from torch.utils.data import Dataset

from lightly_train._data import image_dataset
from lightly_train._data._serialize import memory_mapped_sequence
from lightly_train._data._serialize.memory_mapped_sequence import MemoryMappedSequence
from lightly_train._data.image_dataset import ImageDataset
from lightly_train._embedding.embedding_format import EmbeddingFormat
from lightly_train._env import Env
from lightly_train._models import package_helpers
from lightly_train.types import DatasetItem, PathLike, Transform

logger = logging.getLogger(__name__)


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


def get_global_rank() -> int | None:
    """Get the global rank of the current process.

    Copied from https://github.com/Lightning-AI/pytorch-lightning/blob/06a8d5bf33faf0a4f9a24207ae77b439354350af/src/lightning/fabric/utilities/rank_zero.py#L39-L49
    """
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return None


def get_local_rank() -> int | None:
    """Get the local rank of the current process."""
    rank_keys = ("LOCAL_RANK", "SLURM_LOCALID", "JSM_NAMESPACE_LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return None


def get_node_rank() -> int | None:
    """Get the node rank of the current process."""
    rank_keys = ("NODE_RANK", "GROUP_RANK", "SLURM_NODEID")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return None


def is_global_rank_zero() -> bool:
    """Check if the current process is running on the global rank zero."""
    global_rank = get_global_rank()
    # Check node rank because process might be assigned to a node but not yet
    # a global rank.
    return global_rank == 0 or (global_rank is None and is_node_rank_zero())


def is_local_rank_zero() -> bool:
    """Check if the current process is running on the local rank zero."""
    local_rank = get_local_rank()
    return local_rank == 0 or local_rank is None


def is_node_rank_zero() -> bool:
    """Check if the current process is running on the node rank zero."""
    node_rank = get_node_rank()
    return node_rank == 0 or node_rank is None


def get_out_dir(out: PathLike, resume: bool, overwrite: bool) -> Path:
    out_dir = Path(out).resolve()
    logger.debug(f"Checking if output directory '{out_dir}' exists.")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"Output '{out_dir}' is not a directory!")

        dir_not_empty = any(out_dir.iterdir())

        if dir_not_empty and (not (resume or overwrite)) and is_global_rank_zero():
            raise ValueError(
                f"Output '{out_dir}' is not empty! Set overwrite=True to overwrite the "
                "directory or resume=True to resume training."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_tmp_dir() -> Path:
    """Get the temporary directory for Lightly Train."""
    return Path(tempfile.gettempdir()) / "lightly-train"


def get_data_tmp_dir() -> Path:
    """Get the temporary directory for Lightly Train data."""
    return get_tmp_dir() / "data"


def get_verify_out_tmp_dir() -> Path:
    """Get the temporary directory for Lightly Train verify out."""
    return get_tmp_dir() / "verify-out"


def get_sha256(value: Any) -> str:
    """Get the SHA256 hash of a value."""
    return hashlib.sha256(str(value).encode()).hexdigest()


@contextlib.contextmanager
def verify_out_dir_equal_on_all_local_ranks(out: Path) -> Generator[None, None, None]:
    """Verify that the out path is the same on all local ranks.

    This is important for distributed training, as the out path is used as
    a deterministic value that must be consistent across all local ranks.

    A common case where this can fail is when the out path contains a timestamp
    in the path that is generated inside the training script. For example with:
    >>> out = f"out/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    This will result in different paths for each rank as the timestamp is generated
    on each rank separately.
    """
    out_dir = Path(out).resolve()
    # Add the node rank to the filename. This makes sure that each node verifies
    # its out directory separately, even if the nodes are using a shared filesystem.
    out_tmp = get_verify_out_tmp_dir() / get_sha256(f"{out_dir}-{get_node_rank() or 0}")
    logger.debug(f"Creating temporary file '{out_tmp}' to verify out path.")

    try:
        if is_local_rank_zero():
            _unlink_and_ignore(out_tmp)
            out_tmp.parent.mkdir(parents=True, exist_ok=True)
            out_tmp.touch()
            # Write the out directory to the temporary file for debugging.
            out_tmp.write_text(str(out_dir))
            yield
        else:
            # Wait for rank zero to create the temporary file.
            timeout_sec = Env.LIGHTLY_TRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC.value
            start_time_sec = time.time()
            while not out_tmp.exists():
                if timeout_sec >= 0 and time.time() - start_time_sec > timeout_sec:
                    raise RuntimeError(
                        f"Rank {get_global_rank()}: Timeout after {timeout_sec} seconds "
                        "while verifying that all ranks (processes) have the same 'out' path. "
                        "This means that the 'out' path is not set to the same path on all ranks. "
                        "If the path to your 'out' path contains any timestamps make sure that "
                        "they are provided from OUTSIDE of the training script. Either via the "
                        "command line or an environment variable. Timestamps created inside a Python "
                        "script, for example with 'datetime.now()' or 'time.time()', will result "
                        "in different values for each rank and must not be used. "
                        f"The timeout can be configured with the {Env.LIGHTLY_TRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC.name} "
                        "environment variable. Setting it to -1 disables the timeout. "
                    )
                time.sleep(0.1)
            yield
    finally:
        _unlink_and_ignore(out_tmp)


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

        # Limit the number of automatically created workers in case
        # the system has a lot of CPUs.
        num_workers_auto = min(
            num_workers_auto, Env.LIGHTLY_TRAIN_MAX_NUM_WORKERS_AUTO.value
        )
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
    if not is_global_rank_zero():
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
def get_dataset_temp_mmap_path(out: Path) -> Generator[Path, Any, Any]:
    """Generate file in temporary directory to be used for memory-mapping the dataset.

    Creates a unique filename for the memory-mapped file based on the out path.
    We use the out path as a deterministic value that is consistent across all ranks
    on the same node.

    We need a determinstic value from "outside" at this point in the code as the
    code might already be running on multiple processes depending on how it was
    started. We cannot create a new filename based on a random value as this would
    create a different filename for each process. Creating the filename on global
    rank zero and sharing it across all ranks is also complicated here as
    torch.distributed is not necessarily initialized yet and there are many forms
    of parallelism to handle (fork/spawn, torch.distributed, SLURM, etc.).

    The filename is different on each node. This is necessary to avoid multiple
    processes writing to the same file in case the nodes use a shared filesystem.
    """
    out_hash = get_sha256(f"{out}-{get_node_rank() or 0}")
    mmap_filepath = (get_data_tmp_dir() / out_hash).with_suffix(".mmap")
    mmap_filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Delete the file if it already exists from a previous run.
        if is_local_rank_zero():
            _unlink_and_ignore(mmap_filepath)

        yield mmap_filepath
    finally:
        if is_local_rank_zero():
            _unlink_and_ignore(mmap_filepath)


def get_dataset_mmap_filenames(
    filenames: Iterable[str],
    mmap_filepath: Path,
) -> MemoryMappedSequence[str]:
    """Returns memory-mapped filenames shared across all ranks.

    Filenames are written to mmap_filepath by rank zero and read by all ranks.
    """
    tmp_path = mmap_filepath.with_suffix(".temp")
    try:
        if is_local_rank_zero():
            # Save filenames to temporary file. Create the final file only once rank zero has
            # finished writing all the filenames.
            memory_mapped_sequence.write_filenames_to_file(
                filenames=filenames,
                mmap_filepath=tmp_path,
            )
            # Rename the temporary file to mmap_filepath.
            tmp_path.replace(mmap_filepath.resolve())
        else:
            # Wait for rank zero to finish writing the filenames.
            timeout_sec = Env.LIGHTLY_TRAIN_MMAP_TIMEOUT_SEC.value
            start_time_sec = time.time()
            while not mmap_filepath.exists():
                if tmp_path.exists():
                    # Reset timeout if the temporary file exists. This means that rank zero
                    # is still writing the filenames.
                    start_time_sec = time.time()

                if timeout_sec >= 0 and time.time() - start_time_sec > timeout_sec:
                    raise RuntimeError(
                        f"Rank {get_global_rank()}: Timeout after {timeout_sec} seconds "
                        f"while waiting for the memory-mapped file '{mmap_filepath}' to be created. "
                        "Please contact Lightly support if this happens. This is most likely a bug. "
                        f"You can increase the timeout with the {Env.LIGHTLY_TRAIN_MMAP_TIMEOUT_SEC.name} "
                        "environment variable. Setting it to -1 disables the timeout. "
                    )
                time.sleep(0.2)
    finally:
        _unlink_and_ignore(tmp_path)

    # Return memory-mapped filenames from file.
    return memory_mapped_sequence.memory_mapped_sequence_from_file(
        mmap_filepath=mmap_filepath
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
        mask_dir=Env.LIGHTLY_TRAIN_MASK_DIR.value,
    )


def _unlink_and_ignore(path: Path) -> None:
    """Unlink a file and ignore the error if it fails.

    Errors can happen if we do not have permission to access the file.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
