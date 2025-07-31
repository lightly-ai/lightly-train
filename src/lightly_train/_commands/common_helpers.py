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
import time
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Iterable, Literal, Sequence, Sized, TypeVar

import torch
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.accelerators.mps import MPSAccelerator
from pytorch_lightning.strategies.strategy import Strategy
from torch.nn import Module
from torch.utils.data import Dataset

from lightly_train import _distributed as distributed_helpers
from lightly_train._data import cache, file_helpers
from lightly_train._data._serialize import memory_mapped_sequence
from lightly_train._data._serialize.memory_mapped_sequence import MemoryMappedSequence
from lightly_train._data.image_dataset import ImageDataset
from lightly_train._embedding.embedding_format import EmbeddingFormat
from lightly_train._env import Env
from lightly_train._models import package_helpers
from lightly_train._models.custom.custom_package import CUSTOM_PACKAGE
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import BasePackage
from lightly_train.types import DatasetItem, PathLike, Transform

logger = logging.getLogger(__name__)


def get_resume_interrupted(resume_interrupted: bool, resume: bool | None) -> bool:
    """Function to handle the deprecated 'resume' argument."""
    if resume is None:
        return resume_interrupted
    elif resume_interrupted == resume:
        # Color code for warning is manually added here because this function is called
        # before the logging is set up.
        logger.warning(
            f"\033[93mresume_interrupted={resume_interrupted} and resume={resume} "
            "should not be set at the same time. Please only set 'resume_interrupted' "
            "as 'resume' is deprecated and will be removed in a future version.\x1b[0m"
        )
        return resume_interrupted
    else:
        raise ValueError(
            f"resume_interrupted={resume_interrupted} and resume={resume} cannot be "
            f"set at the same time! Please only set 'resume_interrupted' as 'resume' "
            "is deprecated and will be removed in a future version."
        )


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


def get_out_dir(out: PathLike, resume_interrupted: bool, overwrite: bool) -> Path:
    out_dir = Path(out).resolve()
    logger.debug(f"Checking if output directory '{out_dir}' exists.")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"Output '{out_dir}' is not a directory!")

        dir_not_empty = any(out_dir.iterdir())

        if (
            dir_not_empty
            and (not (resume_interrupted or overwrite))
            and distributed_helpers.is_global_rank_zero()
        ):
            raise ValueError(
                f"Output '{out_dir}' is not empty! Set overwrite=True to overwrite the "
                "directory or resume_interrupted=True to resume training from an "
                "interrupted or crashed run. "
                "See https://docs.lightly.ai/lightly-train/usage/cli.html#resume-training "
                "for more information on how to resume training."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_tmp_dir() -> Path:
    """Get the temporary directory for Lightly Train."""
    tmp_dir = Env.LIGHTLY_TRAIN_TMP_DIR.value.expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


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
    out_tmp = get_verify_out_tmp_dir() / get_sha256(
        f"{out_dir}-{distributed_helpers.get_node_rank() or 0}"
    )
    logger.debug(f"Creating temporary file '{out_tmp}' to verify out path.")

    try:
        if distributed_helpers.is_local_rank_zero():
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
                        f"Rank {distributed_helpers.get_global_rank()}: Timeout after {timeout_sec} seconds "
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


def remove_excessive_args(
    args: dict[str, Any], limit_keys: set[str] | None = None, num_elems: int = 5
) -> dict[str, Any]:
    """Limit the number of elements in sequences of a dict to a certain number. This is
    strictly for logging purposes. Does not work with nested structures.

    Args:
        args: The dictionary to limit.
        limit_keys: The keys to limit. If None, all keys are limited.
        num_elems: Number of elements to keep in the sequence.

    Returns:
        The dictionary with sequences limited to a certain number of elements.
    """
    if limit_keys is None:
        limit_keys = set(args.keys())
    for key in limit_keys:
        if (
            isinstance(args[key], Sequence)
            and not isinstance(args[key], str)
            and len(args[key]) > num_elems
        ):
            args[key] = type(args[key])(
                (*args[key][: num_elems - 2], "...", args[key][-1])
            )
    return args


def sanitize_config_dict(args: dict[str, Any]) -> dict[str, Any]:
    """Replace classes with their names in the train config dictionary."""
    model = args.get("model")
    if model is not None and not isinstance(model, str):
        args["model"] = model.__class__.__name__
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
        # Handle SLURM and respect SLURM_CPUS_PER_TASK setting
        slurm_cpus_per_task = Env.SLURM_CPUS_PER_TASK.value
        if slurm_cpus_per_task is not None:
            # Leave 1 CPU for the main process on every device
            return max(slurm_cpus_per_task - 1, 0)

        num_cpus_per_device = _get_num_cpus_per_device(
            num_devices_per_node=num_devices_per_node
        )
        if num_cpus_per_device is None:
            num_workers_auto = Env.LIGHTLY_TRAIN_DEFAULT_NUM_WORKERS_AUTO.value
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
        return Env.SLURM_CPUS_PER_TASK.value
    else:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            cpu_count = cpu_count // num_devices_per_node
    return cpu_count


def _is_slurm() -> bool:
    return Env.SLURM_JOB_ID.value is not None


class ModelPart(Enum):
    MODEL = "model"
    WRAPPED_MODEL = "wrapped_model"
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


def export_model(
    model: Module | ModelWrapper | EmbeddingModel,
    format: ModelFormat,
    out: Path,
    package: BasePackage | None = None,
    log_example: bool = True,
) -> None:
    if not distributed_helpers.is_global_rank_zero():
        return

    logger.debug(f"Exporting model to '{out}' in format '{format}'.")
    out.parent.mkdir(parents=True, exist_ok=True)
    if format == ModelFormat.TORCH_MODEL:
        torch.save(model, out)
    elif format == ModelFormat.TORCH_STATE_DICT:
        torch.save(model.state_dict(), out)
    elif format == ModelFormat.PACKAGE_DEFAULT:
        if package is None:
            raise ValueError(
                "Package must be provided when exporting in package default format."
            )
        if isinstance(model, EmbeddingModel):
            model = model.wrapped_model.get_model()
        elif isinstance(model, ModelWrapper):
            model = model.get_model()
        package.export_model(model=model, out=out, log_example=log_example)
    else:
        raise ValueError(f"Invalid format: '{format.value}' is not supported ")


def _get_package(model: Module) -> BasePackage:
    # Reimplementation of package_helpers.get_package_from_model, but with a fallback
    # to the custom package if the model is not part of any package instead of raising
    # an error.
    for package in package_helpers.list_packages():
        if package.is_supported_model(model):
            return package
    return CUSTOM_PACKAGE


@contextlib.contextmanager
def get_dataset_temp_mmap_path(
    data: PathLike | Sequence[PathLike],
) -> Generator[Path, Any, Any]:
    """Generate file in temporary directory to be used for memory-mapping the dataset.

    Creates a unique filename for the memory-mapped file based on the data arg.
    We use the data arg as a deterministic value that is consistent across all ranks
    on the same node. Additionally, we can cache the file if required, since the hash
    directly reflects the used config.

    We need a deterministic value from "outside" at this point in the code as the
    code might already be running on multiple processes depending on how it was
    started. We cannot create a new filename based on a random value as this would
    create a different filename for each process. Creating the filename on global
    rank zero and sharing it across all ranks is also complicated here as
    torch.distributed is not necessarily initialized yet and there are many forms
    of parallelism to handle (fork/spawn, torch.distributed, SLURM, etc.).

    The filename is different on each node. This is necessary to avoid multiple
    processes writing to the same file in case the nodes use a shared filesystem.
    """
    out_hash = get_sha256(f"{data}-{distributed_helpers.get_node_rank() or 0}")
    mmap_filepath = (cache.get_data_cache_dir() / out_hash).with_suffix(".mmap")
    mmap_filepath.parent.mkdir(parents=True, exist_ok=True)

    reuse_file = Env.LIGHTLY_TRAIN_MMAP_REUSE_FILE.value
    try:
        # Delete the file if it already exists from a previous run.
        if not reuse_file and distributed_helpers.is_local_rank_zero():
            _unlink_and_ignore(mmap_filepath)

        yield mmap_filepath
    finally:
        if not reuse_file and distributed_helpers.is_local_rank_zero():
            _unlink_and_ignore(mmap_filepath)


def get_dataset_mmap_filenames(
    filenames: Iterable[str],
    mmap_filepath: Path,
) -> MemoryMappedSequence[str]:
    """Returns memory-mapped filenames shared across all ranks.

    Filenames are written to mmap_filepath by rank zero and read by all ranks.
    """
    if Env.LIGHTLY_TRAIN_MMAP_REUSE_FILE.value and mmap_filepath.exists():
        # If the file already exists and we are allowed to reuse it, return it.
        logger.warning(f"Reusing existing memory-mapped file '{mmap_filepath}'.")
        return memory_mapped_sequence.memory_mapped_sequence_from_file(
            mmap_filepath=mmap_filepath
        )

    tmp_path = mmap_filepath.with_suffix(".temp")
    try:
        if distributed_helpers.is_local_rank_zero():
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
                        f"Rank {distributed_helpers.get_global_rank()}: Timeout after {timeout_sec} seconds "
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
    data: PathLike | Sequence[PathLike] | Dataset[DatasetItem],
    transform: Transform,
    mmap_filepath: Path | None,
) -> Dataset[DatasetItem]:
    if isinstance(data, Dataset):
        logger.debug("Using provided dataset.")
        return data

    if mmap_filepath is None:
        raise ValueError("Memory-mapped file path must be provided.")

    mask_dir = Env.LIGHTLY_TRAIN_MASK_DIR.value

    if isinstance(data, (str, Path)):
        data = Path(data).resolve()
        if not data.exists():
            raise ValueError(f"Data directory '{data}' does not exist!")
        elif not data.is_dir():
            raise ValueError(f"Data path '{data}' is not a directory!")
        elif data.is_dir() and not any(data.iterdir()):
            raise ValueError(f"Data directory '{data}' is empty!")
        # Use relative paths as filenames when a single directory or file is provided to
        # reduce the file size.
        # NOTE(Guarin, 01/25): The bottleneck for dataset initialization is filename
        # listing and not the memory mapping. Listing the train set from ImageNet takes
        # about 30 seconds. This is mostly because os.walk is not parallelized.
        filenames = file_helpers.list_image_filenames(image_dir=data)
        return ImageDataset(
            image_dir=data,
            image_filenames=get_dataset_mmap_filenames(
                filenames=filenames,
                mmap_filepath=mmap_filepath,
            ),
            transform=transform,
            mask_dir=Path(mask_dir) if mask_dir is not None else None,
        )

    elif isinstance(data, Sequence):
        _data: Sequence[Path] = [Path(d).resolve() for d in data]
        if mask_dir is not None:
            raise ValueError(
                "Mask directory is not supported when multiple directories or files "
                "are provided."
            )

        for d in _data:
            if not d.exists():
                raise ValueError(f"Data directory or file '{d}' does not exist!")
            elif d.is_dir() and not any(d.iterdir()):
                raise ValueError(f"Data directory '{d}' is empty!")
        files = file_helpers.list_image_files(imgs_and_dirs=_data)
        filenames = file_helpers.list_image_filenames(files=files)
        return ImageDataset(
            image_dir=None,
            image_filenames=get_dataset_mmap_filenames(
                filenames=filenames,
                mmap_filepath=mmap_filepath,
            ),
            transform=transform,
        )
    else:
        raise ValueError(
            "Data must be a directory, a list of directories or files, or a dataset."
        )


def _unlink_and_ignore(path: Path) -> None:
    """Unlink a file and ignore the error if it fails.

    Errors can happen if we do not have permission to access the file.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


_T = TypeVar("_T")


def get_global_batch_size(
    global_batch_size: int,
    dataset: Dataset[_T],
    total_num_devices: int,
    loader_args: dict[str, Any] | None,
) -> int:
    """Calculates the global batch size based on the dataset size and number of
    available nodes and devices.

    Args:
        global_batch_size:
            The global batch size. This is the total batch size across all nodes and
            devices.
        dataset:
            Dataset. If the dataset is smaller than the global batch size, the global
            batch size is reduced to the dataset size.
        total_num_devices:
            The total number of devices across all nodes.
        loader_args:
            Additional arguments for the DataLoader. If the batch size is provided in
            loader_args, the global batch size is calculated based on this value as
            loader_args["batch_size"] * total_num_devices.

    Raises:
        ValueError: If the global batch size is not divisible by total_num_devices.
    """
    if loader_args is not None and "batch_size" in loader_args:
        # Don't do fancy calculations if the user provides a fixed batch size for the
        # dataloader.
        batch_size_per_device = loader_args["batch_size"]
        global_batch_size = batch_size_per_device * total_num_devices
        logger.debug(
            f"Got batch size per device {batch_size_per_device} based on loader_args. "
        )
        logger.debug(f"Using global batch size {global_batch_size}.")
        return global_batch_size

    # Limit batch size for small datasets.
    if isinstance(dataset, Sized):
        dataset_size = len(dataset)
        logger.debug(f"Detected dataset size {dataset_size}.")
        if dataset_size < global_batch_size:
            old_global_batch_size = global_batch_size
            global_batch_size = dataset_size
            logger.warning(
                f"Detected dataset size {dataset_size} and batch size "
                f"{old_global_batch_size}. Reducing batch size to {global_batch_size}."
            )

    if global_batch_size % total_num_devices != 0:
        raise ValueError(
            f"Batch size {global_batch_size} must be divisible by "
            f"(num_nodes * devices) = {total_num_devices}."
        )
    return global_batch_size
