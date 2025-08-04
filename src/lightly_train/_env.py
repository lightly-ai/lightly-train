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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class EnvVar(Generic[T]):
    name: str
    _default: Callable[[], T] | T
    _type: Callable[[str], T]
    # If True, empty strings are converted to the default value. This happens for:
    # MY_ENV_VAR=
    # MY_ENV_VAR=""
    convert_empty_str_to_default: bool = True

    @property
    def default(self) -> T:
        """Returns the default value of the environment variable."""
        return self._default() if callable(self._default) else self._default

    @property
    def value(self) -> T:
        """Returns the value of the environment variable converted to its type."""
        raw = os.getenv(self.name)
        if raw is None:
            return self.default
        elif self.convert_empty_str_to_default and raw == "":
            return self.default
        else:
            return self._type(raw)

    @property
    def raw_value(self) -> str | None:
        """Returns the raw value of the environment variable as a string.

        Returns None if the variable is not set and has no default value.
        """
        raw = os.getenv(self.name)
        return (
            raw
            if raw is not None
            else str(self.default)
            if self.default is not None
            else None
        )


class Env:
    LIGHTLY_TRAIN_LOG_LEVEL: EnvVar[str] = EnvVar(
        name="LIGHTLY_TRAIN_LOG_LEVEL",
        _default=logging.getLevelName(logging.INFO),
        _type=str,
    )
    # Base directory for all Lightly Train cache files.
    LIGHTLY_TRAIN_CACHE_DIR: EnvVar[Path] = EnvVar(
        name="LIGHTLY_TRAIN_CACHE_DIR",
        _default=Path.home() / ".cache" / "lightly-train",
        _type=Path,
    )
    # Path to directory where weights of pretrained models are cached.
    LIGHTLY_TRAIN_MODEL_CACHE_DIR: EnvVar[Path] = EnvVar(
        name="LIGHTLY_TRAIN_MODEL_CACHE_DIR",
        _default=lambda: Env.LIGHTLY_TRAIN_CACHE_DIR.value / "models",
        _type=Path,
    )
    # Path to directory where temporary files are stored.
    # TODO(Lionel, 08/25): Use a true temporary directory instead. Kept like this for
    # now to avoid breaking changes.
    LIGHTLY_TRAIN_TMP_DIR: EnvVar[Path] = EnvVar(
        name="LIGHTLY_TRAIN_TMP_DIR",
        _default=lambda: Env.LIGHTLY_TRAIN_CACHE_DIR.value / "data",
        _type=Path,
    )
    # Path to directory where data is cached. These are mainly the memory-mapped files.
    # TODO(Lionel, 08/25): Change the default to LIGHTLY_TRAIN_CACHE_DIR.value / "data".
    LIGHTLY_TRAIN_DATA_CACHE_DIR: EnvVar[Path] = EnvVar(
        name="LIGHTLY_TRAIN_DATA_CACHE_DIR",
        _default=lambda: Env.LIGHTLY_TRAIN_TMP_DIR.value,
        _type=Path,
    )
    # Timeout in seconds for the dataloader to collect a batch from the workers. This is
    # used to prevent the dataloader from hanging indefinitely. Set to 0 to disable the
    # timeout.
    LIGHTLY_TRAIN_DATALOADER_TIMEOUT_SEC: EnvVar[int] = EnvVar(
        name="LIGHTLY_TRAIN_DATALOADER_TIMEOUT_SEC",
        _default=180,
        _type=int,
    )
    # Mode in which images are loaded. This can be "RGB" to load images in RGB or
    # "UNCHANGED" to load images in their original format without any conversion.
    LIGHTLY_TRAIN_IMAGE_MODE: EnvVar[str] = EnvVar(
        name="LIGHTLY_TRAIN_IMAGE_MODE",
        _default="RGB",
        _type=str,
    )
    LIGHTLY_TRAIN_MASK_DIR: EnvVar[Path | None] = EnvVar(
        name="LIGHTLY_TRAIN_MASK_DIR",
        _default=None,
        _type=Path,
    )
    # Maximum number of workers in case num_workers is set to "auto".
    LIGHTLY_TRAIN_MAX_NUM_WORKERS_AUTO: EnvVar[int] = EnvVar(
        name="LIGHTLY_TRAIN_MAX_NUM_WORKERS_AUTO",
        _default=8,
        _type=int,
    )
    # Default number of workers in case num_workers is set to "auto" but LightlyTrain
    # cannot automatically determined the number of available CPUs.
    LIGHTLY_TRAIN_DEFAULT_NUM_WORKERS_AUTO: EnvVar[int] = EnvVar(
        name="LIGHTLY_TRAIN_DEFAULT_NUM_WORKERS_AUTO",
        _default=8,
        _type=int,
    )
    LIGHTLY_TRAIN_MMAP_TIMEOUT_SEC: EnvVar[float] = EnvVar(
        name="LIGHTLY_TRAIN_MMAP_TIMEOUT_SEC",
        _default=300,
        _type=float,
    )
    LIGHTLY_TRAIN_MMAP_REUSE_FILE: EnvVar[bool] = EnvVar(
        name="LIGHTLY_TRAIN_MMAP_REUSE_FILE",
        _default=False,
        _type=lambda x: x.lower() in ("true", "t", "1", "yes", "y"),
    )
    LIGHTLY_TRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC: EnvVar[float] = EnvVar(
        name="LIGHTLY_TRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC",
        _default=30,
        _type=float,
    )
    LIGHTLY_TRAIN_DOWNLOAD_CHUNK_TIMEOUT_SEC: EnvVar[float] = EnvVar(
        name="LIGHTLY_TRAIN_DOWNLOAD_CHUNK_TIMEOUT_SEC",
        _default=180,
        _type=float,
    )
    MLFLOW_TRACKING_URI: EnvVar[str | None] = EnvVar(
        name="MLFLOW_TRACKING_URI",
        _default=None,
        _type=str,
    )
    SLURM_CPUS_PER_TASK: EnvVar[int | None] = EnvVar(
        name="SLURM_CPUS_PER_TASK",
        _default=None,
        _type=int,
    )
    SLURM_JOB_ID: EnvVar[str | None] = EnvVar(
        name="SLURM_JOB_ID",
        _default=None,
        _type=str,
    )
