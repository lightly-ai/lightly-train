#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class EnvVar(Generic[T]):
    name: str
    default: T
    type_: Callable[[str], T]
    # If True, empty strings are converted to the default value. This happens for:
    # MY_ENV_VAR=
    # MY_ENV_VAR=""
    convert_empty_str_to_default: bool = True

    @property
    def value(self) -> T:
        """Returns the value of the environment variable converted to its type."""
        raw = os.getenv(self.name)
        if raw is None:
            return self.default
        elif self.convert_empty_str_to_default and raw == "":
            return self.default
        else:
            return self.type_(raw)

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
    LIGHTLY_TRAIN_CACHE_DIR: EnvVar[Path] = EnvVar(
        name="LIGHTLY_TRAIN_CACHE_DIR",
        default=Path.home() / ".cache" / "lightly-train",
        type_=Path,
    )
    LIGHTLY_TRAIN_MASK_DIR: EnvVar[Path | None] = EnvVar(
        name="LIGHTLY_TRAIN_MASK_DIR",
        default=None,
        type_=Path,
    )
    LIGHTLY_TRAIN_MAX_NUM_WORKERS_AUTO: EnvVar[int] = EnvVar(
        name="LIGHTLY_TRAIN_MAX_NUM_WORKERS_AUTO",
        default=8,
        type_=int,
    )
    LIGHTLY_TRAIN_MMAP_TIMEOUT_SEC: EnvVar[float] = EnvVar(
        name="LIGHTLY_TRAIN_MMAP_TIMEOUT_SEC",
        default=300,
        type_=float,
    )
    LIGHTLY_TRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC: EnvVar[float] = EnvVar(
        name="LIGHTLY_TRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC",
        default=30,
        type_=float,
    )
    SLURM_CPUS_PER_TASK: EnvVar[int | None] = EnvVar(
        name="SLURM_CPUS_PER_TASK",
        default=None,
        type_=int,
    )
