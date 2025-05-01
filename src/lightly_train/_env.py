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

    @property
    def value(self) -> T:
        raw = self.raw_value
        return self.type_(raw) if raw is not None else self.default

    @property
    def raw_value(self) -> str | None:
        return os.getenv(self.name)


class Env:
    LIGHTLY_TRAIN_CACHE_DIR = EnvVar[Path](
        name="LIGHTLY_TRAIN_CACHE_DIR",
        default=Path.home() / ".cache" / "lightly-train",
        type_=Path,
    )
    LIGHTLY_TRAIN_MASK_DIR = EnvVar[Path | None](
        name="LIGHTLY_TRAIN_MASK_DIR",
        default=None,
        type_=Path,
    )
    LIGHTLY_TRAIN_MMAP_TIMEOUT_SEC = EnvVar[float](
        name="LIGHTLY_TRAIN_MMAP_TIMEOUT_SEC",
        default=300,
        type_=float,
    )
    LIGHTLY_TRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC = EnvVar[float](
        name="LIGHTLY_TRAIN_VERIFY_OUT_DIR_TIMEOUT_SEC",
        default=30,
        type_=float,
    )
