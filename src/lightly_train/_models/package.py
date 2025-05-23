#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train.types import PackageModel


class BasePackage(ABC):
    name: str  # The name of the package.

    @classmethod
    @abstractmethod
    def export_model(
        cls,
        model: PackageModel | ModelWrapper | Any,
        out: Path,
        log_example: bool = True,
    ) -> None:
        """Export the model in the package's format. The model can be either a
        ModelWrapper or one of the package's underlying models.
        """
        ...

    @classmethod
    @abstractmethod
    def is_supported_model(cls, model: PackageModel | ModelWrapper | Any) -> bool:
        """Check if the model is either a ModelWrapper or one of the package's
        underlying models.
        """
        ...


class Package(BasePackage):
    @classmethod
    @abstractmethod
    def list_model_names(cls) -> list[str]:
        """List all supported models by this package."""
        ...

    @classmethod
    @abstractmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> PackageModel:
        """Get the underlying model of the package by its name."""
        ...

    @classmethod
    @abstractmethod
    def get_model_wrapper(cls, model: PackageModel) -> ModelWrapper:
        """Wrap the underlying model with the ModelWrapper."""
        ...
