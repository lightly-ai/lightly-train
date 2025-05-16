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

from torch.nn import Module

from lightly_train._models.model_wrapper import ModelWrapper


class BasePackage(ABC):
    name: str  # The name of the package.

    @classmethod
    @abstractmethod
    def export_model(
        cls, model: Module, out: Path, log_example: bool = True
    ) -> None: ...

    @classmethod
    @abstractmethod
    def is_supported_model(cls, model: Module | ModelWrapper) -> bool:
        """Check if the model is supported by this package."""
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
    ) -> Module:
        """Get the model by name.

        Assumes that the model is supported by the package.
        """
        ...

    @classmethod
    @abstractmethod
    def get_model_wrapper(cls, model: Module) -> ModelWrapper:
        """Get the feature extractor class for the model from this package.

        Assumes that the model is supported by the package.
        """
        ...
