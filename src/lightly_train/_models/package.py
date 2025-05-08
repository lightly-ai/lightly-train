#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from torch.nn import Module

from lightly_train._models.model_wrapper import ModelWrapper


@runtime_checkable
class LimitedPackage(Protocol):
    name: str  # The name of the package.

    @classmethod
    def export_model(
        cls, model: Module, out: Path, log_example: bool = True
    ) -> None: ...

    @classmethod
    def is_supported_model(cls, model: Module) -> bool:
        """Check if the model is supported by this package."""
        ...


@runtime_checkable
class FrameworkPackage(LimitedPackage, Protocol):
    @classmethod
    def list_model_names(cls) -> list[str]:
        """List all supported models by this package."""
        ...

    @classmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> Module:
        """Get the model by name.

        Assumes that the model is supported by the package.
        """
        ...

    @classmethod
    def get_model_wrapper(cls, model: Module) -> ModelWrapper:
        """Get the feature extractor class for the model from this package.

        Assumes that the model is supported by the package.
        """
        ...
