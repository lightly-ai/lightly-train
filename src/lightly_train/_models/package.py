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

from lightly_train._models.feature_extractor import FeatureExtractor


class Package(ABC):
    """Interface for a package that provides models and feature extractors that are
    compatible with lightly_train.

    Every package must implement this interface.
    """

    name: str  # The name of the package.

    @classmethod
    @abstractmethod
    def list_model_names(cls) -> list[str]:
        """List all supported models by this package."""
        ...

    @classmethod
    @abstractmethod
    def is_supported_model(cls, model: Module) -> bool:
        """Check if the model is supported by this package."""
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
    def get_feature_extractor(cls, model: Module) -> FeatureExtractor:
        """Get the feature extractor class for the model from this package.

        Assumes that the model is supported by the package.
        """
        ...

    @classmethod
    def export_model(cls, model: Module, out: Path) -> None:
        raise NotImplementedError(f"Exporting {cls.name} models is not supported.")
