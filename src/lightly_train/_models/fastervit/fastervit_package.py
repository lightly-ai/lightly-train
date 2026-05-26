#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module

from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package
from lightly_train._models.fastervit.fastervit import FasterViTModelWrapper

logger = logging.getLogger(__name__)


class FasterViTPackage(Package):
    name = "fastervit"

    @classmethod
    def list_model_names(cls) -> list[str]:
        try:
            from fastervit.models.registry import list_models
        except ImportError:
            return []
        return [f"{cls.name}/{model_name}" for model_name in list_models()]

    @classmethod
    def is_supported_model(cls, model: Module | ModelWrapper | Any) -> bool:
        if isinstance(model, ModelWrapper):
            model = model.get_model()
        class_hierarchy = inspect.getmro(model.__class__)
        return any(_cls.__module__.startswith("fastervit") for _cls in class_hierarchy)

    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> Module:
        try:
            import fastervit
        except ImportError:
            raise ValueError(
                f"Cannot create model '{model_name}' because fastervit is not installed."
            )
        args: dict[str, Any] = {"pretrained": load_weights, "in_chans": num_input_channels}
        if model_args is not None:
            args.update(model_args)
        if not load_weights:
            args["pretrained"] = False

        model: Module = fastervit.create_model(model_name, **args)  # type: ignore[arg-type]
        return model

    @classmethod
    def get_model_wrapper(cls, model: Module) -> FasterViTModelWrapper:
        return FasterViTModelWrapper(model)

    @classmethod
    def export_model(
        cls, model: Module | ModelWrapper | Any, out: Path, log_example: bool = True
    ) -> None:
        if isinstance(model, ModelWrapper):
            model = model.get_model()

        if not cls.is_supported_model(model):
            raise ValueError(
                f"FasterViTPackage only supports exporting models of type 'Module' and "
                f"'ModelWrapper', but received '{type(model)}'."
            )

        torch.save(model.state_dict(), out)

        if log_example:
            logger.warning(
                "Usage example cannot be constructed for FasterViT models: "
                "model name cannot be recovered from the model instance. "
                "Load the exported weights manually with fastervit.create_model() "
                f"and model.load_state_dict(torch.load('{out}'))."
            )


# Create singleton instance of the package. The singleton should be used whenever
# possible.
FASTERVIT_PACKAGE = FasterViTPackage()
