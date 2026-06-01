#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module

from lightly_train._models.fastvit.fastvit import FastViTModelWrapper
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package

logger = logging.getLogger(__name__)

_FASTVIT_MODEL_NAMES = [
    "fastvit_t8",
    "fastvit_t12",
    "fastvit_s12",
    "fastvit_sa12",
    "fastvit_sa24",
    "fastvit_sa36",
    "fastvit_ma36",
]


def _get_model_factory() -> dict[str, Any]:
    from lightly_train._models.fastvit.components.models.fastvit import (
        fastvit_ma36,
        fastvit_s12,
        fastvit_sa12,
        fastvit_sa24,
        fastvit_sa36,
        fastvit_t8,
        fastvit_t12,
    )

    return {
        "fastvit_t8": fastvit_t8,
        "fastvit_t12": fastvit_t12,
        "fastvit_s12": fastvit_s12,
        "fastvit_sa12": fastvit_sa12,
        "fastvit_sa24": fastvit_sa24,
        "fastvit_sa36": fastvit_sa36,
        "fastvit_ma36": fastvit_ma36,
    }


class FastViTPackage(Package):
    name = "fastvit"

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [f"{cls.name}/{n}" for n in _FASTVIT_MODEL_NAMES]

    @classmethod
    def is_supported_model(cls, model: Module | ModelWrapper | Any) -> bool:
        if isinstance(model, ModelWrapper):
            model = model.get_model()
        try:
            from lightly_train._models.fastvit.components.models.fastvit import FastViT

            return isinstance(model, FastViT)
        except ImportError:
            return False

    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> Module:
        if num_input_channels != 3:
            raise ValueError(
                f"FastViT only supports 3 input channels, but got {num_input_channels}. "
                "The convolutional stem is hardcoded to 3 input channels."
            )
        if load_weights:
            logger.warning(
                "FastViT does not provide pretrained weights in this integration. "
                "The model will be initialized with random weights."
            )
        factory = _get_model_factory()
        if model_name not in factory:
            raise ValueError(
                f"Unknown FastViT model name: '{model_name}'. "
                f"Supported models: {list(factory)}."
            )
        args: dict[str, Any] = model_args or {}
        model: Module = factory[model_name](pretrained=False, **args)
        return model

    @classmethod
    def get_model_wrapper(cls, model: Module) -> FastViTModelWrapper:
        return FastViTModelWrapper(model)

    @classmethod
    def export_model(
        cls, model: Module | ModelWrapper | Any, out: Path, log_example: bool = True
    ) -> None:
        if isinstance(model, ModelWrapper):
            model = model.get_model()

        if not cls.is_supported_model(model):
            raise ValueError(
                "FastViTPackage only supports exporting FastViT models, "
                f"but received '{type(model)}'."
            )

        torch.save(model.state_dict(), out)

        if log_example:
            logger.warning(
                "Usage example cannot be constructed for FastViT models: "
                "model name cannot be recovered from the model instance. "
                "Load the exported weights manually with the factory function "
                f"and model.load_state_dict(torch.load('{out}'))."
            )


# Create singleton instance of the package. The singleton should be used whenever
# possible.
FASTVIT_PACKAGE = FastViTPackage()
