#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import Module

from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package
from lightly_train._models.radio.radio import RadioModelWrapper
from lightly_train._models.radio.radio_loader import load_radio_model

_RADIO_MODEL_MARKER = "_lightly_train_radio_model_name"

MODEL_NAMES = (
    "c-radio_v3-b",
    "c-radio_v3-l",
    "c-radio_v3-h",
    "c-radio_v3-g",
    "c-radio_v4-so400m",
    "c-radio_v4-h",
)


class RadioPackage(Package):
    """Package for NVIDIA C-RADIO models with vendored runtime code."""

    name = "radio"

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [f"{cls.name}/{model_name}" for model_name in MODEL_NAMES]

    @classmethod
    def is_supported_model(cls, model: Module | ModelWrapper | Any) -> bool:
        if isinstance(model, RadioModelWrapper):
            return True
        if isinstance(model, ModelWrapper):
            model = model.get_model()
        return isinstance(model, Module) and hasattr(model, _RADIO_MODEL_MARKER)

    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> Module:
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f"Unknown RADIO model: '{model_name}'. Available models are: "
                f"{cls.list_model_names()}."
            )
        if num_input_channels != 3:
            raise ValueError(
                "RADIO backbones only support 3 input channels, got "
                f"num_input_channels={num_input_channels}."
            )
        if not load_weights:
            raise ValueError(
                "C-RADIO models are distributed with pretrained weights; "
                "load_weights=False is not supported."
            )

        if model_args:
            raise ValueError(f"Unsupported C-RADIO model_args: {sorted(model_args)}.")

        model = load_radio_model(model_name, progress=True)
        if not isinstance(model, Module):
            raise RuntimeError(
                "C-RADIO loader returned an object that is not a torch.nn.Module."
            )
        setattr(model, _RADIO_MODEL_MARKER, model_name)
        return model

    @classmethod
    def get_model_wrapper(cls, model: Module) -> RadioModelWrapper:
        if not cls.is_supported_model(model):
            raise ValueError(
                f"RadioPackage cannot create a model wrapper for model of type "
                f"{type(model)}. Load it with RadioPackage.get_model()."
            )
        if isinstance(model, RadioModelWrapper):
            return model
        return RadioModelWrapper(model=model)

    @classmethod
    def export_model(
        cls,
        model: Module | ModelWrapper | Any,
        out: Path,
        log_example: bool = True,
    ) -> None:
        if isinstance(model, ModelWrapper):
            model = model.get_model()
        if not cls.is_supported_model(model):
            raise ValueError(
                f"RadioPackage cannot export model of type {type(model)}. The model "
                "must be loaded with RadioPackage.get_model()."
            )
        torch.save(model.state_dict(), out)


RADIO_PACKAGE = RadioPackage()
