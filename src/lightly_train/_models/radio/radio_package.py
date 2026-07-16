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

RADIO_TORCH_HUB_REVISION = "c0f37017930e9dda53f93424cf4bf39fc51f287e"
_RADIO_MODEL_MARKER = "_lightly_train_radio_hub_revision"

MODEL_NAMES = (
    "radio_v1",
    "radio_v2.5-b",
    "radio_v2.5-l",
    "radio_v2.5-h",
    "radio_v2.5-h-norm",
    "radio_v2.5-g",
    "c-radio_v3-b",
    "c-radio_v3-l",
    "c-radio_v3-h",
    "c-radio_v3-g",
    "c-radio_v4-so400m",
    "c-radio_v4-h",
)


class RadioPackage(Package):
    """Package for NVIDIA RADIO models loaded through Torch Hub."""

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
                "RADIO Torch Hub models are distributed with pretrained weights; "
                "load_weights=False is not supported."
            )

        args = dict(model_args or {})
        hub_ref = args.pop("hub_ref", RADIO_TORCH_HUB_REVISION)
        if args:
            raise ValueError(
                "Unsupported RADIO model_args: "
                f"{sorted(args)}. Only 'hub_ref' is supported."
            )

        model = torch.hub.load(
            f"NVlabs/RADIO:{hub_ref}",
            "radio_model",
            version=model_name,
            progress=True,
            skip_validation=True,
        )  # type: ignore[no-untyped-call]
        if not isinstance(model, Module):
            raise RuntimeError(
                "RADIO Torch Hub returned an object that is not a torch.nn.Module."
            )
        setattr(model, _RADIO_MODEL_MARKER, hub_ref)
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
