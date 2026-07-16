#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Load C-RADIO checkpoints with LightlyTrain's vendored runtime code."""

from __future__ import annotations

import warnings
from importlib import import_module
from types import ModuleType
from typing import Any

import torch
from timm.models import clean_state_dict
from torch.nn import Module

_C_RADIO_V3_MODELS = frozenset(
    {"c-radio_v3-b", "c-radio_v3-l", "c-radio_v3-h", "c-radio_v3-g"}
)
_C_RADIO_V4_MODELS = frozenset({"c-radio_v4-so400m", "c-radio_v4-h"})


def _source_module(model_name: str, module_name: str) -> ModuleType:
    if model_name in _C_RADIO_V3_MODELS:
        source_package = "lightly_train._models.radio.c_radio_v3_src"
    elif model_name in _C_RADIO_V4_MODELS:
        source_package = "lightly_train._models.radio.c_radio_v4_src"
    else:
        raise ValueError(f"Unknown C-RADIO model: {model_name!r}.")
    return import_module(f"{source_package}.{module_name}")


def _get_prefix_state_dict(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def load_radio_model(model_name: str, progress: bool = True) -> Module:
    """Create a pretrained C-RADIO model from its published checkpoint."""
    common = _source_module(model_name, "common")
    radio_model = _source_module(model_name, "radio_model")
    input_conditioner = _source_module(model_name, "input_conditioner")
    feature_normalizer = _source_module(model_name, "feature_normalizer")
    spectral_reparam = _source_module(model_name, "enable_spectral_reparam")

    resource = common.RESOURCE_MAP[model_name]
    checkpoint = torch.hub.load_state_dict_from_url(
        resource.url,
        progress=progress,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise RuntimeError("C-RADIO checkpoint has an unexpected format.")

    args = checkpoint["args"]
    if "state_dict_ema" in checkpoint:
        state_dict = checkpoint["state_dict_ema"]
        args.spectral_reparam = False
        args.spectral_heads = False
        args.damp = None
    else:
        state_dict = checkpoint["state_dict"]

    model = radio_model.create_model_from_args(args)
    model_state_dict = _get_prefix_state_dict(state_dict, "base_model.")

    if getattr(args, "spectral_reparam", False):
        spectral_reparam.configure_spectral_reparam_from_args(
            model, args, state_dict_guidance=model_state_dict
        )

    if getattr(args, "damp", None):
        damp = _source_module(model_name, "enable_damp")
        damp.configure_damp_from_args(model, args)

    state_dict = clean_state_dict(state_dict)
    key_warn = model.load_state_dict(model_state_dict, strict=False)
    if key_warn.missing_keys:
        raise RuntimeError(
            "C-RADIO checkpoint is incompatible with its vendored runtime: "
            f"missing keys={key_warn.missing_keys}."
        )
    if key_warn.unexpected_keys:
        warnings.warn(
            "C-RADIO checkpoint contains unused parameters for the vendored "
            f"runtime: unexpected keys={key_warn.unexpected_keys}.",
            stacklevel=2,
        )

    if getattr(args, "spectral_reparam", False):
        spectral_reparam.disable_spectral_reparam(model)

    conditioner = input_conditioner.get_default_conditioner()
    conditioner.load_state_dict(
        _get_prefix_state_dict(state_dict, "input_conditioner.")
    )

    dtype = getattr(args, "dtype", torch.float32)
    model.to(dtype=dtype)
    conditioner.dtype = dtype

    summary_indices = torch.tensor(
        [
            index
            for index, teacher in enumerate(args.teachers)
            if teacher.get("use_summary", True)
        ],
        dtype=torch.int64,
    )

    feature_normalizer_state = _get_prefix_state_dict(
        state_dict, "_feature_normalizer."
    )
    normalizer: Module | None = None
    if feature_normalizer_state:
        normalizer = feature_normalizer.FeatureNormalizer(
            feature_normalizer_state["mean"].shape[0], dtype=dtype
        )
        normalizer.load_state_dict(feature_normalizer_state)

    intermediate_normalizer_state = _get_prefix_state_dict(
        state_dict, "_intermediate_feature_normalizer."
    )
    intermediate_normalizer: Module | None = None
    if intermediate_normalizer_state:
        intermediate_normalizer = feature_normalizer.IntermediateFeatureNormalizer(
            *intermediate_normalizer_state["means"].shape[:2],
            rot_per_layer=intermediate_normalizer_state["rotation"].ndim == 3,
            dtype=dtype,
        )
        intermediate_normalizer.load_state_dict(intermediate_normalizer_state)

    loaded_model: Module = radio_model.RADIOModel(
        model,
        conditioner,
        summary_idxs=summary_indices,
        patch_size=resource.patch_size,
        max_resolution=resource.max_resolution,
        preferred_resolution=resource.preferred_resolution,
        feature_normalizer=normalizer,
        inter_feature_normalizer=intermediate_normalizer,
    )
    return loaded_model
