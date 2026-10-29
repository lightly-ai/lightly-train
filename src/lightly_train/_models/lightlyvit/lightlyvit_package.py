#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""The network-free LightlyViT model package."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from lightly_train._models import log_usage_example
from lightly_train._models.lightlyvit.lightlyvit import (
    LightlyViT,
    lightlyvit_m16,
    lightlyvit_s16,
    lightlyvit_t16,
)
from lightly_train._models.lightlyvit.lightlyvit_wrapper import LightlyViTModelWrapper
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import MultiScaleFeaturePackage

logger = logging.getLogger(__name__)

# Constructor arguments (beyond the public factory's image/input-channel/drop-path
# set) that select an experimental LightlyViT variant for the efficiency sweep.
# When any of these appear in ``model_args``, :meth:`get_model` builds the variant
# directly through the ``LightlyViT`` constructor instead of the strict public
# factory. This keeps ``lightlyvit/lightlyvit_s16`` (no variant args) exactly
# the shipped model while allowing matched pretraining of the sweep variants
# (e.g. ``model_args={"global_mixer_kind": "linear"}``).
_VARIANT_KWARGS: set[str] = {
    "embed_dim",
    "depth",
    "num_heads",
    "mlp_ratio",
    "mixer_kernel_size",
    "local_mixer_kind",
    "global_mixer_kind",
    "window_size",
    "global_block_indices",
    "num_register_tokens",
    "pos_encoding",
    "rope_base",
    "rope_rescale_coords",
}


def _variant_defaults(
    *,
    embed_dim: int,
    depth: int,
    num_heads: int,
    global_block_indices: tuple[int, ...],
    global_mixer_kind: str,
    pos_encoding: str,
) -> dict[str, Any]:
    """Return the ``LightlyViT`` constructor kwargs for one family variant."""

    return {
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "mlp_ratio": 4.0,
        "mixer_kernel_size": 7,
        "local_mixer_kind": "dw_conv",
        "global_mixer_kind": global_mixer_kind,
        "pos_encoding": pos_encoding,
        "window_size": 7,
        "global_block_indices": global_block_indices,
        "num_register_tokens": 4,
    }


# Per-model defaults. Each model name maps to the recipe its released
# checkpoints were trained with, so weights load without extra model_args:
# T/16 and S/16 use the linear global mixer with RoPE, M/16 the linear mixer
# with a learned position encoding. Each value is (factory, constructor
# defaults) so :meth:`get_model` can either call the strict factory or build a
# variant by overriding the defaults through ``_VARIANT_KWARGS`` in
# ``model_args``.
_MODEL_SPECS: dict[str, tuple[Any, dict[str, Any]]] = {
    "lightlyvit_t16": (
        lightlyvit_t16,
        _variant_defaults(
            embed_dim=224,
            depth=12,
            num_heads=4,
            global_block_indices=(3, 7, 11),
            global_mixer_kind="linear",
            pos_encoding="rope",
        ),
    ),
    "lightlyvit_s16": (
        lightlyvit_s16,
        _variant_defaults(
            embed_dim=432,
            depth=12,
            num_heads=9,
            global_block_indices=(3, 7, 11),
            global_mixer_kind="linear",
            pos_encoding="rope",
        ),
    ),
    "lightlyvit_m16": (
        lightlyvit_m16,
        _variant_defaults(
            embed_dim=640,
            depth=16,
            num_heads=10,
            global_block_indices=(3, 7, 11, 15),
            global_mixer_kind="linear",
            pos_encoding="learned_absolute",
        ),
    ),
}


class LightlyViTPackage(MultiScaleFeaturePackage):
    name = "lightlyvit"
    _MODEL_NAMES = tuple(_MODEL_SPECS.keys())

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [f"{cls.name}/{model}" for model in cls._MODEL_NAMES]

    @classmethod
    def parse_model_name(cls, model_name: str) -> str:
        model_name = model_name.replace("-", "_")
        if model_name not in cls._MODEL_NAMES:
            raise ValueError(
                f"Unknown LightlyViT model '{model_name}'. Available models are: "
                f"{cls.list_model_names()}."
            )
        return model_name

    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> LightlyViT:
        model_name = cls.parse_model_name(model_name)
        factory, base_defaults = _MODEL_SPECS[model_name]
        args = dict(model_args or {})
        weights = args.pop("weights", None)
        image_size = args.pop("image_size", None)
        drop_path_rate = args.pop("drop_path_rate", None)
        # patch_size is fixed at 16 by the factory; tolerate it being passed
        # (e.g. from the LT-DETR backbone_args convention) but reject any other
        # value. The LT-DETR patch_size="auto" resolver reads it from
        # backbone_args, so it must be accepted, not listed as unsupported.
        patch_size = args.pop("patch_size", None)
        if patch_size is not None and int(patch_size) != 16:
            raise ValueError(
                f"LightlyViT only supports patch_size=16, got {patch_size}."
            )
        variant_args = {
            key: args.pop(key) for key in list(args) if key in _VARIANT_KWARGS
        }
        if args:
            raise ValueError(
                f"Unsupported lightlyvit model_args: {sorted(args)}. Allowed keys: "
                f"weights, image_size, drop_path_rate, {sorted(_VARIANT_KWARGS)}."
            )
        if variant_args:
            kwargs = dict(base_defaults)
            kwargs.update(variant_args)
            if image_size is not None:
                kwargs["image_size"] = image_size
            if drop_path_rate is not None:
                kwargs["drop_path_rate"] = drop_path_rate
            model = LightlyViT(in_chans=num_input_channels, patch_size=16, **kwargs)
        else:
            factory_kwargs: dict[str, Any] = {}
            if image_size is not None:
                factory_kwargs["image_size"] = image_size
            if drop_path_rate is not None:
                factory_kwargs["drop_path_rate"] = drop_path_rate
            model = factory(num_input_channels=num_input_channels, **factory_kwargs)
        if load_weights and weights is not None:
            weights_path = Path(weights).expanduser().resolve()
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"LightlyViT weights file not found: {weights_path}"
                )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            logger.info("Loaded LightlyViT weights from %s", weights_path)
        return model

    @classmethod
    def get_model_wrapper(cls, model: LightlyViT) -> LightlyViTModelWrapper:
        if not isinstance(model, LightlyViT):
            raise TypeError(f"Expected LightlyViT, got {type(model)}.")
        return LightlyViTModelWrapper(model)

    @classmethod
    def is_supported_model(cls, model: LightlyViT | ModelWrapper | Any) -> bool:
        if isinstance(model, LightlyViT):
            return True
        return isinstance(model, LightlyViTModelWrapper)

    @classmethod
    def export_model(
        cls,
        model: LightlyViT | ModelWrapper | Any,
        out: Path,
        log_example: bool = True,
    ) -> None:
        if isinstance(model, LightlyViTModelWrapper):
            model = model.get_model()
        if not isinstance(model, LightlyViT):
            raise ValueError(
                f"LightlyViTPackage cannot export model of type {type(model)}."
            )
        torch.save(model.state_dict(), out)
        if log_example:
            lines = [
                "from lightly_train._models.lightlyvit.lightlyvit_package import LightlyViTPackage",
                "import torch",
                "",
                "# Load the pretrained model",
                "# Replace <model_name> with the model name used in train. E.g. 'lightlyvit_s16'.",
                "# If the model was trained with variant kwargs (e.g. global_mixer_kind or",
                "# pos_encoding), pass the same values as model_args={...}.",
                "model = LightlyViTPackage.get_model('<model_name>')",
                f"model.load_state_dict(torch.load('{out}', weights_only=True))",
                "",
                "# Finetune or evaluate the model",
                "...",
            ]
            logger.info(log_usage_example.format_log_msg_model_usage_example(lines))


LIGHTLYVIT_PACKAGE = LightlyViTPackage()
