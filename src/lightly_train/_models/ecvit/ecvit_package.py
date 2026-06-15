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
from typing import Any, TypedDict

import torch

from lightly_train._env import Env
from lightly_train._models.ecvit.ecvit import (
    ECVIT_PRESETS,
    ECVIT_PRETRAINED_URLS,
    ECViTModelWrapper,
)
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class _ECViTModelInfo(TypedDict):
    default_weights: str | None
    local_path: str | None
    list: bool


MODEL_NAME_TO_INFO: dict[str, _ECViTModelInfo] = {
    "ecvitt": _ECViTModelInfo(
        default_weights=ECVIT_PRETRAINED_URLS["ecvitt"],
        local_path="ecvitt.pth",
        list=True,
    ),
    "ecvittplus": _ECViTModelInfo(
        default_weights=ECVIT_PRETRAINED_URLS["ecvittplus"],
        local_path="ecvittplus.pth",
        list=True,
    ),
    "ecvits": _ECViTModelInfo(
        default_weights=ECVIT_PRETRAINED_URLS["ecvits"],
        local_path="ecvits.pth",
        list=True,
    ),
    "ecvitsplus": _ECViTModelInfo(
        default_weights=ECVIT_PRETRAINED_URLS["ecvitsplus"],
        local_path="ecvitsplus.pth",
        list=True,
    ),
}


class EdgeCrafterPackage(Package):
    """Package for EdgeCrafter ECViT backbones.

    The public package name exposed in model strings is ``"edgecrafter"``
    (e.g. ``"edgecrafter/ecvitt-ltdetr"``), but the implementation lives under
    ``_models/ecvit`` to keep parity with the existing ECViT port.
    """

    # Public package name used in model strings and in package_helpers lookups.
    name = "edgecrafter"

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{cls.name}/{model_name}"
            for model_name, info in MODEL_NAME_TO_INFO.items()
            if info["list"]
        ]

    @classmethod
    def parse_model_name(cls, model_name: str) -> str:
        if model_name not in MODEL_NAME_TO_INFO:
            raise ValueError(
                f"Unknown EdgeCrafter/ECViT model: '{model_name}'. Available "
                f"models are: {cls.list_model_names()}."
            )
        return model_name

    @classmethod
    def is_supported_model(cls, model: ECViTModelWrapper | ModelWrapper | Any) -> bool:
        return isinstance(model, ECViTModelWrapper)

    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> ECViTModelWrapper:
        """Build an :class:`ECViTModelWrapper` for the given preset.

        Multi-channel input is intentionally not supported: ``num_input_channels``
        must be 3. The parameter is kept on the signature for parity with
        :class:`Package` but is otherwise ignored.
        """
        del model_args  # No ECViT-specific overrides today.
        if num_input_channels != 3:
            raise ValueError(
                "ECViT backbones only support 3 input channels, got "
                f"num_input_channels={num_input_channels}."
            )

        preset_name = cls.parse_model_name(model_name=model_name)
        if preset_name not in ECVIT_PRESETS:
            # Defensive: parse_model_name already enforces this.
            raise ValueError(
                f"Unknown ECViT preset: '{preset_name}'. Available presets: "
                f"{list(ECVIT_PRESETS)}."
            )

        weights_path: PathLike | None = None
        if load_weights:
            weights_path = _maybe_download_weights(
                preset_name=preset_name, model_info=MODEL_NAME_TO_INFO[preset_name]
            )

        return ECViTModelWrapper(name=preset_name, weights_path=weights_path)

    @classmethod
    def get_model_wrapper(cls, model: ECViTModelWrapper) -> ECViTModelWrapper:
        if not cls.is_supported_model(model):
            raise ValueError(
                f"EdgeCrafterPackage cannot create a model wrapper for model of "
                f"type {type(model)}. The model must be an ECViTModelWrapper."
            )
        # ECViTModelWrapper is already a ModelWrapper.
        return model

    @classmethod
    def export_model(
        cls,
        model: ECViTModelWrapper | ModelWrapper | Any,
        out: Path,
        log_example: bool = True,
    ) -> None:
        if isinstance(model, ModelWrapper):
            model = model.get_model()
        if not cls.is_supported_model(model):
            raise ValueError(
                f"EdgeCrafterPackage cannot export model of type {type(model)}. "
                "The model must be an ECViTModelWrapper."
            )
        torch.save(model.state_dict(), out)


# TODO(Guarin, 06/26): Check hash of downloaded weights. Mirrors the same TODO in
# DINOv3Package on first cut.
def _maybe_download_weights(preset_name: str, model_info: _ECViTModelInfo) -> PathLike:
    download_dir = Env.LIGHTLY_TRAIN_MODEL_CACHE_DIR.value.expanduser().resolve()
    url = model_info["default_weights"]
    assert model_info["local_path"] is not None
    assert url is not None
    download_dest = download_dir / model_info["local_path"]
    if not download_dest.exists():
        download_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"ECViT ({preset_name}) weights not found locally. Downloading "
            f"weights from {url} to {download_dest}"
        )
        torch.hub.download_url_to_file(url, dst=str(download_dest))
    return download_dest


# Create singleton instance of the package. The singleton should be used whenever
# possible.
EDGE_CRAFTER_PACKAGE = EdgeCrafterPackage()


# Backwards-compatible alias used internally while we settle on the public name.
ECVIT_PACKAGE = EDGE_CRAFTER_PACKAGE
