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
from typing import Any, Callable, TypedDict

import torch
from torch.nn import Module

from lightly_train._env import Env
from lightly_train._models.fastvit.fastvit import FastViTModelWrapper
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import MultiScaleFeaturePackage

logger = logging.getLogger(__name__)


class _FastViTModelInfo(TypedDict):
    factory_name: str
    default_weights: str | None
    local_path: str | None
    list: bool


_FASTVIT_VARIANTS = (
    "fastvit_t8",
    "fastvit_t12",
    "fastvit_s12",
    "fastvit_sa12",
    "fastvit_sa24",
    "fastvit_sa36",
    "fastvit_ma36",
)

_SUPERVISED_CHECKPOINT_URL = (
    "https://docs-assets.developer.apple.com/ml-research/models/fastvit/"
    "image_classification_models/{variant}.pth.tar"
)
_DISTILLED_CHECKPOINT_URL = (
    "https://docs-assets.developer.apple.com/ml-research/models/fastvit/"
    "image_classification_distilled_models/{variant}.pth.tar"
)

# The checkpoint variants intentionally match the FastViT ImageNet-1K entries
# supported by timm. The official checkpoints are the unfused versions suitable
# for fine-tuning and downstream dense-prediction tasks.
MODEL_NAME_TO_INFO: dict[str, _FastViTModelInfo] = {
    **{
        variant: _FastViTModelInfo(
            factory_name=variant,
            default_weights=None,
            local_path=None,
            list=True,
        )
        for variant in _FASTVIT_VARIANTS
    },
    **{
        f"{variant}-in1k": _FastViTModelInfo(
            factory_name=variant,
            default_weights=_SUPERVISED_CHECKPOINT_URL.format(variant=variant),
            local_path=f"{variant}_in1k.pth.tar",
            list=True,
        )
        for variant in _FASTVIT_VARIANTS
    },
    **{
        f"{variant}-dist-in1k": _FastViTModelInfo(
            factory_name=variant,
            default_weights=_DISTILLED_CHECKPOINT_URL.format(variant=variant),
            local_path=f"{variant}_dist_in1k.pth.tar",
            list=True,
        )
        for variant in _FASTVIT_VARIANTS
    },
}


def _get_model_factory() -> dict[str, Callable[..., Module]]:
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


class FastViTPackage(MultiScaleFeaturePackage):
    name = "fastvit"

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{cls.name}/{model_name}"
            for model_name, info in MODEL_NAME_TO_INFO.items()
            if info["list"]
        ]

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
        if model_name not in MODEL_NAME_TO_INFO:
            raise ValueError(
                f"Unknown FastViT model name: '{model_name}'. "
                f"Supported models: {cls.list_model_names()}."
            )
        model_info = MODEL_NAME_TO_INFO[model_name]
        factory = _get_model_factory()
        args: dict[str, Any] = model_args or {}
        model: Module = factory[model_info["factory_name"]](pretrained=False, **args)
        if load_weights and model_info["default_weights"] is not None:
            weights_path = _maybe_download_weights(model_info)
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("state_dict", checkpoint)
            compatible_state_dict = {
                name: value
                for name, value in state_dict.items()
                if name in model.state_dict()
                and value.shape == model.state_dict()[name].shape
            }
            model.load_state_dict(compatible_state_dict, strict=False)
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


def _maybe_download_weights(model_info: _FastViTModelInfo) -> Path:
    """Return the locally cached official checkpoint for a FastViT preset."""
    url = model_info["default_weights"]
    local_path = model_info["local_path"]
    assert url is not None
    assert local_path is not None

    download_dir = Env.LIGHTLY_TRAIN_MODEL_CACHE_DIR.value.expanduser().resolve()
    download_dest = download_dir / local_path
    if not download_dest.exists():
        download_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "FastViT weights not found locally. Downloading weights from %s to %s",
            url,
            download_dest,
        )
        torch.hub.download_url_to_file(url, dst=str(download_dest))
    return download_dest


# Create singleton instance of the package. The singleton should be used whenever
# possible.
FASTVIT_PACKAGE = FastViTPackage()
