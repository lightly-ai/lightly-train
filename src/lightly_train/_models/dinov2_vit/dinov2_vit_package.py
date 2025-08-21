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

from lightly_train._data import cache
from lightly_train._models import log_usage_example
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_src.configs import (
    MODELS as VIT_MODELS,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.configs import (
    get_config_path,
    load_and_merge_config,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.dinov2_helper import load_weights
from lightly_train._models.dinov2_vit.dinov2_vit_src.models import (
    vision_transformer as vits,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package

logger = logging.getLogger(__name__)


class DINOv2ViTPackage(Package):
    name = "dinov2"

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{cls.name}/{model_name}"
            for model_name, model_info in VIT_MODELS.items()
            if model_info["list"]
        ]

    @classmethod
    def is_supported_model(
        cls, model: DinoVisionTransformer | ModelWrapper | Any
    ) -> bool:
        if isinstance(model, ModelWrapper):
            return isinstance(model.get_model(), DinoVisionTransformer)
        return isinstance(model, DinoVisionTransformer)

    @classmethod
    def parse_model_name(cls, model_name: str) -> str:
        # Replace "_" with "-" for backwards compatibility.
        # - "vitb14_pretrained" -> "vitb14-pretrained"
        # - "_vittest14_pretrained" -> "_vittest14-pretrained"
        # We keep leading underscores for private test models.
        if model_name:
            model_name = model_name[0] + model_name[1:].replace("_", "-")
        # Replace "-pretrain" with "-pretrained" suffix for backwards compatibility.
        if model_name.endswith("-pretrain"):
            model_name = model_name[: -len("-pretrain")]
        model_info = VIT_MODELS.get(model_name)
        if model_info is None:
            raise ValueError(
                f"Unknown model: {model_name} available models are: {cls.list_model_names()}"
            )
        # Map to original model name if current name is an alias.
        model_name = model_info.get("alias_for", model_name)
        return model_name

    @classmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> DinoVisionTransformer:
        """
        Get a DINOv2 ViT model by name. Here the student version is build.
        """
        model_name = cls.parse_model_name(model_name)

        # Get the model cfg
        config_path = get_config_path(config_name=VIT_MODELS[model_name]["config"])
        cfg = load_and_merge_config(str(config_path))

        # Build the model using the cfg
        model_builders = {
            "_vit_test": vits._vit_test,
            "vit_small": vits.vit_small,
            "vit_base": vits.vit_base,
            "vit_large": vits.vit_large,
            "vit_giant2": vits.vit_giant2,
        }
        model_builder = model_builders.get(cfg.student.arch, None)
        if model_builder is None:
            raise TypeError(f"Unsupported architecture type {cfg.student.arch}.")

        kwargs = dict(
            img_size=cfg.crops.global_crops_size,
            patch_size=cfg.student.patch_size,
            init_values=cfg.student.layerscale,
            ffn_layer=cfg.student.ffn_layer,
            block_chunks=cfg.student.block_chunks,
            qkv_bias=cfg.student.qkv_bias,
            proj_bias=cfg.student.proj_bias,
            ffn_bias=cfg.student.ffn_bias,
            num_register_tokens=cfg.student.num_register_tokens,
            interpolate_offset=cfg.student.interpolate_offset,
            interpolate_antialias=cfg.student.interpolate_antialias,
            drop_path_rate=cfg.student.drop_path_rate,
            drop_path_uniform=cfg.student.drop_path_uniform,
        )
        kwargs.update(model_args or {})

        model = model_builder(**kwargs)

        # Load weights if available.
        weights_url = VIT_MODELS[model_name].get("url")
        if weights_url:
            checkpoint_dir = cache.get_model_cache_dir()
            model = load_weights(
                model=model,
                checkpoint_dir=checkpoint_dir,
                url=weights_url,
            )

        return model

    @classmethod
    def get_model_wrapper(cls, model: DinoVisionTransformer) -> DINOv2ViTModelWrapper:
        return DINOv2ViTModelWrapper(model=model)

    @classmethod
    def export_model(
        cls,
        model: DinoVisionTransformer | ModelWrapper | Any,
        out: Path,
        log_example: bool = True,
    ) -> None:
        if isinstance(model, ModelWrapper):
            model = model.get_model()

        if not cls.is_supported_model(model):
            raise ValueError(
                f"DINOv2ViTPackage cannot export model of type {type(model)}. "
                "The model must be a ModelWrapper or a DinoVisionTransformer."
            )

        torch.save(model.state_dict(), out)

        if log_example:
            log_message_code = [
                "from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage",
                "import torch",
                "",
                "# Load the pretrained model",
                "model = DINOv2ViTPackage.get_model('dinov2/<vitXX>') # Replace with the model name used in train",
                f"model.load_state_dict(torch.load('{out}', weights_only=True))",
                "",
                "# Finetune or evaluate the model",
                "...",
            ]
            logger.info(
                log_usage_example.format_log_msg_model_usage_example(log_message_code)
            )


# Create singleton instance of the package. The singleton should be used whenever
# possible.
DINOV2_VIT_PACKAGE = DINOv2ViTPackage()
