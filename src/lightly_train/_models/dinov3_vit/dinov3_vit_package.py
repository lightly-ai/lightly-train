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
from lightly_train._models.dinov3_vit.dinov3_vit import DINOv3ViTModelWrapper
from lightly_train._models.dinov3_vit.dinov3_vit_src.dinov3_helper import load_weights
from lightly_train._models.dinov3_vit.dinov3_vit_src.models import (
    vision_transformer as vits,
)
from lightly_train._models.dinov3_vit.dinov3_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package

logger = logging.getLogger(__name__)


class DINOv3ViTPackage(Package):
    name = "dinov3"

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{cls.name}/vits16"
        ]

    @classmethod
    def is_supported_model(
        cls, model: DinoVisionTransformer | ModelWrapper | Any
    ) -> bool:
        if isinstance(model, ModelWrapper):
            return isinstance(model.get_model(), DinoVisionTransformer)
        return isinstance(model, DinoVisionTransformer)

    @classmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> DinoVisionTransformer:
        """
        Get a DINOv3 ViT model by name. Here the student version is build.
        """
        model = vits.vit_small(**model_args)

        checkpoint_dir = cache.get_model_cache_dir()
        model = load_weights(
            model=model,
            checkpoint_dir=checkpoint_dir,
        )

        return model

    @classmethod
    def get_model_wrapper(cls, model: DinoVisionTransformer) -> DINOv3ViTModelWrapper:
        return DINOv3ViTModelWrapper(model=model)

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
                f"DINOv3ViTPackage cannot export model of type {type(model)}. "
                "The model must be a ModelWrapper or a DinoVisionTransformer."
            )

        torch.save(model.state_dict(), out)

        if log_example:
            log_message_code = [
                "from lightly_train._models.dinov3_vit.dinov3_vit_package import DINOv3ViTPackage",
                "import torch",
                "",
                "# Load the pretrained model",
                "model = DINOv3ViTPackage.get_model('dinov3/<vitXX>') # Replace with the model name used in train",
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
DINOV3_VIT_PACKAGE = DINOv3ViTPackage()
