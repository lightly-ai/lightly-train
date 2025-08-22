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

from lightly_train._models import log_usage_example
from lightly_train._models.dinov3.dinov3_src.hub import backbones
from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package

logger = logging.getLogger(__name__)

MODEL_NAME_TO_GETTER = {
    "vits16": backbones.dinov3_vits16,
    "vits16plus": backbones.dinov3_vits16plus,
    "vitb16": backbones.dinov3_vitb16,
    "vitl16": backbones.dinov3_vitl16,
    "vitl16plus": backbones.dinov3_vitl16plus,
    "vith16plus": backbones.dinov3_vith16plus,
    "vit7b16": backbones.dinov3_vit7b16,
}


class DINOv3Package(Package):
    name = "dinov3"

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [
            f"{cls.name}/{model_name}" for model_name in MODEL_NAME_TO_GETTER.keys()
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
        # model_info = VIT_MODELS.get(model_name)
        # if model_info is None:
        #     raise ValueError(
        #         f"Unknown model: {model_name} available models are: {cls.list_model_names()}"
        #     )
        # # Map to original model name if current name is an alias.
        # model_name = model_info.get("alias_for", model_name)
        return model_name

    @classmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> DinoVisionTransformer:
        """
        Get a DINOv3 ViT model by name. Here the student version is build.
        """
        model_args = {} if model_args is None else model_args
        model = MODEL_NAME_TO_GETTER[model_name](**model_args)
        assert isinstance(model, DinoVisionTransformer)
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
                f"DINOv3Package cannot export model of type {type(model)}. "
                "The model must be a ModelWrapper or a DinoVisionTransformer."
            )

        torch.save(model.state_dict(), out)

        if log_example:
            log_message_code = [
                "from lightly_train._models.dinov3.dinov3_package import DINOv3Package",
                "import torch",
                "",
                "# Load the pretrained model",
                "model = DINOv3Package.get_model('dinov3/<vitXX>') # Replace with the model name used in train",
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
DINOV3_PACKAGE = DINOv3Package()
