#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module

from lightly_train._models import log_usage_example
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package
from lightly_train._models.timm.timm import TIMMModelWrapper

logger = logging.getLogger(__name__)


class TIMMPackage(Package):
    name = "timm"

    @classmethod
    def list_model_names(cls) -> list[str]:
        try:
            import timm
        except ImportError:
            return []
        return [f"{cls.name}/{model_name}" for model_name in timm.list_models()]

    @classmethod
    def is_supported_model(cls, model: Module | ModelWrapper | Any) -> bool:
        # Get the class hierarchy (MRO: Method Resolution Order) and check if
        # any of the (super)classes are from the timm package.
        if isinstance(model, ModelWrapper):
            model = model.get_model()
        class_hierarchy = inspect.getmro(model.__class__)
        return any(_cls.__module__.startswith(cls.name) for _cls in class_hierarchy)

    @classmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> Module:
        try:
            import timm
        except ImportError:
            raise ValueError(
                f"Cannot create model '{model_name}' because timm is not installed."
            )
        args = dict(pretrained=False)
        # vit and eva models have dynamic_img_size defaulting to False, which would not allow inputs with varying image sizes, e.g., for DINO
        if (
            model_name.startswith("vit")
            or model_name.startswith("eva")
            or model_name.startswith("deit")
        ):
            args.update({"dynamic_img_size": True})
        if model_args is not None:
            args.update(model_args)

        # Type ignore as typing **args correctly is too complex
        model: Module = timm.create_model(model_name, **args)  # type: ignore[arg-type]
        return model

    @classmethod
    def get_model_wrapper(cls, model: Module) -> TIMMModelWrapper:
        return TIMMModelWrapper(model)

    @classmethod
    def export_model(
        cls, model: Module | ModelWrapper | Any, out: Path, log_example: bool = True
    ) -> None:
        if isinstance(model, ModelWrapper):
            model = model.get_model()

        if not cls.is_supported_model(model):
            raise ValueError(
                f"TIMMPackage only supports exporting models of type 'Module' and "
                f"'ModelWrapper', but received '{type(model)}'."
            )

        torch.save(model.state_dict(), out)

        if log_example:
            model_name = model.pretrained_cfg.get("architecture", None)
            if not model_name:
                logger.warning(
                    "Usage example can not be constructed since the model name is unknown."
                )
                return

            log_message_code = [
                "import timm",
                "",
                "# Load the pretrained model",
                "model = timm.create_model(",
                f"    model_name='{model_name}',",
                f"    checkpoint_path='{out}',",
                ")",
                "",
                "# Finetune or evaluate the model",
                "...",
            ]
            logger.info(
                log_usage_example.format_log_msg_model_usage_example(log_message_code)
            )


# Create singleton instance of the package. The singleton should be used whenever
# possible.
TIMM_PACKAGE = TIMMPackage()
