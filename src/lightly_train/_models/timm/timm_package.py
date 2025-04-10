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

from lightly_train._models import package_helpers
from lightly_train._models.feature_extractor import FeatureExtractor
from lightly_train._models.package import Package
from lightly_train._models.timm.timm import TIMMFeatureExtractor

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
    def is_supported_model(cls, model: Module) -> bool:
        # Get the class hierarchy (MRO: Method Resolution Order) and check if
        # any of the (super)classes are from the timm package.
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
        if model_args is not None:
            args.update(model_args)
        # Type ignore as typing **args correctly is too complex
        model: Module = timm.create_model(model_name, **args)  # type: ignore[arg-type]
        return model

    @classmethod
    def get_feature_extractor(cls, model: Module) -> FeatureExtractor:
        return TIMMFeatureExtractor(model)

    @classmethod
    def export_model(cls, model: Module, out: Path) -> None:
        torch.save(model.state_dict(), out)

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
            package_helpers.format_log_msg_model_usage_example(log_message_code)
        )


# Create singleton instance of the package. The singleton should be used whenever
# possible.
TIMM_PACKAGE = TIMMPackage()
