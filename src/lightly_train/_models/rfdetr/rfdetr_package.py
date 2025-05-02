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
from torch.nn import Module

from lightly_train._models import package_helpers
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package
from lightly_train._models.rfdetr.rfdetr import RFDETRModelWrapper

logger = logging.getLogger(__name__)


class RFDETRPackage(Package):
    name = "rfdetr"

    @classmethod
    def list_model_names(cls) -> list[str]:
        try:
            from rfdetr.main import HOSTED_MODELS
        except ImportError:
            return []
        # We use the model names from the checkpoint .pth filenames Roboflow provided
        return [
            f"{cls.name}/{model_name.split('.')[0]}"
            for model_name in HOSTED_MODELS.keys()
        ]

    @classmethod
    def is_supported_model(cls, model: Module) -> bool:
        try:
            from rfdetr.models.lwdetr import LWDETR
        except ImportError:
            return False
        return isinstance(model, LWDETR)

    @classmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> Module:
        try:
            from rfdetr import RFDETRBase, RFDETRLarge
            from rfdetr.detr import RFDETR
            from rfdetr.main import HOSTED_MODELS
            from rfdetr.models.lwdetr import LWDETR
        except ImportError:
            raise ValueError(
                f"Cannot create model '{model_name}' because rfdetr is not installed."
            )

        args = {} if model_args is None else model_args.copy()
        # Remove these arguments so that get_model() only returns the full model
        args.pop("encoder_only", None)
        args.pop("backbone_only", None)

        model_names = [model_name.split(".")[0] for model_name in HOSTED_MODELS.keys()]
        if model_name not in model_names:
            raise ValueError(
                f"Model name '{model_name}' is not supported. "
                f"Supported model names are: {model_names}"
            )
        if "base" in model_name:
            # Type ignore as typing **args correctly is too complex
            model_rfdetr: RFDETR = RFDETRBase(**args)  # type: ignore[arg-type, no-untyped-call]
        elif "large" in model_name:
            # Type ignore as typing **args correctly is too complex
            model_rfdetr = RFDETRLarge(**args)  # type: ignore[arg-type, no-untyped-call]
        else:
            raise ValueError(
                f"Model name '{model_name}' is not supported. "
                f"Supported model names are: {cls.list_model_names()}"
            )

        model_full = model_rfdetr.model.model  # The actual LWDETR model, which is a submodule of nn.Module, is stored in RFDETR().model.model
        if isinstance(model_full, LWDETR):
            return model_full  # type: ignore
        else:
            raise ValueError(f"Model must be of type 'LWDETR', got {type(model_full)}")

    @classmethod
    def get_model_wrapper(cls, model: Module) -> ModelWrapper:
        return RFDETRModelWrapper(model)

    @classmethod
    def export_model(cls, model: Module, out: Path) -> None:
        try:
            from rfdetr.models.lwdetr import LWDETR
        except ImportError:
            raise ValueError(
                f"Cannot create model because '{cls.name}' is not installed."
            )
        if not isinstance(model, LWDETR):
            raise ValueError(f"Model must be of type 'LWDETR', got {type(model)}")

        torch.save({"model": model.state_dict()}, out)

        log_message_code = [
            "from rfdetr import RFDETRBase, RFDETRLarge # based on the model you used",
            "",
            "# Load the pretrained model",
            f"model = RFDETRBase(pretrain_weights={out})",
            "",
            "# Finetune or evaluate the model",
            "...",
        ]
        logger.info(
            package_helpers.format_log_msg_model_usage_example(log_message_code)
        )


# Create singleton instance of the package. The singleton should be used whenever
# possible.
RFDETR_PACKAGE = RFDETRPackage()
