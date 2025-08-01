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

from lightly_train._models import log_usage_example
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import BasePackage

logger = logging.getLogger(__name__)


class CustomPackage(BasePackage):
    name = "custom"

    @classmethod
    def is_supported_model(cls, model: Module | ModelWrapper | Any) -> bool:
        return isinstance(model, ModelWrapper)

    @classmethod
    def export_model(
        cls, model: Module | ModelWrapper | Any, out: Path, log_example: bool = True
    ) -> None:
        if isinstance(model, ModelWrapper):
            model = model.get_model()
        elif isinstance(model, Module):
            model = model
        else:
            raise ValueError(
                f"CustomPackage only supports exporting ModelWrapper or torch.nn.Module, "
                f"but got {type(model)}"
            )

        torch.save(model.state_dict(), out)
        if log_example:
            model_name = model.__class__.__name__
            log_message_code = [
                f"import {model_name} # Import the model that was used here",
                "import torch",
                "",
                "# Load the pretrained model",
                f"model = {model_name}(...)",
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
CUSTOM_PACKAGE = CustomPackage()
