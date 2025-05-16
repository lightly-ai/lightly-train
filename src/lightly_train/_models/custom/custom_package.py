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

import torch
from torch.nn import Module

from lightly_train._models import package_helpers
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import BasePackage

logger = logging.getLogger(__name__)


class CustomPackage(BasePackage):
    name = "custom"

    @classmethod
    def is_supported_model(cls, model: Module | ModelWrapper) -> bool:
        """Check if the model is supported by this package."""
        return isinstance(model, ModelWrapper)

    @classmethod
    def export_model(cls, model: Module, out: Path, log_example: bool = True) -> None:
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
                package_helpers.format_log_msg_model_usage_example(log_message_code)
            )


# Create singleton instance of the package. The singleton should be used whenever
# possible.
CUSTOM_PACKAGE = CustomPackage()
