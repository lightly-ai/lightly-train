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
from lightly_train._models.feature_extractor import FeatureExtractor
from lightly_train._models.package import Package

logger = logging.getLogger(__name__)


class CustomPackage(Package):
    name = "custom"

    @classmethod
    def list_model_names(cls) -> list[str]:
        return []

    @classmethod
    def is_supported_model(cls, model: Module) -> bool:
        return isinstance(model, FeatureExtractor)

    @classmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> Module:
        raise NotImplementedError()

    @classmethod
    def get_feature_extractor(cls, model: Module) -> FeatureExtractor:
        if not isinstance(model, FeatureExtractor):
            raise TypeError(
                "Unsupported model type: Model does not implement FeatureExtractor interface."
            )
        return model

    @classmethod
    def export_model(cls, model: Module, out: Path) -> None:
        torch.save(model.state_dict(), out)

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
