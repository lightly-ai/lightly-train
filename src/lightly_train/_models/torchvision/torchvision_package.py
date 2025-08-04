#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module
from torchvision import models as torchvision_models

from lightly_train._models import log_usage_example
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package
from lightly_train._models.torchvision.convnext import ConvNeXtModelWrapper
from lightly_train._models.torchvision.resnet import ResNetModelWrapper
from lightly_train._models.torchvision.shufflenet import ShuffleNetV2ModelWrapper
from lightly_train._models.torchvision.torchvision import TorchvisionModelWrapper
from lightly_train.errors import UnknownModelError

logger = logging.getLogger(__name__)


class TorchvisionPackage(Package):
    name = "torchvision"

    _FEATURE_EXTRACTORS = [
        ConvNeXtModelWrapper,
        ResNetModelWrapper,
        ShuffleNetV2ModelWrapper,
    ]

    @classmethod
    def list_model_names(cls) -> list[str]:
        regex_str = "|".join(
            f"({fe._torchvision_model_name_pattern})" for fe in cls._FEATURE_EXTRACTORS
        )
        pattern = re.compile(pattern=regex_str)
        model_names = {
            f"{cls.name}/{model_name}"
            for model_name in torchvision_models.list_models()
            if pattern.match(model_name)
        }
        return sorted(model_names)

    @classmethod
    def is_supported_model(cls, model: Module | ModelWrapper | Any) -> bool:
        if isinstance(model, ModelWrapper):
            model = model.get_model()
        return type(model) in cls._model_cls_to_extractor_cls()

    @classmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> Module:
        args = dict()
        if model_args is not None:
            args.update(model_args)
        model: Module = torchvision_models.get_model(model_name, **args)
        return model

    @classmethod
    def get_model_wrapper(cls, model: Module) -> TorchvisionModelWrapper:
        feature_extractor_cls = cls._model_cls_to_extractor_cls().get(type(model))
        if feature_extractor_cls is not None:
            return feature_extractor_cls(model)
        raise UnknownModelError(f"Unknown torchvision model: '{model}'")

    @classmethod
    def _model_cls_to_extractor_cls(
        cls,
    ) -> dict[type[Module], type[TorchvisionModelWrapper]]:
        module_to_cls = {}
        for feature_extractor_cls in cls._FEATURE_EXTRACTORS:
            for model_cls in feature_extractor_cls._torchvision_models:
                module_to_cls[model_cls] = feature_extractor_cls
        return module_to_cls

    @classmethod
    def export_model(
        cls, model: Module | ModelWrapper | Any, out: Path, log_example: bool = True
    ) -> None:
        if isinstance(model, ModelWrapper):
            model = model.get_model()

        if not cls.is_supported_model(model):
            raise TypeError(
                f"TorchvisionPackage only supports exporting models from torchvision "
                f"but received '{type(model)}'."
            )

        torch.save(model.state_dict(), out)
        if log_example:
            model_name = "<model_name>"
            log_message_code = [
                "from torchvision import models",
                "import torch",
                "",
                "# Load the pretrained model",
                f"model = models.{model_name}()  # Specify the used model here",
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
TORCHVISION_PACKAGE = TorchvisionPackage()
