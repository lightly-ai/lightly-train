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
from lightly_train._models.super_gradients.customizable_detector import (
    CustomizableDetectorFeatureExtractor,
)
from lightly_train._models.super_gradients.segmentation_module import (
    SegmentationModuleFeatureExtractor,
)
from lightly_train.errors import UnknownModelError

logger = logging.getLogger(__name__)


class SuperGradientsPackage(Package):
    name = "super_gradients"

    # Sadly SuperGradients doesn't expose a common interface for all models. We have to
    # define different feature extractors depending on the model types.
    _FEATURE_EXTRACTORS = [
        CustomizableDetectorFeatureExtractor,
        SegmentationModuleFeatureExtractor,
    ]

    @classmethod
    def list_model_names(cls) -> list[str]:
        try:
            from super_gradients.training import models
        except ImportError:
            return []
        model_names = {
            f"{cls.name}/{model_name}"
            for model_name, model_cls in models.ARCHITECTURES.items()
            if cls.is_supported_model_cls(model_cls=model_cls)
        }
        return sorted(model_names)

    @classmethod
    def is_supported_model(cls, model: Module) -> bool:
        return cls.is_supported_model_cls(model_cls=type(model))

    @classmethod
    def is_supported_model_cls(cls, model_cls: type[Module]) -> bool:
        return any(
            fe for fe in cls._FEATURE_EXTRACTORS if fe.is_supported_model_cls(model_cls)
        )

    @classmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> Module:
        try:
            from super_gradients.training import models
        except ImportError:
            raise ValueError(
                f"Cannot create model '{model_name}' because '{cls.name}' is not "
                "installed."
            )
        args = dict(num_classes=10)
        if model_args is not None:
            args.update(model_args)
        model: Module = models.get(model_name=model_name, **args)
        return model

    @classmethod
    def get_feature_extractor(cls, model: Module) -> FeatureExtractor:
        for fe in cls._FEATURE_EXTRACTORS:
            if fe.is_supported_model_cls(model_cls=type(model)):
                return fe(model)
        raise UnknownModelError(f"Unknown {cls.name} model: '{type(model)}'")

    @classmethod
    def export_model(cls, model: Module, out: Path) -> None:
        torch.save(model.state_dict(), out)

        model_name = getattr(model, "_sg_model_name", None)
        num_classes = getattr(model, "num_classes", None)
        if not model_name:
            logger.warning(
                "Usage example can not be constructed since the model name is unknown."
            )
            # TODO this should not happen! We should always have a model name.
            return

        log_message_code = [
            "from super_gradients.training import models",
            "",
            "# Load the pretrained model",
            "model = models.get(",
            f"    model_name='{model_name}',",
            f"    checkpoint_path='{out}',",
            f"    num_classes={num_classes}," if num_classes is not None else "None",
            "    <custom_args>,",
            ")",
            "",
            "# Finetune or evaluate the model",
            "...",
        ]

        # Filter out None values
        log_message_code = [line for line in log_message_code if line != "None"]

        logger.info(
            package_helpers.format_log_msg_model_usage_example(log_message_code)
        )


# Create singleton instance of the package. The singleton should be used whenever
# possible.
SUPER_GRADIENTS_PACKAGE = SuperGradientsPackage()
