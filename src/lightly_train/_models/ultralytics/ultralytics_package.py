#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module

if TYPE_CHECKING:
    from ultralytics import YOLO

from lightly_train._models import log_usage_example
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package
from lightly_train._models.ultralytics.ultralytics import (
    UltralyticsModelWrapper,
)

logger = logging.getLogger(__name__)


class UltralyticsPackage(Package):
    name = "ultralytics"

    @classmethod
    def list_model_names(cls) -> list[str]:
        try:
            from ultralytics.nn import tasks
            from ultralytics.utils import downloads
        except ImportError:
            return []

        # We currently only support models that have a backbone ending with an SPPF
        # layer. See ultralytics/cfg/models for different architectures.
        supported_models = {"yolov5", "yolov6", "yolov8", "yolo11", "yolo12"}

        # These models are downloaded from the ultralytics repository.
        pretrained_models = {
            f"{model_name}.pt"
            for model_name in downloads.GITHUB_ASSETS_STEMS
            if any(model_name.startswith(model) for model in supported_models)
        }

        # These models are instantiated from their yaml configuration files.
        untrained_models = {
            f"{model_name}{scale}.yaml"
            for model_name in supported_models
            for scale in ["n", "s", "m", "l", "x"]
        }
        untrained_models.update(
            {model_name.replace(".pt", ".yaml") for model_name in pretrained_models}
        )
        # Check that the model configs are available. Ultralytics doesn't provide
        # a list of available models/configs.
        for model_name in list(untrained_models):
            try:
                # This only loads the model config, not the model itself.
                tasks.yaml_model_load(model_name)
            except FileNotFoundError:
                untrained_models.remove(model_name)

        model_names = pretrained_models.union(untrained_models)
        model_names = {f"{cls.name}/{model_name}" for model_name in model_names}
        return sorted(model_names)

    @classmethod
    def is_supported_model(cls, model: Module | ModelWrapper) -> bool:
        try:
            from ultralytics import YOLO
        except ImportError:
            return False
        if isinstance(model, ModelWrapper):
            model = model.get_model()
        return isinstance(model, YOLO)

    @classmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> Module:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ValueError(
                f"Cannot create model '{model_name}' because '{cls.name}' is not "
                "installed."
            )
        args = {} if model_args is None else model_args
        model: Module = YOLO(model=model_name, **args)
        return model

    @classmethod
    def get_model_wrapper(cls, model: Module) -> UltralyticsModelWrapper:
        return UltralyticsModelWrapper(model=model)

    @classmethod
    def export_model(
        cls, model: YOLO | ModelWrapper | Any, out: Path, log_example: bool = True
    ) -> None:
        try:
            import ultralytics
        except ImportError:
            raise ValueError(
                f"Cannot export model because '{cls.name}' is not installed."
            )
        if isinstance(model, ModelWrapper):
            model = model.get_model()

        if not cls.is_supported_model(model):
            raise TypeError(
                f"UltralyticsPackage supports exporting models of type 'YOLO' and "
                f"'ModelWrapper', but received '{type(model)}'."
            )

        export_model = copy.deepcopy(model)
        if export_model.ckpt is None:
            export_model.ckpt = {}

        version = ultralytics.__version__
        if version.startswith("8.1.") or version.startswith("8.2."):
            # Ultralytics 8.1.8 introduced a Model.save() method. But this method only
            # worked for models trained with the Ultralytics trainer. This was fixed in
            # 8.3.0. Before the fix, the model has to be saved as a dict.
            torch.save({"model": export_model.model}, out)
        elif hasattr(export_model, "save"):
            # Ultralytics >=8.3.0
            export_model.save(out)
        else:
            raise RuntimeError(f"Cannot save Ultralytics model with version {version}")

        if log_example:
            log_message_code = [
                "from ultralytics import YOLO",
                "",
                "# Load the pretrained model",
                f"model = YOLO('{out}')",
                "",
                "# Finetune or evaluate the model",
                "...",
            ]
            logger.info(
                log_usage_example.format_log_msg_model_usage_example(log_message_code)
            )


# Create singleton instance of the package. The singleton should be used whenever
# possible.
ULTRALYTICS_PACKAGE = UltralyticsPackage()
