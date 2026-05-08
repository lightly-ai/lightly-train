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

try:
    from rfdetr.detr import RFDETR
except ImportError:

    class RFDETR:  # type: ignore[no-redef]
        pass


from lightly_train._models import log_usage_example
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package
from lightly_train._models.rfdetr.rfdetr import RFDETRModelWrapper

logger = logging.getLogger(__name__)


class RFDETRPackage(Package):
    name = "rfdetr"

    @classmethod
    def _supported_model_names(cls) -> list[str]:
        return [
            "rf-detr-base",
            "rf-detr-base-o365",
            "rf-detr-base-2",
            "rf-detr-large",
            "rf-detr-nano",
            "rf-detr-small",
            "rf-detr-medium",
            "rf-detr-seg-preview",
        ]

    @classmethod
    def list_model_names(cls) -> list[str]:
        try:
            import rfdetr  # noqa: F401
        except ImportError:
            return []
        return [
            f"{cls.name}/{model_name}" for model_name in cls._supported_model_names()
        ]

    @classmethod
    def is_supported_model(cls, model: RFDETR | ModelWrapper | Any) -> bool:
        if isinstance(model, ModelWrapper):
            return isinstance(model.get_model(), RFDETR)
        return isinstance(model, RFDETR)

    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> RFDETR:
        try:
            from rfdetr.assets.model_weights import ModelWeights
            from rfdetr.detr import (
                RFDETRBase,
                RFDETRLarge,
                RFDETRMedium,
                RFDETRNano,
                RFDETRSegPreview,
                RFDETRSmall,
            )
        except ImportError:
            raise ValueError(
                f"Cannot create model '{model_name}' because rfdetr is not installed."
            )
        if num_input_channels != 3:
            raise ValueError(
                f"RFDETR models only support 3 input channels, but got "
                f"{num_input_channels}."
            )

        args = {} if model_args is None else model_args.copy()
        # Remove these arguments so that get_model() only returns the full model
        args.pop("encoder_only", None)
        args.pop("backbone_only", None)
        model_name = Path(model_name).stem
        model_assets = {Path(asset.filename).stem: asset for asset in ModelWeights}
        if model_name not in model_assets:
            raise ValueError(
                f"Model name '{model_name}' is not supported. "
                f"Supported model names are: {cls.list_model_names()}"
            )

        if load_weights:
            args.setdefault("pretrain_weights", model_assets[model_name].filename)
        else:
            args["pretrain_weights"] = None

        model_constructors: dict[str, type[RFDETR]] = {
            "rf-detr-base": RFDETRBase,
            "rf-detr-base-o365": RFDETRBase,
            "rf-detr-base-2": RFDETRBase,
            "rf-detr-large": RFDETRLarge,
            "rf-detr-nano": RFDETRNano,
            "rf-detr-small": RFDETRSmall,
            "rf-detr-medium": RFDETRMedium,
            "rf-detr-seg-preview": RFDETRSegPreview,
        }
        model_ctor = model_constructors.get(model_name)
        if model_ctor is None:
            raise ValueError(
                f"Model name '{model_name}' is not supported. "
                f"Supported model names are: {cls.list_model_names()}"
            )

        model_rfdetr = model_ctor(**args)  # type: ignore[no-untyped-call]
        return model_rfdetr

    @classmethod
    def get_model_wrapper(cls, model: RFDETR) -> RFDETRModelWrapper:
        return RFDETRModelWrapper(model)

    @classmethod
    def export_model(
        cls,
        model: RFDETR | ModelWrapper | Any,
        out: Path,
        log_example: bool = True,
    ) -> None:
        try:
            from rfdetr.models.backbone.dinov2 import (
                DinoV2,
                WindowedDinov2WithRegistersBackbone,
            )
            from rfdetr.models.lwdetr import LWDETR
        except ImportError:
            raise ValueError(
                f"Cannot create model because '{cls.name}' is not installed."
            )

        if isinstance(model, ModelWrapper):
            model = model.get_model()

        if not cls.is_supported_model(model):
            raise ValueError(
                f"Model must be of type 'RFDETR' or 'RFDETRModelWrapper', got {type(model)}"
            )

        lwdetr_model = model.model.model
        assert isinstance(lwdetr_model, LWDETR)

        assert isinstance(
            lwdetr_model.backbone[0].encoder.encoder,
            WindowedDinov2WithRegistersBackbone,
        ), type(lwdetr_model.backbone[0].encoder)
        assert isinstance(lwdetr_model.backbone[0].encoder, DinoV2)

        torch.save({"model": lwdetr_model.state_dict()}, out)
        if log_example:
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
                log_usage_example.format_log_msg_model_usage_example(log_message_code)
            )


# Create singleton instance of the package. The singleton should be used whenever
# possible.
RFDETR_PACKAGE = RFDETRPackage()
