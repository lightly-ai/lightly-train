#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Callable, TypedDict

import torch

from lightly_train._env import Env
from lightly_train._models import log_usage_example
from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper
from lightly_train._models.dinov3.dinov3_src.hub import backbones
from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt
from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.package import Package

logger = logging.getLogger(__name__)


class _DINOv3ModelInfo(TypedDict):
    builder: Callable[..., DinoVisionTransformer | ConvNeXt]
    default_weights: str | None
    local_path: str | None


MODEL_NAME_TO_INFO: dict[str, _DINOv3ModelInfo] = {
    # Test model for development purposes only.
    "_vittest16": _DINOv3ModelInfo(
        builder=backbones._dinov3_vit_test,
        default_weights=None,
        local_path=None,
    ),
    "_convnexttest": _DINOv3ModelInfo(
        builder=backbones._dinov3_convnext_test,
        default_weights=None,
        local_path=None,
    ),
    # Tiny models
    # Setting pretrained to False is sufficient (weights argument is ignored).
    "vitt16-notpretrained": _DINOv3ModelInfo(
        builder=functools.partial(backbones.dinov3_vitt16, pretrained=False),
        default_weights=None,
        local_path=None,
    ),
    "vitt16plus-notpretrained": _DINOv3ModelInfo(
        builder=functools.partial(backbones.dinov3_vitt16plus, pretrained=False),
        default_weights=None,
        local_path=None,
    ),
    "vitt16": _DINOv3ModelInfo(
        builder=backbones.dinov3_vitt16,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vitt16_distillationv2.pth",
        local_path="dinov3_vitt16_distillationv2.pth",
    ),
    "vitt16plus": _DINOv3ModelInfo(
        builder=backbones.dinov3_vitt16plus,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vitt16plus_distillationv2.pth",
        local_path="dinov3_vitt16plus_distillationv2.pth",
    ),
    "vitt16-distillationv1": _DINOv3ModelInfo(
        builder=backbones.dinov3_vitt16,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vitt16_distillationv1.pth",
        local_path="dinov3_vitt16_distillationv1.pth",
    ),
    "vitt16plus-distillationv1": _DINOv3ModelInfo(
        builder=backbones.dinov3_vitt16plus,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vitt16plus_distillationv1.pth",
        local_path="dinov3_vitt16plus_distillationv1.pth",
    ),
    # LVD-1689M ViT models
    "vits16": _DINOv3ModelInfo(
        builder=backbones.dinov3_vits16,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vits16_lvd1689m.pth",
        local_path="dinov3_vits16_lvd1689m.pth",
    ),
    "vits16plus": _DINOv3ModelInfo(
        builder=backbones.dinov3_vits16plus,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vits16plus_lvd1689m.pth",
        local_path="dinov3_vits16plus_lvd1689m.pth",
    ),
    "vitb16": _DINOv3ModelInfo(
        builder=backbones.dinov3_vitb16,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vitb16_lvd1689m.pth",
        local_path="dinov3_vitb16_lvd1689m.pth",
    ),
    "vitl16": _DINOv3ModelInfo(
        builder=backbones.dinov3_vitl16,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vitl16_lvd1689m.pth",
        local_path="dinov3_vitl16_lvd1689m.pth",
    ),
    "vith16plus": _DINOv3ModelInfo(
        builder=backbones.dinov3_vith16plus,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vith16plus_lvd1689m.pth",
        local_path="dinov3_vith16plus_lvd1689m.pth",
    ),
    "vit7b16": _DINOv3ModelInfo(
        builder=backbones.dinov3_vit7b16,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vit7b16_lvd1689m.pth",
        local_path="dinov3_vit7b16_lvd1689m.pth",
    ),
    # SAT-493M ViT models
    "vitl16-sat493m": _DINOv3ModelInfo(
        builder=functools.partial(backbones.dinov3_vitl16, is_sat493m_weights=True),
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vitl16_sat493m.pth",
        local_path="dinov3_vitl16_sat493m.pth",
    ),
    "vit7b16-sat493m": _DINOv3ModelInfo(
        builder=backbones.dinov3_vit7b16,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_vit7b16_sat493.pth",
        local_path="dinov3_vit7b16_sat493m.pth",
    ),
    # ConvNeXt LVD-1689M models
    "convnext-tiny": _DINOv3ModelInfo(
        builder=backbones.dinov3_convnext_tiny,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_convnext_tiny_lvd1689m.pth",
        local_path="dinov3_convnext_tiny_lvd1689m.pth",
    ),
    "convnext-small": _DINOv3ModelInfo(
        builder=backbones.dinov3_convnext_small,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_convnext_small_lvd1689m.pth",
        local_path="dinov3_convnext_small_lvd1689m.pth",
    ),
    "convnext-base": _DINOv3ModelInfo(
        builder=backbones.dinov3_convnext_base,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_convnext_base_lvd1689m.pth",
        local_path="dinov3_convnext_base_lvd1689m.pth",
    ),
    "convnext-large": _DINOv3ModelInfo(
        builder=backbones.dinov3_convnext_large,
        default_weights="https://lightly-train-checkpoints.s3.us-east-1.amazonaws.com/dinov3/dinov3_convnext_large_lvd1689m.pth",
        local_path="dinov3_convnext_large_lvd1689m.pth",
    ),
}


class DINOv3Package(Package):
    name = "dinov3"

    @classmethod
    def list_model_names(cls) -> list[str]:
        return [f"{cls.name}/{model_name}" for model_name in MODEL_NAME_TO_INFO.keys()]

    @classmethod
    def is_supported_model(
        cls, model: DinoVisionTransformer | ConvNeXt | ModelWrapper | Any
    ) -> bool:
        if isinstance(model, ModelWrapper):
            return isinstance(model.get_model(), (DinoVisionTransformer, ConvNeXt))
        return isinstance(model, (DinoVisionTransformer, ConvNeXt))

    @classmethod
    def parse_model_name(cls, model_name: str) -> str:
        # Replace "_" with "-" for backwards compatibility.
        # - "vitb14_pretrained" -> "vitb14-pretrained"
        # - "_vittest14_pretrained" -> "_vittest14-pretrained"
        # We keep leading underscores for private test models.
        if model_name:
            model_name = model_name[0] + model_name[1:].replace("_", "-")
        # Replace "-pretrain" with "-pretrained" suffix for backwards compatibility.
        if model_name.endswith("-pretrain"):
            model_name = model_name[: -len("-pretrain")]
        # model_info = VIT_MODELS.get(model_name)
        # if model_info is None:
        #     raise ValueError(
        #         f"Unknown model: {model_name} available models are: {cls.list_model_names()}"
        #     )
        # # Map to original model name if current name is an alias.
        # model_name = model_info.get("alias_for", model_name)
        return model_name

    @classmethod
    def get_model(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> DinoVisionTransformer | ConvNeXt:
        """
        Get a DINOv3 ViT model by name. Here the student version is build.
        """
        args: dict[str, Any] = {"in_chans": num_input_channels}
        if model_args is not None:
            args.update(model_args)
        model_info = MODEL_NAME_TO_INFO[model_name]
        model_builder = model_info["builder"]
        if (
            load_weights
            and ("weights" not in args)
            and model_info["default_weights"] is not None
        ):
            weight_path = _maybe_download_weights(model_getter=model_info)
            args["weights"] = str(weight_path)
        if not load_weights:
            args["weights"] = None
            args["pretrained"] = False
        model = model_builder(**args)
        assert isinstance(model, (DinoVisionTransformer, ConvNeXt))
        return model

    @classmethod
    def get_model_wrapper(
        cls, model: DinoVisionTransformer | ConvNeXt
    ) -> DINOv3ViTModelWrapper | DINOv3VConvNeXtModelWrapper:
        if isinstance(model, DinoVisionTransformer):
            return DINOv3ViTModelWrapper(model=model)
        elif isinstance(model, ConvNeXt):
            return DINOv3VConvNeXtModelWrapper(model=model)
        else:
            raise ValueError(
                f"DINOv3Package cannot create a model wrapper for model of type {type(model)}. "
                "The model must be a DinoVisionTransformer or ConvNeXt."
            )

    @classmethod
    def export_model(
        cls,
        model: DinoVisionTransformer | ConvNeXt | ModelWrapper | Any,
        out: Path,
        log_example: bool = True,
    ) -> None:
        if isinstance(model, ModelWrapper):
            model = model.get_model()

        if not cls.is_supported_model(model):
            raise ValueError(
                f"DINOv3Package cannot export model of type {type(model)}. "
                "The model must be a ModelWrapper or a DinoVisionTransformer."
            )

        torch.save(model.state_dict(), out)

        if log_example:
            log_message_code = [
                "from lightly_train._models.dinov3.dinov3_package import DINOv3Package",
                "import torch",
                "",
                "# Load the pretrained model",
                "model = DINOv3Package.get_model('dinov3/<XYZ>') # Replace with the model name used in train. E.g. 'dinov3/vitb16'",
                f"model.load_state_dict(torch.load('{out}', weights_only=True))",
                "",
                "# Finetune or evaluate the model",
                "...",
            ]
            logger.info(
                log_usage_example.format_log_msg_model_usage_example(log_message_code)
            )


# TODO(Guarin, 10/25): Check hash of downloaded weights.
def _maybe_download_weights(model_getter: _DINOv3ModelInfo) -> Path:
    download_dir: Path = Env.LIGHTLY_TRAIN_MODEL_CACHE_DIR.value.expanduser().resolve()
    url = model_getter["default_weights"]
    assert model_getter["local_path"] is not None
    assert url is not None
    download_dest = download_dir / model_getter["local_path"]
    if not download_dest.exists():
        download_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"DINOv3 weights not found locally. Downloading weights from {url} to "
            f"{download_dest}"
        )
        torch.hub.download_url_to_file(url, dst=str(download_dest))
    return download_dest


# Create singleton instance of the package. The singleton should be used whenever
# possible.
DINOV3_PACKAGE = DINOv3Package()
