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
from typing import Any, Callable, cast

import torch
from torch.nn import Module

from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.package import Package
from lightly_train._modules.teachers.dinov2.configs import MODELS as VIT_MODELS
from lightly_train._modules.teachers.dinov2.configs import (
    get_config_path,
    load_and_merge_config,
)
from lightly_train._modules.teachers.dinov2.models import vision_transformer as vits
from lightly_train._modules.teachers.dinov2.models.vision_transformer import (
    DinoVisionTransformer,
)

logger = logging.getLogger(__name__)


class DINOv2ViTPackage(Package):
    name = "DINOv2VIT"

    @classmethod
    def list_model_names(cls) -> list[str]:
        return list(VIT_MODELS.keys())
    
    @classmethod
    def is_supported_model(cls, model: Module) -> bool:
        return isinstance(model, DinoVisionTransformer)

    @classmethod
    def get_model(
        cls, model_name: str, model_args: dict[str, Any] | None = None
    ) -> DinoVisionTransformer:
        if model_name not in VIT_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_info = VIT_MODELS[model_name]
        #url = model_info["url"] # TODO do we want to allow pretrained loading?
        config_name = model_info["config"]

        # Load config.
        config_path = get_config_path(config_name)
        cfg = load_and_merge_config(str(config_path))

        suffix = "_memeff"
        if cfg.student.arch.endswith(suffix):
            cfg.student.arch = cfg.student.arch[: -len(suffix)]

        model_builders = {  
            "vit_small": vits.vit_small,
            "vit_base": vits.vit_base,
            "vit_large": vits.vit_large,
            "vit_giant2": vits.vit_giant2,
        }
        model_builder = model_builders.get(cfg.student.arch, None)
        if model_builder is None:
            raise TypeError(
                f"Unsupported architecture type {cfg.student.arch}."
            )
        
        # Cast the model builder to the correct type
        model_builder = cast(
            Callable[...,DinoVisionTransformer],
            model_builder
        )
        
        model = model_builder(
            mg_size=cfg.crops.global_crops_size,
            patch_size=cfg.student.patch_size,
            init_values=cfg.student.layerscale,
            ffn_layer=cfg.student.ffn_layer,
            block_chunks=cfg.student.block_chunks,
            qkv_bias=cfg.student.qkv_bias,
            proj_bias=cfg.student.proj_bias,
            ffn_bias=cfg.student.ffn_bias,
            num_register_tokens=cfg.student.num_register_tokens,
            interpolate_offset=cfg.student.interpolate_offset,
            interpolate_antialias=cfg.student.interpolate_antialias,
            # Student only--------------------------------------------------------------
            drop_path_rate=cfg.student.drop_path_rate, 
            drop_path_uniform=cfg.student.drop_path_uniform,
        )
        return model
    

    @classmethod
    def get_model_wrapper(cls, model: Module) -> DINOv2ViTModelWrapper:
        if not isinstance(model, DINOv2ViTModelWrapper):
            raise TypeError(
                "Unsupported model type: Model does not implement FeatureExtractor interface."
            )
        return model

    @classmethod
    def export_model(cls, model: Module, out: Path, log_example: bool = True) -> None:
        torch.save(model.state_dict(), out)
        #TODO: finish implement export_model

# Create singleton instance of the package. The singleton should be used whenever
# possible.
DINOV2_VIT_Package_PACKAGE = DINOv2ViTPackage()
