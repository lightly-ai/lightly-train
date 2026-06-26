#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from lightly_train._task_models.depth_estimation.task_model import get_model_image_size
from lightly_train._transforms.depth_estimation_transform import (
    DepthEstimationTransform,
    DepthEstimationTransformArgs,
)
from lightly_train._transforms.transform import (
    NormalizeArgs,
    RandomCropArgs,
    RandomFlipArgs,
)
from lightly_train.types import ImageSizeTuple

# All depth backbones are DINOv2 ViTs with a patch size of 14, so the image size must be
# a multiple of 14. Non-divisible sizes are silently truncated to the patch grid in the
# DPT decoder (``H // patch_size``), so we validate against this value up front.
_PATCH_SIZE = 14


class DepthEstimationTrainTransformArgs(DepthEstimationTransformArgs):
    """Default transform arguments for depth estimation training."""

    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    num_channels: int | Literal["auto"] = "auto"
    normalize: NormalizeArgs | Literal["auto"] = "auto"
    random_flip: RandomFlipArgs | None = Field(default_factory=RandomFlipArgs)
    random_crop: RandomCropArgs | None = None

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        _resolve_depth_transform_auto(self, model_init_args=model_init_args)


class DepthEstimationValTransformArgs(DepthEstimationTransformArgs):
    """Default transform arguments for depth estimation validation."""

    image_size: ImageSizeTuple | Literal["auto"] = "auto"
    num_channels: int | Literal["auto"] = "auto"
    normalize: NormalizeArgs | Literal["auto"] = "auto"
    random_flip: RandomFlipArgs | None = None
    random_crop: RandomCropArgs | None = None

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        _resolve_depth_transform_auto(self, model_init_args=model_init_args)


class DepthEstimationTrainTransform(DepthEstimationTransform):
    transform_args_cls = DepthEstimationTrainTransformArgs


class DepthEstimationValTransform(DepthEstimationTransform):
    transform_args_cls = DepthEstimationValTransformArgs


def _resolve_depth_transform_auto(
    args: DepthEstimationTransformArgs, model_init_args: dict[str, Any]
) -> None:
    if args.image_size == "auto":
        model_name = model_init_args.get("model_name")
        if model_name is not None:
            size = get_model_image_size(model_name=model_name)
            args.image_size = (size, size)
        else:
            args.image_size = (504, 504)

    height, width = args.image_size
    if height % _PATCH_SIZE != 0 or width % _PATCH_SIZE != 0:
        raise ValueError(
            f"image_size {args.image_size} (height, width) must be a multiple of the "
            f"patch size {_PATCH_SIZE} in both dimensions. The DepthAnything V3 base "
            f"resolution is (504, 504); other supported sizes from the paper include "
            f"(504, 378), (504, 336), (504, 280), (336, 504), (896, 504), (756, 504), "
            f"and (672, 504)."
        )

    if args.normalize == "auto":
        args.normalize = NormalizeArgs()

    if args.num_channels == "auto":
        args.num_channels = len(args.normalize.mean)
