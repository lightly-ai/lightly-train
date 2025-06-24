#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Note: This file is identical (up to renaming) to src/lightly_train/_methods/distillation/distillation_transform.py
from __future__ import annotations

from pydantic import Field

from lightly_train._transforms.transform import (
    ChannelDropArgs,
    ColorJitterArgs,
    GaussianBlurArgs,
    MethodTransform,
    MethodTransformArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomResizeArgs,
    RandomResizedCropArgs,
    RandomRotationArgs,
    SolarizeArgs,
)
from lightly_train._transforms.view_transform import (
    ViewTransform,
    ViewTransformArgs,
)
from lightly_train.types import (
    TransformInput,
    TransformOutput,
)


class DistillationV2RandomResizeArgs(RandomResizeArgs):
    min_scale: float = 0.14


class DistillationV2ColorJitterArgs(ColorJitterArgs):
    prob: float = 0.8
    strength: float = 0.5
    brightness: float = 0.8
    contrast: float = 0.8
    saturation: float = 0.4
    hue: float = 0.2


class DistillationV2GaussianBlurArgs(GaussianBlurArgs):
    prob: float = 1.0
    sigmas: tuple[float, float] = Field(default=(0.0, 0.1), strict=False)
    blur_limit: int | tuple[int, int] = 0


class DistillationV2TransformArgs(MethodTransformArgs):
    image_size: tuple[int, int] = Field(default=(224, 224), strict=False)
    channel_drop: ChannelDropArgs | None = None
    random_resize: DistillationV2RandomResizeArgs | None = Field(
        default_factory=DistillationV2RandomResizeArgs
    )
    random_flip: RandomFlipArgs | None = Field(default_factory=RandomFlipArgs)
    random_rotation: RandomRotationArgs | None = None
    color_jitter: DistillationV2ColorJitterArgs | None = Field(
        default_factory=DistillationV2ColorJitterArgs
    )
    random_gray_scale: float | None = 0.2
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    gaussian_blur: DistillationV2GaussianBlurArgs | None = Field(
        default_factory=DistillationV2GaussianBlurArgs
    )
    solarize: SolarizeArgs | None = None


class DistillationV2Transform(MethodTransform):
    def __init__(self, transform_args: DistillationV2TransformArgs):
        super().__init__(transform_args=transform_args)

        self.transform = ViewTransform(
            ViewTransformArgs(
                channel_drop=transform_args.channel_drop,
                random_resized_crop=RandomResizedCropArgs(
                    size=transform_args.image_size,
                    scale=transform_args.random_resize,
                ),
                random_flip=transform_args.random_flip,
                random_rotation=transform_args.random_rotation,
                color_jitter=transform_args.color_jitter,
                random_gray_scale=transform_args.random_gray_scale,
                gaussian_blur=transform_args.gaussian_blur,
                solarize=transform_args.solarize,
                normalize=transform_args.normalize,
            )
        )

    def __call__(self, input: TransformInput) -> TransformOutput:
        return [self.transform(input)]

    @staticmethod
    def transform_args_cls() -> type[DistillationV2TransformArgs]:
        return DistillationV2TransformArgs
