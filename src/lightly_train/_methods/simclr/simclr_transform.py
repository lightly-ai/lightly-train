#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
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


class SimCLRColorJitterArgs(ColorJitterArgs):
    prob: float = 0.8
    strength: float = 1.0
    brightness: float = 0.8
    contrast: float = 0.8
    saturation: float = 0.8
    hue: float = 0.2


class SimCLRGaussianBlurArgs(GaussianBlurArgs):
    prob: float = 0.5
    sigmas: tuple[float, float] = Field(default=(0.1, 2), strict=False)
    blur_limit: int | tuple[int, int] = 0


class SimCLRTransformArgs(MethodTransformArgs):
    image_size: tuple[int, int] = Field(default=(224, 224), strict=False)
    channel_drop: ChannelDropArgs | None = None
    random_resize: RandomResizeArgs | None = Field(default_factory=RandomResizeArgs)
    random_flip: RandomFlipArgs | None = Field(default_factory=RandomFlipArgs)
    random_rotation: RandomRotationArgs | None = None
    color_jitter: SimCLRColorJitterArgs | None = Field(
        default_factory=SimCLRColorJitterArgs
    )
    random_gray_scale: float | None = 0.2
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    gaussian_blur: SimCLRGaussianBlurArgs | None = Field(
        default_factory=SimCLRGaussianBlurArgs
    )
    solarize: SolarizeArgs | None = None


class SimCLRTransform(MethodTransform):
    def __init__(self, transform_args: SimCLRTransformArgs):
        super().__init__(transform_args=transform_args)
        # Defaults from https://github.com/lightly-ai/lightly/blob/fac3dcb56745d8e5edcc59307866060cf7530bfa/lightly/transforms/simclr_transform.py#L130-L149
        view_transform = ViewTransform(
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
        self.transforms = [view_transform, view_transform]

    def __call__(self, input: TransformInput) -> TransformOutput:
        return [transform(input) for transform in self.transforms]

    @staticmethod
    def transform_args_cls() -> type[SimCLRTransformArgs]:
        return SimCLRTransformArgs
