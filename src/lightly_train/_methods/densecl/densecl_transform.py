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


class DenseCLRandomResizeArgs(RandomResizeArgs):
    min_scale: float = 0.2
    max_scale: float = 1.0


class DenseCLColorJitterArgs(ColorJitterArgs):
    prob: float = 0.8
    strength: float = 1.0
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.1


class DenseCLGaussianBlurArgs(GaussianBlurArgs):
    prob: float = 0.5
    sigmas: tuple[float, float] = Field(default=(0.1, 2), strict=False)
    blur_limit: int | tuple[int, int] = 0


class DenseCLTransformArgs(MethodTransformArgs):
    image_size: tuple[int, int] = Field(default=(224, 224), strict=False)
    channel_drop: ChannelDropArgs | None = None
    random_resize: DenseCLRandomResizeArgs | None = Field(
        default_factory=DenseCLRandomResizeArgs
    )
    random_flip: RandomFlipArgs | None = Field(default_factory=RandomFlipArgs)
    random_rotation: RandomRotationArgs | None = None
    color_jitter: DenseCLColorJitterArgs | None = Field(
        default_factory=DenseCLColorJitterArgs
    )
    random_gray_scale: float | None = 0.2
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    gaussian_blur: DenseCLGaussianBlurArgs | None = Field(
        default_factory=DenseCLGaussianBlurArgs
    )
    solarize: SolarizeArgs | None = None


class DenseCLTransform(MethodTransform):
    def __init__(self, transform_args: DenseCLTransformArgs):
        super().__init__(transform_args=transform_args)
        # The DenseCLTransform is equal to the MoCoV2Transform.
        # Defaults from https://github.com/lightly-ai/lightly/blob/98756fcffeaef6d3b9a57f311468d8ee755aa26c/lightly/transforms/moco_transform.py#L109-L113
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
    def transform_args_cls() -> type[DenseCLTransformArgs]:
        return DenseCLTransformArgs
