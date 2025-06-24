#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pydantic import Field

from lightly_train._configs.config import PydanticConfig
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
from lightly_train.types import TransformInput, TransformOutput


class DetConSColorJitterArgs(ColorJitterArgs):
    prob: float = 0.8
    strength: float = 1.0
    brightness: float = 0.8
    contrast: float = 0.8
    saturation: float = 0.8
    hue: float = 0.2


class DetConSGaussianBlurArgs(GaussianBlurArgs):
    prob: float = 0.5
    sigmas: tuple[float, float] = Field(default=(0.1, 2.0), strict=False)
    blur_limit: int | tuple[int, int] = (23, 23)


class DetConSView1GaussianBlurArgs(DetConSGaussianBlurArgs):
    prob: float = 0.0


class DetConSView1TransformArgs(PydanticConfig):
    gaussian_blur: DetConSView1GaussianBlurArgs | None = Field(
        default_factory=DetConSView1GaussianBlurArgs
    )


class DetConSTransformArgs(MethodTransformArgs):
    image_size: tuple[int, int] = Field(default=(224, 224), strict=False)
    channel_drop: ChannelDropArgs | None = None
    random_resize: RandomResizeArgs | None = Field(default_factory=RandomResizeArgs)
    random_flip: RandomFlipArgs | None = Field(default_factory=RandomFlipArgs)
    random_rotation: RandomRotationArgs | None = None
    color_jitter: DetConSColorJitterArgs | None = Field(
        default_factory=DetConSColorJitterArgs
    )
    random_gray_scale: float | None = 0.2
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    gaussian_blur: DetConSGaussianBlurArgs | None = Field(
        default_factory=DetConSGaussianBlurArgs
    )
    solarize: SolarizeArgs | None = None
    view_1: DetConSView1TransformArgs = Field(default_factory=DetConSView1TransformArgs)


class DetConBColorJitterArgs(ColorJitterArgs):
    prob: float = 0.8
    strength: float = 1.0
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.2
    hue: float = 0.1


class DetConBGaussianBlurArgs(GaussianBlurArgs):
    prob: float = 1.0
    sigmas: tuple[float, float] = Field(default=(0.1, 2.0), strict=False)
    blur_limit: int | tuple[int, int] = (23, 23)


class DetConBView1GaussianBlurArgs(DetConBGaussianBlurArgs):
    prob: float = 0.1


class DetConBView1SolarizeArgs(SolarizeArgs):
    prob: float = 0.2
    threshold: float = 0.5


class DetConBView1TransformArgs(PydanticConfig):
    gaussian_blur: DetConBView1GaussianBlurArgs | None = Field(
        default_factory=DetConBView1GaussianBlurArgs
    )
    solarize: DetConBView1SolarizeArgs | None = Field(
        default_factory=DetConBView1SolarizeArgs
    )


class DetConBTransformArgs(MethodTransformArgs):
    image_size: tuple[int, int] = Field(default=(224, 224), strict=False)
    channel_drop: ChannelDropArgs | None = None
    random_resize: RandomResizeArgs | None = Field(default_factory=RandomResizeArgs)
    random_flip: RandomFlipArgs | None = Field(default_factory=RandomFlipArgs)
    random_rotation: RandomRotationArgs | None = None
    color_jitter: DetConBColorJitterArgs | None = Field(
        default_factory=DetConBColorJitterArgs
    )
    random_gray_scale: float | None = 0.2
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    gaussian_blur: DetConBGaussianBlurArgs | None = Field(
        default_factory=DetConBGaussianBlurArgs
    )
    solarize: SolarizeArgs | None = None
    view_1: DetConBView1TransformArgs = Field(default_factory=DetConBView1TransformArgs)


class DetConSTransform(MethodTransform):
    # TODO - Sublass MethodTransform
    def __init__(self, transform_args: DetConSTransformArgs) -> None:
        super().__init__(transform_args=transform_args)
        # blur_limit does not specify x, y sizes of the kernel, but instead defines max
        # and min from where to sample the (quadratic) size of the kernel
        view_transform_0 = ViewTransform(
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

        view_transform_1 = ViewTransform(
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
                gaussian_blur=transform_args.view_1.gaussian_blur,
                solarize=transform_args.solarize,
                normalize=transform_args.normalize,
            )
        )
        self.transforms = [view_transform_0, view_transform_1]

    def __call__(self, input: TransformInput) -> TransformOutput:
        return [transform(input) for transform in self.transforms]

    @staticmethod
    def transform_args_cls() -> type[DetConSTransformArgs]:
        return DetConSTransformArgs


class DetConBTransform(MethodTransform):
    # TODO - Sublass MethodTransform
    def __init__(self, transform_args: DetConBTransformArgs) -> None:
        super().__init__(transform_args=transform_args)

        view_transform_0 = ViewTransform(
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

        view_transform_1 = ViewTransform(
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
                gaussian_blur=transform_args.view_1.gaussian_blur,
                solarize=transform_args.view_1.solarize,
                normalize=transform_args.normalize,
            )
        )

        self.transforms = [view_transform_0, view_transform_1]

    def __call__(self, input: TransformInput) -> TransformOutput:
        return [transform(input) for transform in self.transforms]

    @staticmethod
    def transform_args_cls() -> type[DetConBTransformArgs]:
        return DetConBTransformArgs
