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
from lightly_train.types import (
    TransformInput,
    TransformOutput,
)


class DINORandomResizeArgs(RandomResizeArgs):
    min_scale: float = 0.14


class DINOLocalViewRandomResizeArgs(RandomResizeArgs):
    min_scale: float = 0.05
    max_scale: float = 0.14


class DINOColorJitterArgs(ColorJitterArgs):
    prob: float = 0.8
    strength: float = 0.5
    brightness: float = 0.8
    contrast: float = 0.8
    saturation: float = 0.4
    hue: float = 0.2


class DINOGaussianBlurArgs(GaussianBlurArgs):
    prob: float = 1.0
    sigmas: tuple[float, float] = Field(default=(0.1, 2), strict=False)
    blur_limit: int | tuple[int, int] = 0


class DINOGlobalView1GaussianBlurArgs(DINOGaussianBlurArgs):
    prob: float = 0.1


class DINOGlobalView1SolarizeArgs(SolarizeArgs):
    prob: float = 0.2
    threshold: float = 0.5


class DINOLocalViewGaussianBlurArgs(DINOGaussianBlurArgs):
    prob: float = 0.5


class DINOGlobalView1TransformArgs(PydanticConfig):
    gaussian_blur: DINOGlobalView1GaussianBlurArgs | None = Field(
        default_factory=DINOGlobalView1GaussianBlurArgs
    )
    solarize: DINOGlobalView1SolarizeArgs | None = Field(
        default_factory=DINOGlobalView1SolarizeArgs
    )


class DINOLocalViewTransformArgs(PydanticConfig):
    num_views: int = 6
    # Strict is set to False because OmegaConf does not support parsing tuples from the
    # CLI. Setting strict to False allows Pydantic to convert lists to tuples.
    view_size: tuple[int, int] = Field(default=(96, 96), strict=False)
    random_resize: DINOLocalViewRandomResizeArgs | None = Field(
        default_factory=DINOLocalViewRandomResizeArgs
    )
    gaussian_blur: DINOLocalViewGaussianBlurArgs | None = Field(
        default_factory=DINOLocalViewGaussianBlurArgs
    )


class DINOTransformArgs(MethodTransformArgs):
    # TODO: Authors recommend to use different scales for convnets than
    # transformers. We should add a check for the model type and use the appropriate
    # scales accordingly.
    # https://github.com/facebookresearch/dino#resnet-50-and-other-convnets-trainings
    image_size: tuple[int, int] = Field(default=(224, 224), strict=False)
    channel_drop: ChannelDropArgs | None = None
    random_resize: DINORandomResizeArgs | None = Field(
        default_factory=DINORandomResizeArgs
    )
    random_flip: RandomFlipArgs | None = Field(default_factory=RandomFlipArgs)
    random_rotation: RandomRotationArgs | None = None
    color_jitter: DINOColorJitterArgs | None = Field(
        default_factory=DINOColorJitterArgs
    )
    random_gray_scale: float | None = 0.2
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    gaussian_blur: DINOGaussianBlurArgs | None = Field(
        default_factory=DINOGaussianBlurArgs
    )
    solarize: SolarizeArgs | None = None
    global_view_1: DINOGlobalView1TransformArgs = Field(
        default_factory=DINOGlobalView1TransformArgs
    )
    local_view: DINOLocalViewTransformArgs | None = Field(
        default_factory=DINOLocalViewTransformArgs
    )


class DINOTransform(MethodTransform):
    """

    equivalent to the lightly.transforms.dino_transform.py:DINOTransform class
    """

    def __init__(self, transform_args: DINOTransformArgs):
        super().__init__(transform_args=transform_args)
        # Default from https://github.com/lightly-ai/lightly/blob/fac3dcb56745d8e5edcc59307866060cf7530bfa/lightly/transforms/dino_transform.py#L115

        global_transform_0 = ViewTransform(
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

        global_transform_1 = ViewTransform(
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
                gaussian_blur=transform_args.global_view_1.gaussian_blur,
                solarize=transform_args.global_view_1.solarize,
                normalize=transform_args.normalize,
            )
        )

        transforms = [global_transform_0, global_transform_1]

        # Only add local transforms if local_view is provided
        if transform_args.local_view is not None:
            local_transform = ViewTransform(
                ViewTransformArgs(
                    channel_drop=transform_args.channel_drop,
                    random_resized_crop=RandomResizedCropArgs(
                        size=transform_args.local_view.view_size,
                        scale=transform_args.local_view.random_resize,
                    ),
                    random_flip=transform_args.random_flip,
                    random_rotation=transform_args.random_rotation,
                    color_jitter=transform_args.color_jitter,
                    random_gray_scale=transform_args.random_gray_scale,
                    gaussian_blur=transform_args.local_view.gaussian_blur,
                    solarize=transform_args.solarize,
                    normalize=transform_args.normalize,
                )
            )
            local_transforms = [local_transform] * transform_args.local_view.num_views
            transforms.extend(local_transforms)

        self.transforms = transforms

    def __call__(self, input: TransformInput) -> TransformOutput:
        return [transform(input) for transform in self.transforms]

    @staticmethod
    def transform_args_cls() -> type[DINOTransformArgs]:
        return DINOTransformArgs
