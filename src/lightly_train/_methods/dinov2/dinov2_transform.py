#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# * Note: This file is almost identical to src/lightly_train/_methods/dino/dino_transform.py
from __future__ import annotations

from pydantic import Field

from lightly_train._configs.config import PydanticConfig
from lightly_train._transforms.transform import (
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


class DINOv2RandomResizeArgs(RandomResizeArgs):
    min_scale: float = 0.32


class DINOv2LocalViewRandomResizeArgs(RandomResizeArgs):
    min_scale: float = 0.05
    max_scale: float = 0.32


class DINOv2ColorJitterArgs(ColorJitterArgs):
    prob: float = 0.8
    strength: float = 0.5
    brightness: float = 0.8
    contrast: float = 0.8
    saturation: float = 0.4
    hue: float = 0.2


class DINOv2GaussianBlurArgs(GaussianBlurArgs):
    prob: float = 1.0
    sigmas: tuple[float, float] = Field(default=(0.1, 2), strict=False)
    blur_limit: int | tuple[int, int] = 0


class DINOv2GlobalView1GaussianBlurArgs(DINOv2GaussianBlurArgs):
    prob: float = 0.1


class DINOv2GlobalView1SolarizeArgs(SolarizeArgs):
    prob: float = 0.2
    threshold: float = 0.5


class DINOv2LocalViewGaussianBlurArgs(DINOv2GaussianBlurArgs):
    prob: float = 0.5


class DINOv2GlobalView1TransformArgs(PydanticConfig):
    gaussian_blur: DINOv2GlobalView1GaussianBlurArgs | None = Field(
        default_factory=DINOv2GlobalView1GaussianBlurArgs
    )
    solarize: DINOv2GlobalView1SolarizeArgs | None = Field(
        default_factory=DINOv2GlobalView1SolarizeArgs
    )


class DINOv2LocalViewTransformArgs(PydanticConfig):
    num_views: int = 8
    # Strict is set to False because OmegaConf does not support parsing tuples from the
    # CLI. Setting strict to False allows Pydantic to convert lists to tuples.
    view_size: tuple[int, int] = Field(default=(96, 96), strict=False)
    random_resize: DINOv2LocalViewRandomResizeArgs | None = Field(
        default_factory=DINOv2LocalViewRandomResizeArgs
    )
    gaussian_blur: DINOv2LocalViewGaussianBlurArgs | None = Field(
        default_factory=DINOv2LocalViewGaussianBlurArgs
    )


class DINOv2TransformArgs(MethodTransformArgs):
    image_size: tuple[int, int] = Field(default=(224, 224), strict=False)
    random_resize: DINOv2RandomResizeArgs | None = Field(
        default_factory=DINOv2RandomResizeArgs
    )
    random_flip: RandomFlipArgs | None = Field(default_factory=RandomFlipArgs)
    random_rotation: RandomRotationArgs | None = None
    color_jitter: DINOv2ColorJitterArgs | None = Field(
        default_factory=DINOv2ColorJitterArgs
    )
    random_gray_scale: float | None = 0.2
    normalize: NormalizeArgs = Field(default_factory=NormalizeArgs)
    gaussian_blur: DINOv2GaussianBlurArgs | None = Field(
        default_factory=DINOv2GaussianBlurArgs
    )
    solarize: SolarizeArgs | None = None
    global_view_1: DINOv2GlobalView1TransformArgs = Field(
        default_factory=DINOv2GlobalView1TransformArgs
    )
    local_view: DINOv2LocalViewTransformArgs | None = Field(
        default_factory=DINOv2LocalViewTransformArgs
    )


class DINOv2Transform(MethodTransform):
    """

    equivalent to the lightly.transforms.dino_transform.py:DINOTransform class
    """

    def __init__(self, transform_args: DINOv2TransformArgs):
        super().__init__(transform_args=transform_args)
        # Default from https://github.com/facebookresearch/dinov2/blob/main/dinov2/configs/ssl_default_config.yaml

        global_transform_0 = ViewTransform(
            ViewTransformArgs(
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
    def transform_args_cls() -> type[DINOv2TransformArgs]:
        return DINOv2TransformArgs
