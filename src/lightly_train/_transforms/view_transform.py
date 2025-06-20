#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import cv2
from albumentations import (
    BasicTransform,
    ColorJitter,
    Compose,
    GaussianBlur,
    HorizontalFlip,
    Normalize,
    RandomResizedCrop,
    Rotate,
    Solarize,
    ToGray,
    VerticalFlip,
)
from albumentations.pytorch.transforms import ToTensorV2
from lightning_utilities.core.imports import RequirementCache

from lightly_train._configs.config import PydanticConfig
from lightly_train._transforms.channel_drop import ChannelDrop
from lightly_train._transforms.transform import (
    ChannelDropArgs,
    ColorJitterArgs,
    GaussianBlurArgs,
    NormalizeArgs,
    RandomFlipArgs,
    RandomResizeArgs,
    RandomResizedCropArgs,
    RandomRotationArgs,
    SolarizeArgs,
)
from lightly_train.types import TransformInput, TransformOutputSingleView

ALBUMENTATIONS_VERSION_2XX = RequirementCache("albumentations>=2.0.0")
ALBUMENTATIONS_VERSION_GREATER_EQUAL_1_4_22 = RequirementCache("albumentations>=1.4.22")


class ViewTransformArgs(PydanticConfig):
    channel_drop: ChannelDropArgs | None
    random_resized_crop: RandomResizedCropArgs  # only its .scale attribute can be None
    random_flip: RandomFlipArgs | None
    random_rotation: RandomRotationArgs | None
    color_jitter: ColorJitterArgs | None
    random_gray_scale: float | None
    gaussian_blur: GaussianBlurArgs | None
    solarize: SolarizeArgs | None
    normalize: NormalizeArgs


def _get_RandomResizedCrop(
    args: RandomResizedCropArgs,
) -> RandomResizedCrop:
    # A lot of though went into the choice of interpolation method here.
    # See details in https://github.com/lightly-ai/lightly-train/pull/284

    assert args.scale is not None
    if ALBUMENTATIONS_VERSION_2XX:
        return RandomResizedCrop(
            size=(args.size[0], args.size[1]),
            scale=args.scale.as_tuple(),
            interpolation=cv2.INTER_AREA,
        )
    return RandomResizedCrop(
        height=args.size[0],
        width=args.size[1],
        scale=args.scale.as_tuple(),
        interpolation=cv2.INTER_AREA,
    )


def _get_Solarize(args: SolarizeArgs) -> Solarize:
    if ALBUMENTATIONS_VERSION_GREATER_EQUAL_1_4_22:
        return Solarize(
            threshold_range=(args.threshold, args.threshold),
            p=args.prob,
        )
    return Solarize(
        # Old albumentations versions require the threshold to be in the range [0, 255]
        # for uint8 images. New versions automatically scale the threshold from [0, 1.0]
        # depending on the image type.
        threshold=args.threshold * 255,
        p=args.prob,
    )


class ViewTransform:
    def __init__(
        self,
        args: ViewTransformArgs,
    ) -> None:
        transform: list[BasicTransform] = []

        if args.channel_drop is not None:
            transform += [
                ChannelDrop(
                    num_channels_keep=args.channel_drop.num_channels_keep,
                    weight_drop=args.channel_drop.weight_drop,
                )
            ]

        # .scale here corresponds to MethodTransformArgs.random_resize and may be None
        # .size here corresponds to MethodTransformArgs.image_size and may not be None
        if args.random_resized_crop.scale is None:
            args.random_resized_crop.scale = RandomResizeArgs(
                min_scale=1.0, max_scale=1.0
            )
        transform += [_get_RandomResizedCrop(args.random_resized_crop)]

        if args.random_flip:
            transform += [
                HorizontalFlip(p=args.random_flip.horizontal_prob),
                VerticalFlip(p=args.random_flip.vertical_prob),
            ]

        if args.random_rotation:
            transform += [
                Rotate(
                    # We chose to use the border mode default of cv2.BORDER_REFLECT_101,
                    # even though it is different from PIL, cause it makes the image more
                    # realistic.
                    # We also chose to switch the interpolation method to cv2.INTER_AREA,
                    # from NEAREST for PIL, because it is more realistic.
                    # See details in https://linear.app/lightly/issue/LIG-5911/look-into-albumentations-rotation-difference
                    limit=args.random_rotation.degrees,
                    p=args.random_rotation.prob,
                    interpolation=cv2.INTER_AREA,
                    border_mode=cv2.BORDER_REFLECT_101,
                )
            ]

        if args.color_jitter:
            transform += [
                ColorJitter(
                    brightness=args.color_jitter.strength
                    * args.color_jitter.brightness,
                    contrast=args.color_jitter.strength * args.color_jitter.contrast,
                    saturation=args.color_jitter.strength
                    * args.color_jitter.saturation,
                    hue=args.color_jitter.strength * args.color_jitter.hue,
                    p=args.color_jitter.prob,
                )
            ]

        if args.random_gray_scale:
            transform += [ToGray(p=args.random_gray_scale)]

        if args.gaussian_blur:
            transform += [
                GaussianBlur(
                    # Setting blur_limit=0 is necessary for older versions of albumentations
                    # See details in https://linear.app/lightly/issue/LIG-5871/look-into-albumentations-gaussian-blur-difference
                    blur_limit=args.gaussian_blur.blur_limit,
                    sigma_limit=args.gaussian_blur.sigmas,
                    p=args.gaussian_blur.prob,
                )
            ]

        if args.solarize:
            transform += [_get_Solarize(args.solarize)]

        transform += [Normalize(mean=args.normalize.mean, std=args.normalize.std)]

        transform += [ToTensorV2()]

        self.transform = Compose(list(transform))

    def __call__(self, input: TransformInput) -> TransformOutputSingleView:
        transformed: TransformOutputSingleView = self.transform(**input)
        return transformed
