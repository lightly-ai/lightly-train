#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, cast

import cv2
import torch
from albumentations import (
    BasicTransform,
    ColorJitter,
    Compose,
    GaussianBlur,
    HorizontalFlip,
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
from lightly_train._transforms.normalize import NormalizeDtypeAware as Normalize
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
    # Record per-view crop/flip geometry and attach it to the output view as a
    # "geometry" tensor, used by dense-relational losses (e.g. dinov31).
    record_geometry: bool = False


def _get_RandomResizedCrop(args: RandomResizedCropArgs) -> RandomResizedCrop:
    # A lot of though went into the choice of interpolation method here.
    # See details in https://github.com/lightly-ai/lightly-train-old/pull/284
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


def build_photometric_ops(
    color_jitter: ColorJitterArgs | None,
    random_gray_scale: float | None,
    gaussian_blur: GaussianBlurArgs | None,
    solarize: SolarizeArgs | None,
) -> list[BasicTransform]:
    """Builds the standard DINO photometric augmentation ops in fixed order.

    Shared by ``ViewTransform`` and the dinov31 PaKA clean/local renders so the
    two stay in sync by construction.
    """
    ops: list[BasicTransform] = []
    if color_jitter:
        ops.append(
            ColorJitter(
                brightness=color_jitter.strength * color_jitter.brightness,
                contrast=color_jitter.strength * color_jitter.contrast,
                saturation=color_jitter.strength * color_jitter.saturation,
                hue=color_jitter.strength * color_jitter.hue,
                p=color_jitter.prob,
            )
        )
    if random_gray_scale:
        ops.append(ToGray(p=random_gray_scale))
    if gaussian_blur:
        ops.append(
            GaussianBlur(
                # Setting blur_limit=0 is necessary for older versions of albumentations.
                # See details in https://linear.app/lightly/issue/LIG-5871/look-into-albumentations-gaussian-blur-difference
                blur_limit=gaussian_blur.blur_limit,
                sigma_limit=gaussian_blur.sigmas,
                p=gaussian_blur.prob,
            )
        )
    if solarize:
        ops.append(_get_Solarize(solarize))
    return ops


class ViewTransform:
    def __init__(
        self,
        args: ViewTransformArgs,
    ):
        self._record_geometry = args.record_geometry
        if self._record_geometry:
            if not ALBUMENTATIONS_VERSION_2XX:
                raise ValueError("record_geometry=True requires albumentations>=2.0.0.")
            if args.random_rotation is not None:
                raise ValueError(
                    "record_geometry=True is not supported together with "
                    "random_rotation because rotation invalidates patch boxes."
                )

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

        transform += build_photometric_ops(
            color_jitter=args.color_jitter,
            random_gray_scale=args.random_gray_scale,
            gaussian_blur=args.gaussian_blur,
            solarize=args.solarize,
        )

        transform += [Normalize(mean=args.normalize.mean, std=args.normalize.std)]

        transform += [ToTensorV2()]

        if self._record_geometry:
            # save_applied_params (albumentations>=2.0 only) records the transforms
            # that fired on each call, which is how we recover the crop box / flips
            # for record_geometry. Read it inside __call__ immediately after the
            # call; this keeps a single ViewTransform instance safe to reuse
            # sequentially across views (as DINOTransform does for local views).
            # Do not share a ViewTransform across threads.
            self.transform = Compose(list(transform), save_applied_params=True)
        else:
            self.transform = Compose(list(transform))

    def __call__(self, input: TransformInput) -> TransformOutputSingleView:
        if not self._record_geometry:
            transformed: TransformOutputSingleView = self.transform(**input)
            return transformed

        image_h, image_w = input["image"].shape[:2]
        transformed = self.transform(**input)
        # Albumentations adds an "applied_transforms" list (not part of the
        # TypedDict) when save_applied_params=True.
        applied = cast(dict[str, Any], transformed).pop("applied_transforms")

        crop_coords: tuple[int, int, int, int] | None = None
        hflip = False
        vflip = False
        # applied_transforms lists only the transforms that fired on this call, so a
        # flip entry means the flip was applied.
        for name, params in applied:
            if name == "RandomResizedCrop":
                crop_coords = params["crop_coords"]
            elif name == "HorizontalFlip":
                hflip = True
            elif name == "VerticalFlip":
                vflip = True
        if crop_coords is None:
            raise RuntimeError(
                "record_geometry=True but no crop was applied. This indicates "
                "an incompatible albumentations version or pipeline."
            )
        transformed["geometry"] = torch.tensor(
            [
                float(crop_coords[0]),
                float(crop_coords[1]),
                float(crop_coords[2]),
                float(crop_coords[3]),
                float(image_w),
                float(image_h),
                1.0 if hflip else 0.0,
                1.0 if vflip else 0.0,
            ],
            dtype=torch.float32,
        )
        return transformed
