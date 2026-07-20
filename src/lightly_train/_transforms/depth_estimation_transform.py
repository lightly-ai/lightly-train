#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
from albumentations import (
    BasicTransform,
    ColorJitter,
    Compose,
    GaussianBlur,
    HorizontalFlip,
    RandomCrop,
    Resize,
    ToGray,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from typing_extensions import NotRequired

from lightly_train._configs.validate import no_auto
from lightly_train._transforms.normalize import NormalizeDtypeAware as Normalize
from lightly_train._transforms.task_transform import (
    TaskCollateFunction,
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
)
from lightly_train._transforms.transform import (
    ColorJitterArgs,
    GaussianBlurArgs,
    NormalizeArgs,
    RandomCropArgs,
    RandomFlipArgs,
)
from lightly_train.types import (
    DepthEstimationBatch,
    DepthEstimationDatasetItem,
    ImageSizeTuple,
    NDArrayDepth,
    NDArrayImage,
)

logger = logging.getLogger(__name__)

# Albumentations targets: depth and sky are passed through geometric ops as float
# "mask" targets so they are not normalized or color-jittered.
_ADDITIONAL_TARGETS = {"depth": "mask", "sky": "mask"}


class DepthEstimationTransformInput(TaskTransformInput):
    image: NDArrayImage
    depth: NotRequired[NDArrayDepth]
    sky: NotRequired[NDArrayDepth]


class DepthEstimationTransformOutput(TaskTransformOutput):
    image: Tensor
    depth: NotRequired[Tensor]
    sky: NotRequired[Tensor]


class DepthEstimationTransformArgs(TaskTransformArgs):
    image_size: ImageSizeTuple | Literal["auto"]
    num_channels: int | Literal["auto"]
    normalize: NormalizeArgs | Literal["auto"]
    random_flip: RandomFlipArgs | None
    random_crop: RandomCropArgs | None
    # Photometric augmentations applied to the image only; depth and sky are mask
    # targets and are never color-jittered, blurred, or grayscaled.
    color_jitter: ColorJitterArgs | None
    gaussian_blur: GaussianBlurArgs | None
    random_gray_scale: float | None

    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        pass

    def resolve_incompatible(self) -> None:
        self.normalize = no_auto(self.normalize)
        assert isinstance(self.normalize, NormalizeArgs)
        num_channels = no_auto(self.num_channels)
        assert isinstance(num_channels, int)

        # Adjust normalization mean and std to match num_channels.
        if len(self.normalize.mean) != num_channels:
            self.normalize.mean = tuple(
                self.normalize.mean[i % len(self.normalize.mean)]
                for i in range(num_channels)
            )
        if len(self.normalize.std) != num_channels:
            self.normalize.std = tuple(
                self.normalize.std[i % len(self.normalize.std)]
                for i in range(num_channels)
            )

        # Color jitter and grayscale only support 3-channel images.
        if self.color_jitter is not None and num_channels != 3:
            self.color_jitter = None
        if self.random_gray_scale is not None and num_channels != 3:
            self.random_gray_scale = None


class DepthEstimationTransform(TaskTransform):
    transform_args_cls: type[DepthEstimationTransformArgs] = (
        DepthEstimationTransformArgs
    )

    def __init__(self, transform_args: DepthEstimationTransformArgs) -> None:
        super().__init__(transform_args=transform_args)

        image_size = no_auto(transform_args.image_size)
        transform: list[BasicTransform] = [
            Resize(height=image_size[0], width=image_size[1])
        ]

        # During training the image is randomly cropped to a fixed size; the same
        # geometric crop is applied to depth and sky via the additional targets.
        if transform_args.random_crop is not None:
            transform += [
                RandomCrop(
                    height=no_auto(transform_args.random_crop.height),
                    width=no_auto(transform_args.random_crop.width),
                    pad_if_needed=transform_args.random_crop.pad_if_needed,
                    pad_position=transform_args.random_crop.pad_position,
                    fill=transform_args.random_crop.fill,
                    p=transform_args.random_crop.prob,
                )
            ]

        if transform_args.random_flip is not None:
            if transform_args.random_flip.horizontal_prob > 0.0:
                transform += [
                    HorizontalFlip(p=transform_args.random_flip.horizontal_prob)
                ]
            if transform_args.random_flip.vertical_prob > 0.0:
                transform += [VerticalFlip(p=transform_args.random_flip.vertical_prob)]

        # Photometric augmentations are applied to the image only. Because depth and sky
        # are registered as mask targets they are left untouched, so the pixel-to-label
        # correspondence is preserved.
        if transform_args.color_jitter is not None:
            transform += [
                ColorJitter(
                    brightness=transform_args.color_jitter.strength
                    * transform_args.color_jitter.brightness,
                    contrast=transform_args.color_jitter.strength
                    * transform_args.color_jitter.contrast,
                    saturation=transform_args.color_jitter.strength
                    * transform_args.color_jitter.saturation,
                    hue=transform_args.color_jitter.strength
                    * transform_args.color_jitter.hue,
                    p=transform_args.color_jitter.prob,
                )
            ]

        if transform_args.random_gray_scale:
            transform += [ToGray(p=transform_args.random_gray_scale)]

        if transform_args.gaussian_blur is not None:
            transform += [
                GaussianBlur(
                    blur_limit=transform_args.gaussian_blur.blur_limit,
                    sigma_limit=transform_args.gaussian_blur.sigmas,
                    p=transform_args.gaussian_blur.prob,
                )
            ]

        # Normalize only applies to the image; depth and sky are mask targets.
        transform += [
            Normalize(
                mean=no_auto(transform_args.normalize).mean,
                std=no_auto(transform_args.normalize).std,
            ),
            ToTensorV2(),
        ]

        self.transform = Compose(transform, additional_targets=_ADDITIONAL_TARGETS)

    def __call__(
        self, input: DepthEstimationTransformInput
    ) -> DepthEstimationTransformOutput:
        transformed = self.transform(
            image=input["image"], depth=input["depth"], sky=input["sky"]
        )
        # ToTensorV2 leaves mask targets as (H, W); add the channel dimension and make
        # sure depth/sky are float tensors for the regression losses.
        depth = transformed["depth"].unsqueeze(0).float()
        sky = transformed["sky"].unsqueeze(0).float()
        return {"image": transformed["image"], "depth": depth, "sky": sky}


class DepthEstimationCollateFunction(TaskCollateFunction):
    def __call__(self, batch: list[DepthEstimationDatasetItem]) -> DepthEstimationBatch:
        images = [item["image"] for item in batch]
        depths = [item["depth"] for item in batch]
        skies = [item["sky"] for item in batch]

        # Train and val both use a fixed image size, so all samples are stackable.
        out: DepthEstimationBatch = {
            "image_path": [item["image_path"] for item in batch],
            "image": torch.stack(images),
            "depth": torch.stack(depths),
            "sky": torch.stack(skies),
        }
        return out
