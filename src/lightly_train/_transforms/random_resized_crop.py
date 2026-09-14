#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import cv2
from albumentations import RandomResizedCrop
from lightning_utilities.core.imports import RequirementCache

from lightly_train._transforms.transform import RandomResizedCropArgs
from lightly_train.types import ImageSizeTuple

ALBUMENTATIONS_VERSION_2XX = RequirementCache("albumentations>=2.0.0")


def get_random_resized_crop(
    size: ImageSizeTuple, args: RandomResizedCropArgs
) -> RandomResizedCrop:
    """Builds the albumentations RandomResizedCrop used by pretraining and fine-tuning.

    Args:
        size:
            The (height, width) the crop is resized to.
        args:
            The scale and aspect ratio ranges the crop is sampled from.

    """
    # A lot of thought went into the choice of interpolation method here.
    # See details in https://github.com/lightly-ai/lightly-train-old/pull/284
    if ALBUMENTATIONS_VERSION_2XX:
        return RandomResizedCrop(
            size=(size[0], size[1]),
            scale=args.scale_as_tuple(),
            ratio=args.ratio_as_tuple(),
            interpolation=cv2.INTER_AREA,
        )
    return RandomResizedCrop(
        height=size[0],
        width=size[1],
        scale=args.scale_as_tuple(),
        ratio=args.ratio_as_tuple(),
        interpolation=cv2.INTER_AREA,
    )
