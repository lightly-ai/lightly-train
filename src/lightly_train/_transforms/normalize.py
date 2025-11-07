#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from albumentations import Normalize
from lightning_utilities.core.imports import RequirementCache

from lightly_train.types import NDArrayImage

# Albucore was introduced in albumentations v1.4.15
if RequirementCache("albumentations>=1.4.15"):
    from albucore import (  # type: ignore[import-not-found, import-untyped]
        normalize,
        normalize_per_image,
    )
else:
    from albumentations.augmentations.functional import (  # type: ignore[import-untyped, no-redef]
        normalize,
        normalize_per_image,
    )

# New inteface for albumentations.Normalize was introduced in v1.4.4
ALBUMENTATIONS_GEQ_1_4_4 = RequirementCache("albumentations>=1.4.4")


class NormalizeDtypeAware(Normalize):  # type: ignore[misc]
    """A normalization transform that is compatible with float inputs.

    This class fixes issues with albumentations.Normalize when the input image
    is of type float in range [0,1] and max_pixel_value is not set.

    Args:
        mean (tuple[float, float] | float | None): Mean values for standard normalization.
            For "standard" normalization, the default values are ImageNet mean values: (0.485, 0.456, 0.406).
        std (tuple[float, float] | float | None): Standard deviation values for standard normalization.
            For "standard" normalization, the default values are ImageNet standard deviation :(0.229, 0.224, 0.225).
        max_pixel_value (float | None): Maximum possible pixel value, used for scaling in standard normalization.
            Defaults to 255.0.
        normalization (Literal["standard", "image", "image_per_channel", "min_max", "min_max_per_channel"]):
            Specifies the normalization technique to apply. Defaults to "standard".
            - "standard": Applies the formula `(img - mean * max_pixel_value) / (std * max_pixel_value)`.
                The default mean and std are based on ImageNet. You can use mean and std values of (0.5, 0.5, 0.5)
                for inception normalization. And mean values of (0, 0, 0) and std values of (1, 1, 1) for YOLO.
            - "image": Normalizes the whole image based on its global mean and standard deviation.
            - "image_per_channel": Normalizes the image per channel based on each channel's mean and standard deviation.
            - "min_max": Scales the image pixel values to a [0, 1] range based on the global
                minimum and maximum pixel values.
            - "min_max_per_channel": Scales each channel of the image pixel values to a [0, 1]
                range based on the per-channel minimum and maximum pixel values.

        p (float): Probability of applying the transform. Defaults to 1.0.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        mean: tuple[float, ...] | float | None = (0.485, 0.456, 0.406),
        std: tuple[float, ...] | float | None = (0.229, 0.224, 0.225),
        max_pixel_value: float | None = 255.0,
        normalization: Literal[
            "standard",
            "image",
            "image_per_channel",
            "min_max",
            "min_max_per_channel",
        ] = "standard",
        p: float = 1.0,
    ):
        if ALBUMENTATIONS_GEQ_1_4_4:
            super().__init__(
                mean=mean,
                std=std,
                max_pixel_value=max_pixel_value,
                normalization=normalization,
                p=p,
            )
        else:
            super().__init__(
                mean=mean,
                std=std,
                max_pixel_value=max_pixel_value,
                p=p,
            )

    def apply(self, img: NDArrayImage, **params: Any) -> NDArrayImage:
        if ALBUMENTATIONS_GEQ_1_4_4:
            return self._apply_geq_1_4_4(img, **params)
        else:
            return self._apply_lt_1_4_4(img, **params)

    def _apply_geq_1_4_4(self, img: NDArrayImage, **params: Any) -> NDArrayImage:
        if self.normalization == "standard":
            if img.dtype == np.float32:
                # float32 input is assumed to be in [0.0, 1.0]
                self.mean_np = np.array(self.mean, dtype=np.float32)
                self.denominator = np.reciprocal(
                    np.array(self.std, dtype=np.float32),
                )
            else:
                # uint8 input is assumed to be in [0, 255] unless max_pixel_value is explicitly set
                self.mean_np = (
                    np.array(self.mean, dtype=np.float32) * self.max_pixel_value
                )
                self.denominator = np.reciprocal(
                    np.array(self.std, dtype=np.float32) * self.max_pixel_value,
                )

            return normalize(  # type: ignore[no-any-return]
                img,
                self.mean_np,
                self.denominator,
            )
        return normalize_per_image(img, self.normalization)  # type: ignore[no-any-return]

    def _apply_lt_1_4_4(self, img: NDArrayImage, **params: Any) -> NDArrayImage:
        if img.dtype == np.float32:
            # float32 input is assumed to be in [0.0, 1.0]
            self.mean_np = np.array(self.mean, dtype=np.float32)
            self.denominator = np.reciprocal(
                np.array(self.std, dtype=np.float32),
            )
        else:
            # uint8 input is assumed to be in [0, 255] unless max_pixel_value is explicitly set
            self.mean_np = np.array(self.mean, dtype=np.float32) * self.max_pixel_value
            self.denominator = np.reciprocal(
                np.array(self.std, dtype=np.float32) * self.max_pixel_value,
            )

        return normalize(  # type: ignore[no-any-return]
            img,
            self.mean_np,
            self.denominator,
        )
