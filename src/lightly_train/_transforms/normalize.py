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
from albucore import normalize, normalize_per_image  # type: ignore[import-untyped]
from albumentations import (
    Normalize,
)

from lightly_train.types import NDArrayImage


class NormalizeDtypeAware(Normalize):  # type: ignore[misc]
    """A normalization transform that is compatible with float inputs.

    This class fixes issues with albumentations.Normalize when the input image
    is of type float in range [0,1] and max_pixel_value is not set.

    Args:
        mean (Sequence[float]): Mean values for each channel.
        std (Sequence[float]): Standard deviation values for each channel.
        max_pixel_value (float): Maximum pixel value. Default is 255.0.
        always_apply (bool): Whether to always apply this transform. Default is False.
        p (float): Probability of applying this transform. Default is 1.0.
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
        super().__init__()

    def apply(self, img: NDArrayImage, **params: Any) -> NDArrayImage:
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
