#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import numpy as np
from albumentations import (
    ChannelShuffle,
    ColorJitter,
    ImageOnlyTransform,
    RandomOrder,
)
from numpy.typing import NDArray


class RandomPhotometricDistort(ImageOnlyTransform):
    def __init__(
        self,
        brightness: tuple[float, float],
        contrast: tuple[float, float],
        saturation: tuple[float, float],
        hue: tuple[float, float],
        p: float = 0.5,
    ):
        """
        Apply random photometric distortions to an image.

        This transform is meant to correspond to the RandomPhotometricDistort from
        the torchvision v2 transforms.

        Args:
            brightness:
                Tuple (min, max) from which to uniformly sample brightness adjustment
                factor. Should be non-negative.
            contrast:
                Tuple (min, max) from which to uniformly sample contrast adjustment
                factor. Should be non-negative.
            saturation:
                Tuple (min, max) from which to uniformly sample saturation adjustment
                factor. Should be non-negative.
            hue:
                Tuple (min, max) from which to uniformly sample hue adjustment factor
                in degrees. Should respect -0.5 <= min <= max <= 0.5.
            prob:
                Probability of applying the transform. Should be in [0, 1].
        """
        super().__init__(p=1.0)  # We handle probability in the sub-transforms.
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

        if any(b < 0 for b in brightness):
            raise ValueError(
                f"Brightness values must be non-negative, got {brightness}."
            )
        if any(c < 0 for c in contrast):
            raise ValueError(f"Contrast values must be non-negative, got {contrast}.")

        if any(s < 0 for s in saturation):
            raise ValueError(
                f"Saturation values must be non-negative, got {saturation}."
            )

        if any(-0.5 > h or h > 0.5 for h in hue) or hue[0] > hue[1]:
            raise ValueError(
                f"Hue values must respect -0.5 <= min <= max <= 0.5, got {hue}."
            )

        self.transform = RandomOrder(
            [
                ColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                    p=p,
                ),
                ChannelShuffle(p=p),
            ]
        )

    def apply(
        self, img: NDArray[np.uint8], **params: dict[str, Any]
    ) -> NDArray[np.uint8]:
        """Apply the random photometric distort transform to the image.

        Args:
            img: Input image as numpy array with shape (H, W, C).

        Returns:
            Transformed image as numpy array with shape (H, W, C).
        """
        return self.transform(image=img)["image"]
