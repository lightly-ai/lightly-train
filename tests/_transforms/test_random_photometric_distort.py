#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
import pytest

from lightly_train._transforms.random_photometric_distort import (
    RandomPhotometricDistort,
)


class TestRandomPhotometricDistort:
    @pytest.mark.parametrize(
        "brightness, contrast, saturation, hue, p, error",
        [
            (
                (-0.1, 1.0),
                (1.0, 2.0),
                (1.0, 2.0),
                (0.0, 0.1),
                0.5,
                "Brightness values must be non-negative",
            ),
            (
                (0.5, 1.0),
                (-1.0, 2.0),
                (1.0, 2.0),
                (0.0, 0.1),
                0.5,
                "Contrast values must be non-negative",
            ),
            (
                (0.5, 1.0),
                (1.0, 2.0),
                (-1.0, 2.0),
                (0.0, 0.1),
                0.5,
                "Saturation values must be non-negative",
            ),
            (
                (0.5, 1.0),
                (1.0, 2.0),
                (1.0, 2.0),
                (-0.6, 0.1),
                0.5,
                "Hue values must respect -0.5 <= min <= max <= 0.5",
            ),
            (
                (0.5, 1.0),
                (1.0, 2.0),
                (1.0, 2.0),
                (0.0, 0.6),
                0.5,
                "Hue values must respect -0.5 <= min <= max <= 0.5",
            ),
            (
                (0.5, 1.0),
                (1.0, 2.0),
                (1.0, 2.0),
                (0.2, 0.1),
                0.5,
                "Hue values must respect -0.5 <= min <= max <= 0.5",
            ),
            (
                (0.5, 1.0),
                (1.0, 2.0),
                (1.0, 2.0),
                (0.0, 0.1),
                -0.1,
                "Input should be greater than or equal to 0",
            ),
            (
                (0.5, 1.0),
                (1.0, 2.0),
                (1.0, 2.0),
                (0.0, 0.1),
                1.1,
                "Input should be less than or equal to 1",
            ),
        ],
    )
    def test__init__errors(
        self,
        brightness,
        contrast,
        saturation,
        hue,
        p,
        error,
    ) -> None:
        with pytest.raises(ValueError, match=error):
            RandomPhotometricDistort(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=p,
            )

    def test__init__valid(self) -> None:
        RandomPhotometricDistort(
            brightness=(0.5, 1.5),
            contrast=(0.5, 1.5),
            saturation=(0.5, 1.5),
            hue=(-0.05, 0.05),
            p=1.0,
        )

    def test__call__returns_image(self) -> None:
        # Create a dummy image (uint8, shape HWC)
        img = np.ones((16, 16, 3), dtype=np.uint8) * 127
        transform = RandomPhotometricDistort(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.1, 0.1),
            p=1.0,
        )
        out = transform(image=img)
        assert isinstance(out, dict)
        assert "image" in out
        result = out["image"]
        assert isinstance(result, np.ndarray)
        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test__call__no_transform_when_p0(self) -> None:
        img = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
        transform = RandomPhotometricDistort(
            brightness=(1.0, 1.0),
            contrast=(1.0, 1.0),
            saturation=(1.0, 1.0),
            hue=(0.0, 0.0),
            p=0.0,
        )
        out = transform(image=img)
        np.testing.assert_array_equal(out["image"], img)

    def test__call__always_transform_when_p1(self) -> None:
        np.random.seed(42)
        img = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
        # Use 0.5, 0.9 to make sure that 1.0 is not sampled and the image must change.
        transform = RandomPhotometricDistort(
            brightness=(0.5, 0.9),
            contrast=(0.5, 0.9),
            saturation=(0.5, 0.9),
            hue=(-0.1, 0.1),
            p=1.0,
        )
        out = transform(image=img)
        assert not np.array_equal(out["image"], img)
