#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from lightly_train.types import (
    TransformInput,
    TransformOutput,
    TransformOutputSingleView,
)


class EmbeddingTransform:
    def __init__(
        self,
        image_size: int | tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ):
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.transform = A.Compose(
            [
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

    def __call__(self, input: TransformInput) -> TransformOutput:
        transformed: TransformOutputSingleView = self.transform(**input)
        return [transformed]
