#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from albumentations import (
    BasicTransform,
    Compose,
)
from albumentations.pytorch import ToTensorV2

from lightly_train._transforms.predict_transform import (
    PredictTransform,
    PredictTransformArgs,
)
from lightly_train._transforms.transform import NormalizeArgs
from lightly_train.types import TransformInput, TransformOutput


class PredictSemanticSegmentationTransformArgs(PredictTransformArgs):
    image_size: tuple[int, int]
    normalize: NormalizeArgs


class PredictSemanticSegmentationTransform(PredictTransform):
    transform_args_cls: type[PredictSemanticSegmentationTransformArgs] = (
        PredictSemanticSegmentationTransformArgs
    )

    def __init__(
        self,
        transform_args: PredictSemanticSegmentationTransformArgs,
    ) -> None:
        super().__init__(transform_args=transform_args)

        transform: list[BasicTransform] = [
            # TODO(Yutong, 10/25): enable them once predict_batch is implemented
            # ToFloat(),
            # Normalize(
            #     mean=transform_args.normalize.mean,
            #     std=transform_args.normalize.std,
            #     max_pixel_value=1.0,
            # ),
            # Resize(
            #     height=transform_args.image_size[0],
            #     width=transform_args.image_size[1],
            # ),
            ToTensorV2(),
        ]
        self.transform = Compose(transform)

    def __call__(self, input: TransformInput) -> TransformOutput:
        return [self.transform(**input)]
