#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
from albumentations import BboxParams
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import NotRequired

from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
)
from lightly_train._transforms.transform import (
    RandomFlipArgs,
    RandomPhotometricDistortArgs,
    RandomZoomOutArgs,
    ResizeArgs,
)
from lightly_train.types import NDArrayImage


class ObjectDetectionTransformInput(TaskTransformInput):
    image: NDArrayImage
    bboxes: NotRequired[NDArray[np.float64]]
    class_labels: NotRequired[NDArray[np.int64]]


class ObjectDetectionTransformOutput(TaskTransformOutput):
    image: Tensor
    bboxes: NotRequired[Tensor]
    class_labels: NotRequired[Tensor]


class ObjectDetectionTransformArgs(TaskTransformArgs):
    photometric_distort: RandomPhotometricDistortArgs | None
    random_zoom_out: RandomZoomOutArgs | None
    random_flip: RandomFlipArgs | None
    resize: ResizeArgs | None
    bbox_params: BboxParams


class ObjectDetectionTransform(TaskTransform):
    transform_args_cls = ObjectDetectionTransformArgs

    def __call__(  # type: ignore[empty-body]
        self, input: ObjectDetectionTransformInput
    ) -> ObjectDetectionTransformOutput:
        pass
