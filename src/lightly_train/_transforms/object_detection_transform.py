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
from pydantic import Field
from torch import Tensor

from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
)
from lightly_train._transforms.transform import RandomPhotometricDistortArgs


class ObjectDetectionTransformInput(TaskTransformInput):
    image: NDArray[np.uint8]
    bboxes: NDArray[np.float64]
    class_labels: NDArray[np.int64]


class ObjectDetectionTransformOutput(TaskTransformOutput):
    image: Tensor
    bboxes: Tensor
    class_labels: Tensor


class ObjectDetectionTransformArgs(TaskTransformArgs):
    photometric_distort: RandomPhotometricDistortArgs

    # We use the YOLO format internally for now.
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo", label_fields=["class_labels"]
        ),
    )


class ObjectDetectionTransform(TaskTransform):
    transform_args_cls = ObjectDetectionTransformArgs

    def __call__(  # type: ignore[empty-body]
        self, input: ObjectDetectionTransformInput
    ) -> ObjectDetectionTransformOutput:
        pass
