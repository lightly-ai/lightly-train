#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import TypedDict

import numpy as np
from albumentations import BboxParams
from numpy.typing import NDArray
from pydantic import Field
from torch import Tensor
from typing_extensions import NotRequired

from lightly_train._configs.config import PydanticConfig


class TaskTransformInput(TypedDict):
    image: NDArray[np.uint8]
    mask: NotRequired[NDArray[np.uint8]]
    bboxes: NotRequired[NDArray[np.float64]]
    class_labels: NotRequired[NDArray[np.int64]]


class TaskTransformOutput(TypedDict):
    image: Tensor
    mask: NotRequired[Tensor]
    bboxes: NotRequired[Tensor]
    class_labels: NotRequired[Tensor]


class TaskTransformArgs(PydanticConfig):
    # We use the YOLO format internally for now.
    bbox_params: BboxParams = Field(
        default_factory=lambda: BboxParams(
            format="yolo", label_fields=["class_labels"]
        ),
    )

    class Config:
        arbitrary_types_allowed = True


class TaskTransform:
    transform_args_cls: type[TaskTransformArgs]

    def __init__(self, transform_args: TaskTransformArgs):
        self.transform_args = transform_args

    def __call__(self, input: TaskTransformInput) -> TaskTransformOutput:
        raise NotImplementedError()
