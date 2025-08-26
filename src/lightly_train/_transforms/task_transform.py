#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from lightly_train._configs.config import PydanticConfig

TaskTransformInput = Dict[str, NDArray[np.uint8]]
TaskTransformOutput = Dict[str, Tensor]


class TaskTransformArgs(PydanticConfig):
    def resolve_auto(self) -> None:
        pass


class TaskTransform:
    transform_args_cls: type[TaskTransformArgs]

    def __init__(self, transform_args: TaskTransformArgs):
        if not isinstance(transform_args, self.transform_args_cls):
            raise TypeError(
                f"transform_args must be of type {self.transform_args_cls.__name__}, "
                f"got {type(transform_args).__name__} instead."
            )
        self.transform_args = transform_args

    def __call__(self, input: TaskTransformInput) -> TaskTransformOutput:
        raise NotImplementedError()
