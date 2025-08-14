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
    pass


class TaskTransform:
    def __init__(self, transform_args: TaskTransformArgs):
        self.transform_args = transform_args

    @staticmethod
    def transform_args_cls() -> type[TaskTransformArgs]:
        return TaskTransformArgs

    def __call__(self, input: TaskTransformInput) -> TaskTransformOutput:
        raise NotImplementedError()
