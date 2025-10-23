#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any


class PredictTransformArgs:
    pass


class PredictTransform:
    transform_args_cls: type[PredictTransformArgs]

    def __init__(
        self,
        transform_args: PredictTransformArgs,
    ) -> None:
        if not isinstance(transform_args, self.transform_args_cls):
            raise TypeError(
                f"transform_args must be of type {self.transform_args_cls.__name__}, "
                f"got {type(transform_args).__name__} instead."
            )
        self.transform_args = transform_args

    def __call__(self, input: Any) -> Any:
        raise NotImplementedError()
