#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pydantic import ConfigDict

from lightly_train._configs.config import PydanticConfig
from lightly_train.types import TransformInput, TransformOutput


class PredictTransformArgs(PydanticConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PredictTransform:
    transform_args_cls: type[PredictTransformArgs]

    def __init__(
        self,
        transform_args: PredictTransformArgs,
    ) -> None:
        self.transform_args = transform_args

    def __call__(self, input: TransformInput) -> TransformOutput:
        raise NotImplementedError
