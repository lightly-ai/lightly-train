#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, TypedDict

from pydantic import ConfigDict

from lightly_train._configs.config import PydanticConfig
from lightly_train._data.dataloader_helpers import WorkerSharedStep


class TaskTransformInput(TypedDict):
    pass


class TaskTransformOutput(TypedDict):
    pass


class TaskTransformArgs(PydanticConfig):
    def resolve_auto(self) -> None:
        """Resolve any arguments set to "auto"."""
        pass

    def resolve_incompatible(self) -> None:
        """Resolve any incompatible arguments."""
        pass

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TaskTransform:
    transform_args_cls: type[TaskTransformArgs]

    def __init__(self, transform_args: TaskTransformArgs):
        self._global_step = WorkerSharedStep(step=0)
        if not isinstance(transform_args, self.transform_args_cls):
            raise TypeError(
                f"transform_args must be of type {self.transform_args_cls.__name__}, "
                f"got {type(transform_args).__name__} instead."
            )
        self.transform_args = transform_args

    @property
    def global_step(self) -> int:
        return self._global_step.step

    @global_step.setter
    def global_step(self, step: int) -> None:
        self._global_step.step = step

    def __call__(self, input: Any) -> Any:
        raise NotImplementedError()
