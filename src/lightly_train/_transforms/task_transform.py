#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from multiprocessing.managers import DictProxy
from typing import Any, TypedDict

from pydantic import ConfigDict

from lightly_train._configs.config import PydanticConfig


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

    def __init__(
        self,
        transform_args: TaskTransformArgs,
        shared_dict: DictProxy[str, Any] | dict[str, Any] | None = None,
    ) -> None:
        if shared_dict is None:
            shared_dict = {"step": 0}
        self._shared_dict = shared_dict
        if not isinstance(transform_args, self.transform_args_cls):
            raise TypeError(
                f"transform_args must be of type {self.transform_args_cls.__name__}, "
                f"got {type(transform_args).__name__} instead."
            )
        self.transform_args = transform_args

    @property
    def global_step(self) -> int:
        return int(self._shared_dict["step"])

    @global_step.setter
    def global_step(self, step: int) -> None:
        self._shared_dict["step"] = step

    def __call__(self, input: Any) -> Any:
        raise NotImplementedError()
