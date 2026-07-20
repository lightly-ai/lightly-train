#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import ConfigDict

from lightly_train._configs.config import PydanticConfig


class TaskTransformInput(TypedDict):
    pass


class TaskTransformOutput(TypedDict):
    pass


class TaskTransformArgs(PydanticConfig):
    def resolve_auto(self, model_init_args: dict[str, Any]) -> None:
        """Resolve any arguments set to "auto"."""
        pass

    def resolve_step_schedule(
        self,
        total_steps: int,
        train_num_batches: int,
        gradient_accumulation_steps: int,
    ) -> None:
        """Resolve any step-scheduled transform arguments."""
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
    ) -> None:
        if not isinstance(transform_args, self.transform_args_cls):
            raise TypeError(
                f"transform_args must be of type {self.transform_args_cls.__name__}, "
                f"got {type(transform_args).__name__} instead."
            )
        self.transform_args = transform_args

    def set_step(self, step: int) -> None:
        """Update the current training step for step-aware augmentation logic."""
        pass

    def uses_step_dependent_worker_state(self) -> bool:
        """Report whether worker-side behavior changes with the training step."""
        return False

    def requires_dataloader_reinitialization(self) -> bool:
        """Signal when worker-visible state changed and the loader iterator must reset."""
        return False

    def mark_dataloader_as_reinitialized(self) -> None:
        """Update internal active-status state for step-aware transforms after a dataloader refresh."""
        pass

    def __call__(self, input: Any) -> Any:
        raise NotImplementedError()


class TaskCollateFunction:
    def __init__(
        self, split: Literal["train", "val"], transform_args: TaskTransformArgs
    ):
        self.split = split
        self.transform_args = transform_args

    def set_step(self, step: int) -> None:
        """Update the current training step for step-aware batch processing logic."""
        pass

    def uses_step_dependent_worker_state(self) -> bool:
        """Report whether worker-side collate behavior changes with the training step."""
        return False

    def requires_dataloader_reinitialization(self) -> bool:
        """Signal when worker-visible collate state changed and the loader must reset."""
        return False

    def mark_dataloader_as_reinitialized(self) -> None:
        """Update internal active-status state for step-aware transforms after a dataloader refresh."""
        pass
