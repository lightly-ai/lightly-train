#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from multiprocessing import Value


class WorkerSharedStep:
    """A class to share the current step between dataloader workers."""

    def __init__(self, step: int) -> None:
        self._step_value = Value("i", step)

    @property
    def step(self) -> int:
        with self._step_value.get_lock():
            val = self._step_value.value
            assert isinstance(val, int)
            return val

    @step.setter
    def step(self, step: int) -> None:
        with self._step_value.get_lock():
            self._step_value.value = step
