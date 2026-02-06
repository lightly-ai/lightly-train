#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import time


class TrainingStepTimer:
    """Timer for tracking time spent in different training steps."""

    def __init__(self) -> None:
        self._step_start_times: dict[str, float] = {}
        self._step_last_durations: dict[str, float] = {}
        self._step_total_times: dict[str, float] = {}

    def start_step(self, step: str) -> None:
        """Start timing a step."""
        self._step_start_times[step] = time.perf_counter()

    def end_step(self, step: str) -> None:
        """Stop timing a step."""
        if step not in self._step_start_times:
            raise ValueError(f"Step '{step}' was not started")

        duration = time.perf_counter() - self._step_start_times[step]
        self._step_last_durations[step] = duration
        self._step_total_times[step] = self._step_total_times.get(step, 0.0) + duration
        del self._step_start_times[step]

    def last_step_sec(self, step: str) -> float:
        """Get seconds the last step took."""
        return self._step_last_durations.get(step, 0.0)

    def total_step_sec(self, step: str) -> float:
        """Get total seconds spent in step."""
        return self._step_total_times.get(step, 0.0)

    def total_percentage(self, steps: list[str] | None = None) -> dict[str, float]:
        """Get percentage of time spent in each step."""
        if steps is None:
            steps = list(self._step_total_times.keys())
        total_time = sum(self.total_step_sec(step) for step in steps)
        if total_time == 0:
            return {step: 0.0 for step in steps}
        return {step: (self.total_step_sec(step) / total_time) * 100 for step in steps}
