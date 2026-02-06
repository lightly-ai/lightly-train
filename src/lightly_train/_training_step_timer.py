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

    def total_step_sec(self, step: str) -> float:
        """Get total seconds spent in step."""
        return self._step_total_times.get(step, 0.0)

    def total_percentage(
        self, steps: list[str] | None = None, decimal_places: int = 1
    ) -> dict[str, float]:
        """Get percentage of time spent in each step.

        Args:
            steps: List of step names to include. If None, includes all steps.
            decimal_places: Number of decimal places to round to.

        Returns:
            Dictionary mapping step names to their percentage of total time.
        """
        if steps is None:
            steps = list(self._step_total_times.keys())
        total_time = sum(self.total_step_sec(step) for step in steps)
        if total_time == 0:
            return {step: 0.0 for step in steps}

        return {
            step: round((self.total_step_sec(step) / total_time) * 100, decimal_places)
            for step in steps
        }

    def percentage_for_prefix(
        self, prefix: str, decimal_places: int = 1
    ) -> dict[str, float]:
        """Get percentage of time spent in steps starting with the given prefix.

        Args:
            prefix: Prefix to filter steps by.
            decimal_places: Number of decimal places to round to.

        Returns:
            Dictionary mapping step names (with prefix removed) to their percentage
            of total time across all steps.
        """
        matching_steps = [
            s for s in self._step_total_times.keys() if s.startswith(prefix)
        ]
        if not matching_steps:
            return {}

        total_time = sum(self.total_step_sec(step) for step in matching_steps)
        if total_time == 0:
            return {step[len(prefix) :]: 0.0 for step in matching_steps}

        return {
            step[len(prefix) :]: round(
                (self.total_step_sec(step) / total_time) * 100, decimal_places
            )
            for step in matching_steps
        }

    def percentage_for_prefix_group(
        self, prefixes: dict[str, list[str]], decimal_places: int = 1
    ) -> dict[str, float]:
        """Get percentage of time spent in groups of steps defined by prefixes.

        Args:
            prefixes: Dictionary mapping group names to lists of step prefixes.
            decimal_places: Number of decimal places to round to.
        Returns:
            Dictionary mapping group names to their percentage of total time across all steps.
        """
        seen_steps = set()
        group_times = {}
        for group_name, prefix_list in prefixes.items():
            group_time = 0.0
            for step in self._step_total_times.keys():
                if step in seen_steps:
                    continue
                if any(step.startswith(p) for p in prefix_list):
                    seen_steps.add(step)
                    group_time += self.total_step_sec(step)
            group_times[group_name] = group_time

        total_time = sum(group_times.values())
        if total_time == 0:
            return {group: 0.0 for group in group_times}

        return {
            group: round((time / total_time) * 100, decimal_places)
            for group, time in group_times.items()
        }
