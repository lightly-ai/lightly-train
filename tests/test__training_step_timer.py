#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import time

from lightly_train._training_step_timer import TrainingStepTimer


class TestTrainingStepTimer:
    """Tests for TrainingStepTimer class."""

    def test_training_step_timer__basic_timing(self) -> None:
        """Test basic start/stop timing."""
        timer = TrainingStepTimer()

        timer.start_step("forward")
        time.sleep(0.01)
        timer.end_step("forward")

        assert timer.last_step_sec("forward") >= 0.01
        assert timer.total_step_sec("forward") >= 0.01

    def test_training_step_timer__total_and_last_step(self) -> None:
        """Test that total accumulates and last tracks most recent."""
        timer = TrainingStepTimer()

        # First execution.
        timer.start_step("backward")
        time.sleep(0.01)
        timer.end_step("backward")
        first_duration = timer.last_step_sec("backward")

        # Second execution.
        timer.start_step("backward")
        time.sleep(0.02)
        timer.end_step("backward")
        second_duration = timer.last_step_sec("backward")

        assert second_duration > first_duration
        assert timer.total_step_sec("backward") >= first_duration + second_duration

    def test_training_step_timer__percentage_calculation(self) -> None:
        """Test percentage calculation."""
        timer = TrainingStepTimer()

        # Simulate timing.
        timer.start_step("forward")
        time.sleep(0.01)
        timer.end_step("forward")

        timer.start_step("backward")
        time.sleep(0.01)
        timer.end_step("backward")

        timer.start_step("data_loading")
        time.sleep(0.02)
        timer.end_step("data_loading")

        percentages = timer.total_percentage(["forward", "backward", "data_loading"])

        # Check all keys present.
        assert set(percentages.keys()) == {"forward", "backward", "data_loading"}

        # Check percentages sum to 100.
        assert abs(sum(percentages.values()) - 100.0) < 0.1

        # data_loading should be roughly 50% since it took 0.02s out of ~0.04s total.
        assert 45 < percentages["data_loading"] < 55
