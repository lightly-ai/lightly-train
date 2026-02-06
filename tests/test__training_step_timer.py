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

    def test_total_step_sec__basic(self) -> None:
        """Test basic start/stop timing."""
        timer = TrainingStepTimer()

        timer.start_step("forward")
        time.sleep(0.01)
        timer.end_step("forward")

        assert timer.total_step_sec("forward") >= 0.005

    def test_total_step_sec__accumulates(self) -> None:
        """Test that total accumulates across multiple executions."""
        timer = TrainingStepTimer()

        # First execution.
        timer.start_step("backward")
        time.sleep(0.01)
        timer.end_step("backward")
        first_total = timer.total_step_sec("backward")

        # Second execution.
        timer.start_step("backward")
        time.sleep(0.02)
        timer.end_step("backward")
        second_total = timer.total_step_sec("backward")

        assert second_total > first_total
        assert second_total >= 0.02

    def test_total_percentage(self) -> None:
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
        assert 40 < percentages["data_loading"] < 60

    def test_percentage_for_prefix(self) -> None:
        """Test percentage calculation for steps with a given prefix."""
        timer = TrainingStepTimer()

        timer.start_step("train_forward")
        time.sleep(0.01)
        timer.end_step("train_forward")

        timer.start_step("train_backward")
        time.sleep(0.01)
        timer.end_step("train_backward")

        timer.start_step("train_optimizer")
        time.sleep(0.02)
        timer.end_step("train_optimizer")

        percentages = timer.percentage_for_prefix("train_")

        # Check all keys present with prefix removed.
        assert set(percentages.keys()) == {"forward", "backward", "optimizer"}

        # Check percentages sum to 100.
        assert abs(sum(percentages.values()) - 100.0) < 0.1

        # optimizer should be roughly 50% since it took 0.02s out of ~0.04s total.
        assert 40 < percentages["optimizer"] < 60

    def test_percentage_for_prefix__empty(self) -> None:
        """Test percentage calculation with no matching steps."""
        timer = TrainingStepTimer()

        timer.start_step("forward")
        time.sleep(0.01)
        timer.end_step("forward")

        percentages = timer.percentage_for_prefix("nonexistent_")

        assert percentages == {}

    def test_percentage_for_prefix_group(self) -> None:
        """Test percentage calculation for groups of steps."""
        timer = TrainingStepTimer()

        timer.start_step("train_forward")
        time.sleep(0.01)
        timer.end_step("train_forward")

        timer.start_step("train_backward")
        time.sleep(0.01)
        timer.end_step("train_backward")

        timer.start_step("val_forward")
        time.sleep(0.02)
        timer.end_step("val_forward")

        timer.start_step("data_loading")
        time.sleep(0.02)
        timer.end_step("data_loading")

        percentages = timer.percentage_for_prefix_group(
            {
                "training": ["train_"],
                "validation": ["val_"],
                "data": ["data_"],
            }
        )

        # Check all groups present.
        assert set(percentages.keys()) == {"training", "validation", "data"}

        # Check percentages sum to 100 (within rounding tolerance).
        assert abs(sum(percentages.values()) - 100.0) < 0.2

        # Training and validation should each be roughly 33% and data should be 33%.
        assert 25 < percentages["training"] < 45
        assert 25 < percentages["validation"] < 45
        assert 25 < percentages["data"] < 45

    def test_end_step__without_start(self) -> None:
        """Test that ending a step without starting it raises an error."""
        timer = TrainingStepTimer()

        try:
            timer.end_step("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "was not started" in str(e)

    def test_total_percentage__none_steps(self) -> None:
        """Test total_percentage with None steps includes all steps."""
        timer = TrainingStepTimer()

        timer.start_step("step1")
        time.sleep(0.01)
        timer.end_step("step1")

        timer.start_step("step2")
        time.sleep(0.01)
        timer.end_step("step2")

        percentages = timer.total_percentage()

        assert set(percentages.keys()) == {"step1", "step2"}
        assert abs(sum(percentages.values()) - 100.0) < 0.1
