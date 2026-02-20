#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import time

import pytest
import torch

from lightly_train._training_step_timer import CUDAUtilization, TrainingStepTimer


class TestTrainingStepTimer:
    """Tests for TrainingStepTimer class."""

    def test_total_step_sec__basic(self) -> None:
        """Test basic start/stop timing."""
        timer = TrainingStepTimer()

        timer.start_step("step")
        time.sleep(0.01)
        timer.end_step("step")

        assert timer.total_step_sec("step") >= 0.005

    def test_total_step_sec__accumulates(self) -> None:
        """Test that total accumulates across multiple executions."""
        timer = TrainingStepTimer()

        timer.start_step("step")
        time.sleep(0.01)
        timer.end_step("step")
        first_total = timer.total_step_sec("step")

        timer.start_step("step")
        time.sleep(0.02)
        timer.end_step("step")
        second_total = timer.total_step_sec("step")

        assert second_total > first_total
        assert second_total >= 0.01

    def test_get_step_count(self) -> None:
        """Test step count tracking."""
        timer = TrainingStepTimer()

        assert timer.get_step_count("step") == 0

        timer.start_step("step")
        timer.end_step("step")
        assert timer.get_step_count("step") == 1

        timer.start_step("step")
        timer.end_step("step")
        assert timer.get_step_count("step") == 2

    def test_get_avg_step_time(self) -> None:
        """Test average step time calculation."""
        timer = TrainingStepTimer()

        # No steps executed yet.
        assert timer.get_avg_step_time("step") == 0.0

        timer.start_step("step")
        time.sleep(0.01)
        timer.end_step("step")

        timer.start_step("step")
        time.sleep(0.01)
        timer.end_step("step")

        avg_time = timer.get_avg_step_time("step")
        assert avg_time >= 0.005

    def test_get_throughput(self) -> None:
        """Test throughput calculation."""
        timer = TrainingStepTimer()

        # No steps executed yet.
        assert timer.get_throughput("step", 32) == 0.0

        timer.start_step("step")
        time.sleep(0.01)
        timer.end_step("step")

        throughput = timer.get_throughput("step", 32)
        assert throughput > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_stats(self) -> None:
        """Test GPU stats tracking when CUDA is available."""
        device = torch.device("cuda")
        cuda_util = CUDAUtilization(device=device)
        timer = TrainingStepTimer(cuda_utilization=cuda_util)

        timer.reset_gpu_max_memory("train")
        timer.record_gpu_stats("train")

        util = timer.get_phase_gpu_util("train")
        max_mem = timer.get_phase_gpu_max_mem("train")

        assert 0.0 <= util <= 100.0
        assert max_mem >= 0.0

    def test_gpu_stats__no_cuda(self) -> None:
        """Test GPU stats methods when CUDA is not available."""
        device = torch.device("cpu")
        cuda_util = CUDAUtilization(device=device)
        timer = TrainingStepTimer(cuda_utilization=cuda_util)

        # Should not raise errors.
        timer.reset_gpu_max_memory("train")
        timer.record_gpu_stats("train")

        assert timer.get_phase_gpu_util("train") == 0.0
        assert timer.get_phase_gpu_max_mem("train") == 0.0

    def test_end_step__without_start(self) -> None:
        """Test that ending a step without starting it raises an error."""
        timer = TrainingStepTimer()

        with pytest.raises(ValueError, match="was not started"):
            timer.end_step("nonexistent")
