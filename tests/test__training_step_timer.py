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
from lightning_fabric import Fabric

from lightly_train._training_step_timer import CUDAUtilization, TrainingStepTimer


class TestTrainingStepTimer:
    """Tests for TrainingStepTimer class."""

    def test_start_end_step(self) -> None:
        """Test that total accumulates across multiple executions."""
        fabric = Fabric(accelerator="cpu", devices=1, num_nodes=1)
        fabric.launch()
        timer = TrainingStepTimer()

        timer.start_step("step")
        time.sleep(0.01)
        timer.end_step("step")
        first_total = timer.get_aggregated_metrics(fabric)["step_total_times"]["step"]

        timer.start_step("step")
        time.sleep(0.02)
        timer.end_step("step")
        second_total = timer.get_aggregated_metrics(fabric)["step_total_times"]["step"]

        assert second_total > first_total
        assert second_total >= 0.01

    def test_end_step__without_start(self) -> None:
        """Test that ending a step without starting it raises an error."""
        timer = TrainingStepTimer()

        with pytest.raises(ValueError, match="was not started"):
            timer.end_step("nonexistent")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_stats__cuda(self) -> None:
        """Test GPU stats tracking when CUDA is available."""
        fabric = Fabric(accelerator="cuda", devices=1, num_nodes=1)
        fabric.launch()
        device = torch.device("cuda")
        cuda_util = CUDAUtilization(device=device)
        timer = TrainingStepTimer(cuda_utilization=cuda_util)

        timer.reset_gpu_max_memory("train")
        timer.record_gpu_stats("train")

        agg = timer.get_aggregated_metrics(fabric)
        util = agg["phase_gpu_utils"]["train"]
        max_mem = agg["phase_gpu_max_mem"]["train"]

        assert 0.0 <= util <= 100.0
        assert max_mem >= 0.0

    def test_gpu_stats__cpu(self) -> None:
        """Test GPU stats methods when CUDA is not available."""
        fabric = Fabric(accelerator="cpu", devices=1, num_nodes=1)
        fabric.launch()
        device = torch.device("cpu")
        cuda_util = CUDAUtilization(device=device)
        timer = TrainingStepTimer(cuda_utilization=cuda_util)

        # Should not raise errors.
        timer.reset_gpu_max_memory("train")
        timer.record_gpu_stats("train")

        agg = timer.get_aggregated_metrics(fabric)
        assert not agg["phase_gpu_utils"]  # Should be empty
        assert not agg["phase_gpu_max_mem"]  # Should be empty

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_aggregated_metrics__cuda(self) -> None:
        fabric = Fabric(accelerator="cuda", devices=1, num_nodes=1)
        fabric.launch()
        timer = TrainingStepTimer()

        # Add some timing data
        timer.start_step("train_step")
        time.sleep(0.01)
        timer.end_step("train_step")

        timer.start_step("val_step")
        time.sleep(0.01)
        timer.end_step("val_step")

        # Get aggregated metrics
        agg = timer.get_aggregated_metrics(fabric)

        # Check that all expected keys are present
        assert "step_total_times" in agg
        assert "step_counts" in agg
        assert "phase_gpu_utils" in agg
        assert "phase_gpu_max_mem" in agg

        # Check that counts are correct
        assert agg["step_counts"]["train_step"] == 1
        assert agg["step_counts"]["val_step"] == 1

        # Check that times are reasonable
        assert agg["step_total_times"]["train_step"] >= 0.005
        assert agg["step_total_times"]["val_step"] >= 0.005
        assert agg["phase_gpu_utils"]["train"] >= 0.0
        assert agg["phase_gpu_max_mem"]["train"] >= 0.0

    def test_get_aggregated_metrics__cpu(self) -> None:
        fabric = Fabric(accelerator="cpu", devices=1, num_nodes=1)
        fabric.launch()
        timer = TrainingStepTimer()

        # Add some timing data
        timer.start_step("train_step")
        time.sleep(0.01)
        timer.end_step("train_step")

        timer.start_step("val_step")
        time.sleep(0.01)
        timer.end_step("val_step")

        # Get aggregated metrics
        agg = timer.get_aggregated_metrics(fabric)

        # Check that all expected keys are present
        assert "step_total_times" in agg
        assert "step_counts" in agg
        assert "phase_gpu_utils" in agg
        assert "phase_gpu_max_mem" in agg

        # Check that counts are correct
        assert agg["step_counts"]["train_step"] == 1
        assert agg["step_counts"]["val_step"] == 1

        # Check that times are reasonable
        assert agg["step_total_times"]["train_step"] >= 0.005
        assert agg["step_total_times"]["val_step"] >= 0.005
