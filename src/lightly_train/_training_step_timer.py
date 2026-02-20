#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Deque, Optional

import torch

if TYPE_CHECKING:
    from lightning_fabric import Fabric


class TrainingStepTimer:
    """Timer for tracking time spent in different training steps."""

    def __init__(self, cuda_utilization: Optional[CUDAUtilization] = None) -> None:
        self._step_start_times: dict[str, float] = {}
        self._step_total_times: dict[str, float] = {}
        self._step_counts: dict[str, int] = {}
        self._cuda_utilization = cuda_utilization
        self._phase_gpu_utils: dict[str, list[float]] = {}
        self._phase_gpu_max_mem: dict[str, float] = {}

    def start_step(self, step: str) -> None:
        """Start timing a step."""
        self._step_start_times[step] = time.perf_counter()

    def end_step(self, step: str) -> None:
        """Stop timing a step."""
        if step not in self._step_start_times:
            raise ValueError(f"Step '{step}' was not started")

        duration = time.perf_counter() - self._step_start_times[step]
        self._step_total_times[step] = self._step_total_times.get(step, 0.0) + duration
        self._step_counts[step] = self._step_counts.get(step, 0) + 1
        del self._step_start_times[step]

    def total_step_sec(self, step: str) -> float:
        """Get total seconds spent in step."""
        return self._step_total_times.get(step, 0.0)

    def get_step_count(self, step: str) -> int:
        """Get number of times a step was executed."""
        return self._step_counts.get(step, 0)

    def get_avg_step_time(self, step: str) -> float:
        """Get average time per step execution.

        Args:
            step: The step name.

        Returns:
            Average time in seconds, or 0.0 if step was never executed.
        """
        count = self.get_step_count(step)
        if count == 0:
            return 0.0
        return self.total_step_sec(step) / count

    def reset_gpu_max_memory(self, phase: str) -> None:
        """Reset GPU max memory tracking for a phase.

        Args:
            phase: The phase name (e.g., "train", "val").
        """
        if self._cuda_utilization is not None and self._cuda_utilization._enabled:
            torch.cuda.reset_peak_memory_stats()
            self._phase_gpu_max_mem[phase] = 0.0

    def record_gpu_stats(self, phase: str) -> None:
        """Record GPU utilization and max memory for a phase.

        Args:
            phase: The phase name (e.g., "train", "val").
        """
        if self._cuda_utilization is not None and self._cuda_utilization._enabled:
            # Get GPU utilization average
            util_avg, _ = self._cuda_utilization.drain_avg()
            if phase not in self._phase_gpu_utils:
                self._phase_gpu_utils[phase] = []
            if util_avg > 0:
                self._phase_gpu_utils[phase].append(util_avg)

            # Get GPU max memory
            max_mem_bytes = torch.cuda.max_memory_allocated()
            max_mem_mb = max_mem_bytes / (1024**2)
            self._phase_gpu_max_mem[phase] = max(
                self._phase_gpu_max_mem.get(phase, 0.0), max_mem_mb
            )

    def get_phase_gpu_util(self, phase: str) -> float:
        """Get average GPU utilization for a phase.

        Args:
            phase: The phase name (e.g., "train", "val").

        Returns:
            Average GPU utilization percentage, or 0.0 if not available.
        """
        utils = self._phase_gpu_utils.get(phase, [])
        if not utils:
            return 0.0
        return sum(utils) / len(utils)

    def get_phase_gpu_max_mem(self, phase: str) -> float:
        """Get max GPU memory for a phase in MB.

        Args:
            phase: The phase name (e.g., "train", "val").

        Returns:
            Max GPU memory in MB, or 0.0 if not available.
        """
        return self._phase_gpu_max_mem.get(phase, 0.0)

    def get_throughput(self, step: str, global_batch_size: int) -> float:
        """Calculate throughput in images per second.

        Args:
            step: The step name to calculate throughput for.
            global_batch_size: The global batch size across all GPUs.

        Returns:
            Throughput in images/second, or 0.0 if step was never executed.
        """
        avg_time = self.get_avg_step_time(step)
        if avg_time == 0.0:
            return 0.0
        return global_batch_size / avg_time

    def aggregate_across_ranks(self, fabric: Fabric) -> None:
        """Aggregate timing and GPU statistics across all ranks.

        Uses appropriate aggregation strategies:
        - Times: max (slowest GPU determines overall speed)
        - GPU utilization: mean across all GPUs
        - GPU max memory: max across all GPUs

        Args:
            fabric: The Fabric instance for distributed communication.
        """
        # Aggregate step times (use max - slowest GPU determines speed)
        for step in list(self._step_total_times.keys()):
            time_tensor = torch.tensor(
                self._step_total_times[step], device=fabric.device
            )
            max_time = fabric.all_reduce(time_tensor, reduce_op="max")
            self._step_total_times[step] = max_time.item()  # type: ignore[union-attr]

        # Step counts should be the same across all ranks, but take max to be safe
        for step in list(self._step_counts.keys()):
            count_tensor = torch.tensor(self._step_counts[step], device=fabric.device)
            max_count = fabric.all_reduce(count_tensor, reduce_op="max")
            self._step_counts[step] = int(max_count.item())  # type: ignore[union-attr]

        # Aggregate GPU utilization (use mean across GPUs)
        for phase in list(self._phase_gpu_utils.keys()):
            util = self.get_phase_gpu_util(phase)
            util_tensor = torch.tensor(util, device=fabric.device)
            mean_util = fabric.all_reduce(util_tensor, reduce_op="mean")
            # Replace with aggregated value
            self._phase_gpu_utils[phase] = [mean_util.item()]  # type: ignore[union-attr]

        # Aggregate GPU max memory (use max across GPUs)
        for phase in list(self._phase_gpu_max_mem.keys()):
            mem_tensor = torch.tensor(
                self._phase_gpu_max_mem[phase], device=fabric.device
            )
            max_mem = fabric.all_reduce(mem_tensor, reduce_op="max")
            self._phase_gpu_max_mem[phase] = max_mem.item()  # type: ignore[union-attr]


class CUDAUtilization:
    def __init__(self, device: torch.device, interval_s: float = 0.2) -> None:
        self._enabled = device.type == "cuda"
        self._device = device
        self._interval_s = float(interval_s)

        self._buf: Deque[float] = deque()
        self._lock = threading.Lock()

        self._run = threading.Event()
        self._stop = threading.Event()
        self._thr: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._enabled:
            return
        if self._thr is None:
            self._thr = threading.Thread(target=self._loop, daemon=True)
            self._thr.start()
        self._run.set()

    def pause(self) -> None:
        if not self._enabled:
            return
        self._run.clear()

    def stop(self) -> None:
        if not self._enabled:
            return
        self._run.set()
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=2.0)

    def drain(self) -> list[float]:
        if not self._enabled:
            return []
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
        return items

    def drain_avg(self) -> tuple[float, int]:
        items = self.drain()
        if not items:
            return 0.0, 0
        return (sum(items) / len(items), len(items))

    def _push(self, util: float) -> None:
        with self._lock:
            self._buf.append(float(util))

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._run.wait(0.1)
            if not self._run.is_set() or self._stop.is_set():
                continue
            try:
                self._push(float(torch.cuda.utilization(self._device)))
            except Exception:
                pass
            time.sleep(self._interval_s)
