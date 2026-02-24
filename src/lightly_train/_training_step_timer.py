#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import time
from collections import deque
from threading import Event, Lock, Thread
from typing import Deque, TypedDict

import torch
from lightning_fabric import Fabric
from torch import Tensor


class TimerAggregateMetrics(TypedDict):
    step_total_times: dict[str, float]
    step_counts: dict[str, int]
    phase_gpu_utils: dict[str, float]
    phase_gpu_max_mem: dict[str, float]


class TrainingStepTimer:
    """Timer for tracking time spent in different training steps."""

    def __init__(self, cuda_utilization: CUDAUtilization | None = None) -> None:
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
            util_avg, count = self._cuda_utilization.drain_avg()
            if phase not in self._phase_gpu_utils:
                self._phase_gpu_utils[phase] = []
            if count > 0:
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
            Average GPU utilization percentage, or -1 if not available.
        """
        utils = self._phase_gpu_utils.get(phase, [])
        if not utils:
            return -1.0
        return sum(utils) / len(utils)

    def get_aggregated_metrics(self, fabric: Fabric) -> TimerAggregateMetrics:
        """Aggregate timing and GPU statistics across all ranks, returning a summary dict.

        Returns:
            Dictionary with aggregated metrics
        """
        # Prepare dicts of tensors for reduction
        max_dict = {
            "step_total_times": {
                step: torch.tensor(val, device=fabric.device)
                for step, val in self._step_total_times.items()
            },
            "step_counts": {
                step: torch.tensor(val, device=fabric.device)
                for step, val in self._step_counts.items()
            },
            "phase_gpu_max_mem": {
                phase: torch.tensor(val, device=fabric.device)
                for phase, val in self._phase_gpu_max_mem.items()
            },
        }
        mean_dict = {
            "phase_gpu_utils": {
                phase: torch.tensor(
                    self.get_phase_gpu_util(phase), device=fabric.device
                )
                for phase in self._phase_gpu_utils.keys()
            },
        }

        # Reduce dicts
        max_result: dict[str, dict[str, Tensor]] = fabric.all_reduce(  # type: ignore[assignment]
            max_dict, reduce_op="max"
        )
        mean_result: dict[str, dict[str, Tensor]] = fabric.all_reduce(  # type: ignore[assignment]
            mean_dict, reduce_op="mean"
        )

        # Convert tensors to Python scalars
        agg_step_total_times = {
            k: v.item() for k, v in max_result["step_total_times"].items()
        }
        agg_step_counts = {
            k: int(v.item()) for k, v in max_result["step_counts"].items()
        }
        agg_phase_gpu_max_mem = {
            k: v.item() for k, v in max_result["phase_gpu_max_mem"].items()
        }
        agg_phase_gpu_utils = {
            k: v.item() for k, v in mean_result["phase_gpu_utils"].items()
        }

        return {
            "step_total_times": agg_step_total_times,
            "step_counts": agg_step_counts,
            "phase_gpu_utils": agg_phase_gpu_utils,
            "phase_gpu_max_mem": agg_phase_gpu_max_mem,
        }


class CUDAUtilization:
    def __init__(self, device: torch.device, interval_s: float = 0.2) -> None:
        self._enabled = device.type == "cuda"
        self._device = device
        self._interval_s = float(interval_s)

        self._buf: Deque[float] = deque()
        self._lock = Lock()

        self._run = Event()
        self._stop = Event()
        self._thr: Thread | None = None

    def start(self) -> None:
        if not self._enabled:
            return
        if self._thr is None:
            self._thr = Thread(target=self._loop, daemon=True)
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

    def __del__(self) -> None:
        self.stop()

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
