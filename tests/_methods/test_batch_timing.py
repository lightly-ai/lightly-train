#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import time
import warnings

import pytest
import torch

from lightly_train._methods.batch_timing import (
    CpuBatchTimingTracker,
    CudaBatchTimingTracker,
    _PendingCudaSample,
    create_batch_timing_tracker,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class _FakeCudaEvent:
    """Stand-in for ``torch.cuda.Event`` usable without a GPU."""

    def __init__(self, elapsed_ms: float = 0.0) -> None:
        self._elapsed_ms = elapsed_ms

    def query(self) -> bool:
        return True

    def elapsed_time(self, end_event: "_FakeCudaEvent") -> float:
        return self._elapsed_ms


def _run_gpu_work(iters: int = 20, size: int = 2048) -> None:
    """Keep the GPU busy for a while so batch times are reliably measurable."""
    x = torch.randn(size, size, device="cuda")
    for _ in range(iters):
        x = x @ x
    del x


def test_cpu_batch_timing_tracker() -> None:
    times = iter([1.0, 1.4, 1.5, 2.1])
    tracker = CpuBatchTimingTracker(clock=times.__next__)

    assert tracker.on_batch_start() == {}
    assert tracker.on_batch_end() == pytest.approx({"batch_time": 0.4})
    assert tracker.on_batch_start() == pytest.approx({"data_time": 0.1})
    assert tracker.on_batch_end() == pytest.approx({"batch_time": 0.6})


def test_cuda_batch_timing_tracker_on_batch_end_without_start_warns() -> None:
    tracker = CudaBatchTimingTracker()

    with pytest.warns(UserWarning, match="ended before it started"):
        metrics = tracker.on_batch_end()
    assert metrics == {}

    # The warning is only shown once even if the inconsistency recurs.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = tracker.on_batch_end()
    assert metrics == {}


@requires_cuda
def test_cuda_batch_timing_tracker_on_batch_start_without_end_warns() -> None:
    tracker = CudaBatchTimingTracker()
    tracker.on_batch_start()

    with pytest.warns(UserWarning, match="started before the previous batch ended"):
        tracker.on_batch_start()

    # The warning is only shown once even if the inconsistency recurs.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tracker.on_batch_start()

    # The tracker keeps working normally afterwards.
    tracker.on_batch_end()
    torch.cuda.synchronize()
    tracker.on_batch_end()  # closes the sample started right above
    torch.cuda.synchronize()


def test_cuda_batch_timing_tracker_resolves_ready_samples_averages_multiple() -> None:
    tracker = CudaBatchTimingTracker()

    # Sample 1: batch_time = 0.1s, cycle = 0.3s -> data_time = 0.2s.
    tracker._pending_samples.append(
        _PendingCudaSample(
            wall_start_s=0.0,
            next_wall_start_s=0.3,
            start_event=_FakeCudaEvent(elapsed_ms=100.0),
            end_event=_FakeCudaEvent(),
        )
    )
    # Sample 2: batch_time = 0.9s, cycle = 1.0s -> data_time = 0.1s.
    tracker._pending_samples.append(
        _PendingCudaSample(
            wall_start_s=0.3,
            next_wall_start_s=1.3,
            start_event=_FakeCudaEvent(elapsed_ms=900.0),
            end_event=_FakeCudaEvent(),
        )
    )

    # Draining several samples at once must average, not overwrite: only one
    # value per metric name is ever returned.
    metrics = tracker._resolve_ready_samples()

    assert metrics == pytest.approx({"batch_time": 0.5, "data_time": 0.15})
    assert len(tracker._pending_samples) == 0

    # Nothing left to drain.
    assert tracker._resolve_ready_samples() == {}


@requires_cuda
def test_cuda_batch_timing_tracker_resolves_metrics_in_order() -> None:
    tracker = CudaBatchTimingTracker()
    num_cycles = 3

    for _ in range(num_cycles):
        tracker.on_batch_start()
        _run_gpu_work()
        tracker.on_batch_end()

    # Force all queued GPU work to complete, then promote and drain the last
    # active sample by starting one more (unfinished) cycle.
    torch.cuda.synchronize()
    metrics = tracker.on_batch_start()

    assert set(metrics) == {"batch_time", "data_time"}
    assert metrics["batch_time"] > 0.0
    assert metrics["data_time"] >= 0.0

    tracker.on_batch_end()
    torch.cuda.synchronize()


@requires_cuda
def test_cuda_batch_timing_tracker_on_batch_start_does_not_block() -> None:
    tracker = CudaBatchTimingTracker()

    tracker.on_batch_start()
    _run_gpu_work(iters=200, size=4096)  # Deliberately slow, still queued on GPU.
    tracker.on_batch_end()

    start_wall_s = time.perf_counter()
    tracker.on_batch_start()
    elapsed_s = time.perf_counter() - start_wall_s

    # on_batch_start must only enqueue/query events, never synchronize the
    # device, so it should return long before the queued GPU work finishes.
    assert elapsed_s < 0.02

    torch.cuda.synchronize()


@pytest.mark.parametrize(
    ("device_type", "expected_type"),
    [
        ("cpu", CpuBatchTimingTracker),
        ("mps", CpuBatchTimingTracker),
        ("cuda", CudaBatchTimingTracker),
    ],
)
def test_create_batch_timing_tracker(
    device_type: str,
    expected_type: type[CpuBatchTimingTracker] | type[CudaBatchTimingTracker],
) -> None:
    assert isinstance(create_batch_timing_tracker(device_type), expected_type)
