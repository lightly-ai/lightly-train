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

from lightly_train._methods.batch_timing import (
    CpuBatchTimingTracker,
    CudaBatchTimingTracker,
    TimingMetric,
    create_batch_timing_tracker,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _run_gpu_work(iters: int = 20, size: int = 2048) -> None:
    """Keep the GPU busy for a while so batch times are reliably measurable."""
    x = torch.randn(size, size, device="cuda")
    for _ in range(iters):
        x = x @ x
    del x


def test_cpu_batch_timing_tracker() -> None:
    times = iter([1.0, 1.4, 1.5, 2.1])
    tracker = CpuBatchTimingTracker(clock=times.__next__)

    assert tracker.on_batch_start() == []
    batch_name, batch_time_s = tracker.on_batch_end()[0]
    assert batch_name == "batch_time"
    assert batch_time_s == pytest.approx(0.4)

    data_name, data_time_s = tracker.on_batch_start()[0]
    assert data_name == "data_time"
    assert data_time_s == pytest.approx(0.1)

    batch_name, batch_time_s = tracker.on_batch_end()[0]
    assert batch_name == "batch_time"
    assert batch_time_s == pytest.approx(0.6)


def test_cuda_batch_timing_tracker_on_batch_end_without_start_raises() -> None:
    tracker = CudaBatchTimingTracker()
    with pytest.raises(RuntimeError, match="ended before it started"):
        tracker.on_batch_end()


@requires_cuda
def test_cuda_batch_timing_tracker_on_batch_start_without_end_raises() -> None:
    tracker = CudaBatchTimingTracker()
    tracker.on_batch_start()
    with pytest.raises(RuntimeError, match="started before the previous batch ended"):
        tracker.on_batch_start()


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
    metrics: list[TimingMetric] = tracker.on_batch_start()

    assert len(metrics) == num_cycles * 2
    names = [name for name, _ in metrics]
    assert names == ["batch_time", "data_time"] * num_cycles
    for name, value_s in metrics:
        if name == "batch_time":
            assert value_s > 0.0
        else:
            assert value_s >= 0.0

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
