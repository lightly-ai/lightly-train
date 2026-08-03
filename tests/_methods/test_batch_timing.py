#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from lightly_train._methods.batch_timing import (
    CpuBatchTimingTracker,
    CudaBatchTimingTracker,
    create_batch_timing_tracker,
)


class FakeCudaEvent:
    def __init__(self, *, elapsed_ms: float = 0.0, ready: bool = False) -> None:
        self.elapsed_ms = elapsed_ms
        self.ready = ready
        self.recorded = False

    def record(self) -> None:
        self.recorded = True

    def query(self) -> bool:
        return self.ready

    def elapsed_time(self, end_event: object) -> float:
        assert isinstance(end_event, FakeCudaEvent)
        assert end_event.recorded
        return self.elapsed_ms


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


def test_cuda_batch_timing_tracker_queues_unready_samples(
    mocker: MockerFixture,
) -> None:
    times = iter([1.0, 1.5, 2.0])
    created_events = [
        FakeCudaEvent(elapsed_ms=400.0),
        FakeCudaEvent(ready=False),
        FakeCudaEvent(elapsed_ms=700.0),
        FakeCudaEvent(ready=True),
        FakeCudaEvent(),
    ]

    mocker.patch(
        "lightly_train._methods.batch_timing.Event", side_effect=created_events
    )
    tracker = CudaBatchTimingTracker(clock=times.__next__)

    assert tracker.on_batch_start() == []
    assert tracker.on_batch_end() == []
    assert tracker.on_batch_start() == []
    assert tracker.on_batch_end() == []

    # The first sample was retained when it was not ready at the previous start.
    created_events[1].ready = True
    assert tracker.on_batch_start() == [
        ("batch_time", pytest.approx(0.4)),
        ("data_time", pytest.approx(0.1)),
        ("batch_time", pytest.approx(0.7)),
        ("data_time", 0.0),
    ]
    assert all(event.recorded for event in created_events)


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
