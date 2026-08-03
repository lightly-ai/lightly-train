#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Non-blocking measurement of training batch and data-loading times."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, Tuple

from torch.cuda import Event

TimingMetricName = Literal["batch_time", "data_time"]
TimingMetric = Tuple[TimingMetricName, float]


class BatchTimingTracker(Protocol):
    """Measure timing metrics at the boundaries of Lightning's batch hooks.

    ``on_batch_start`` is called after Lightning has fetched a batch, and
    ``on_batch_end`` is called after the training batch hook has returned. A
    tracker may emit metrics at either boundary. CUDA metrics can be delayed
    until the corresponding stream events have completed.
    """

    def on_batch_start(self) -> list[TimingMetric]:
        """Record a batch-start boundary and return metrics ready for logging."""
        ...

    def on_batch_end(self) -> list[TimingMetric]:
        """Record a batch-end boundary and return metrics ready for logging."""
        ...


class CpuBatchTimingTracker(BatchTimingTracker):
    """Measure synchronous CPU batch execution and data-loading gaps.

    CPU execution is synchronous, so wall-clock timestamps at the Lightning
    hook boundaries directly delimit batch execution and the subsequent wait
    for the next batch.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._batch_started_at_s: float | None = None
        self._previous_batch_ended_at_s: float | None = None

    def on_batch_start(self) -> list[TimingMetric]:
        """Measure the gap since the previous end and start the new batch."""
        now_s = self._clock()
        metrics: list[TimingMetric] = []
        if self._previous_batch_ended_at_s is not None:
            metrics.append(("data_time", now_s - self._previous_batch_ended_at_s))
        self._batch_started_at_s = now_s
        return metrics

    def on_batch_end(self) -> list[TimingMetric]:
        """Measure the active batch and begin the next data-loading gap."""
        now_s = self._clock()
        metrics: list[TimingMetric] = []
        if self._batch_started_at_s is not None:
            metrics.append(("batch_time", now_s - self._batch_started_at_s))
        self._previous_batch_ended_at_s = now_s
        return metrics


@dataclass
class _ActiveCudaSample:
    """CUDA timing state for the one batch currently being timed.

    At most one instance exists at a time. It is created in
    ``on_batch_start`` and then mutated in place by ``on_batch_end``, which
    fills in ``end_event``. It is replaced only when the following
    ``on_batch_start`` promotes it to a ``_PendingCudaSample``.
    """

    wall_start_s: float
    start_event: Event
    end_event: Event | None = None


@dataclass(frozen=True)
class _PendingCudaSample:
    """An ``_ActiveCudaSample`` promoted once both its events are recorded.

    Multiple instances can coexist in the pending deque, because CUDA
    execution is asynchronous and the CPU may race several batches ahead of
    the GPU before any single end event has executed. The dataclass is
    frozen since nothing about a sample changes again after promotion — it
    only waits in the deque until its end event is ready to be queried.
    """

    wall_start_s: float
    next_wall_start_s: float
    start_event: Event
    end_event: Event


class CudaBatchTimingTracker(BatchTimingTracker):
    """Measure CUDA batch time without synchronizing the device.

    Calling :meth:`torch.cuda.Event.record` enqueues an event marker on the
    current CUDA stream. It does not capture the CPU call time. The GPU
    timestamp is captured only when stream execution reaches that marker,
    after all previously queued work. Events bracketing a batch therefore
    measure its GPU execution even though CUDA launches are asynchronous.

    Completed samples remain queued until ``Event.query()`` reports that their
    end marker has executed. Querying is non-blocking, so profiling does not
    serialize the CPU with the GPU. Consequently, metrics can be emitted a few
    batches after they were measured.

    Lifecycle, one training cycle at a time:

    - ``on_batch_start``: promote the previous ``_ActiveCudaSample`` (if any)
      to a ``_PendingCudaSample`` on the pending deque, then start a new
      active sample for the batch that is about to run.
    - ``on_batch_end``: record the end event onto the *current* active
      sample in place; no new sample is created here.
    - Pending samples drain from the front of the deque, in stream order, as
      soon as their end events report completion.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._active_sample: _ActiveCudaSample | None = None
        self._pending_samples: deque[_PendingCudaSample] = deque()

    def on_batch_start(self) -> list[TimingMetric]:
        """Close the previous cycle, enqueue a start marker, and poll results.

        The wall-clock start-to-start interval covers one training cycle. Once
        the corresponding GPU duration is available, the residual is used as
        an estimate of data wait and other non-GPU overhead.
        """
        now_s = self._clock()
        if self._active_sample is not None:
            if self._active_sample.end_event is None:
                raise RuntimeError(
                    "CUDA batch started before the previous batch ended."
                )
            # Promote the previous active sample: its events are already
            # queued on the stream, so it now only waits to be resolved.
            self._pending_samples.append(
                _PendingCudaSample(
                    wall_start_s=self._active_sample.wall_start_s,
                    next_wall_start_s=now_s,
                    start_event=self._active_sample.start_event,
                    end_event=self._active_sample.end_event,
                )
            )

        # Enqueue this marker before polling old events so bookkeeping does not
        # move the GPU boundary. It is timestamped when the stream reaches it.
        start_event = Event(enable_timing=True)  # type: ignore[no-untyped-call]
        start_event.record()  # type: ignore[no-untyped-call]
        self._active_sample = _ActiveCudaSample(
            wall_start_s=now_s, start_event=start_event
        )
        return self._resolve_ready_samples()

    def on_batch_end(self) -> list[TimingMetric]:
        """Enqueue an end marker after the batch's previously queued GPU work."""
        if self._active_sample is None:
            raise RuntimeError("CUDA batch ended before it started.")
        end_event = Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end_event.record()  # type: ignore[no-untyped-call]
        self._active_sample.end_event = end_event
        return []

    def _resolve_ready_samples(self) -> list[TimingMetric]:
        """Pop samples from the front of the deque while their events are ready.

        Only the front of the deque is ever checked: CUDA events on the same
        stream complete in the order they were recorded, so once the oldest
        pending sample is not yet ready, none of the newer ones are either.
        This keeps the poll non-blocking and preserves stream order without
        needing to inspect every pending sample on each call.
        """
        metrics: list[TimingMetric] = []
        while (
            self._pending_samples and self._pending_samples[0].end_event.query()  # type: ignore[no-untyped-call]
        ):
            sample = self._pending_samples.popleft()
            batch_time_s = (
                sample.start_event.elapsed_time(  # type: ignore[no-untyped-call]
                    sample.end_event
                )
                / 1_000
            )
            cycle_time_s = sample.next_wall_start_s - sample.wall_start_s
            metrics.extend(
                [
                    ("batch_time", batch_time_s),
                    ("data_time", max(0.0, cycle_time_s - batch_time_s)),
                ]
            )
        return metrics


def create_batch_timing_tracker(device_type: str) -> BatchTimingTracker:
    """Create the timing strategy appropriate for the module's runtime device."""
    if device_type == "cuda":
        return CudaBatchTimingTracker()
    return CpuBatchTimingTracker()
