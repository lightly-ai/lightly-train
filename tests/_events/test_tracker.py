#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os

import pytest
from pytest_mock import MockerFixture

from lightly_train._events import tracker


@pytest.fixture
def mock_events_disabled(mocker: MockerFixture) -> None:
    """Mock events as disabled."""
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_EVENTS_DISABLED": "1"})


@pytest.fixture
def mock_events_enabled(mocker: MockerFixture) -> None:
    """Mock events as enabled and prevent background threads."""
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_EVENTS_DISABLED": "0"})
    mocker.patch("threading.Thread")


@pytest.fixture(autouse=True)
def clear_tracker_state() -> None:
    """Clear tracker state before each test."""
    tracker._events.clear()
    tracker._last_event_time.clear()
    tracker._system_info = None


def test_track_event__success(mock_events_enabled: None) -> None:
    """Test that events are tracked successfully."""
    tracker.track_event(event_name="test_event", properties={"key": "value"})

    assert len(tracker._events) == 1
    assert tracker._events[0]["event"] == "test_event"
    assert tracker._events[0]["properties"]["key"] == "value"


def test_track_event__structure(
    mock_events_enabled: None, mocker: MockerFixture
) -> None:
    """Test that tracked events contain all required fields."""
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_POSTHOG_KEY": "test_key"})

    tracker.track_event(event_name="test_event", properties={"prop1": "value1"})

    assert len(tracker._events) == 1
    event_data = tracker._events[0]
    assert event_data["api_key"] == "test_key"
    assert event_data["event"] == "test_event"
    assert event_data["distinct_id"] == tracker._session_id
    assert "prop1" in event_data["properties"]
    assert event_data["properties"]["prop1"] == "value1"
    assert "os" in event_data["properties"]


def test_track_event__rate_limited(
    mock_events_enabled: None, mocker: MockerFixture
) -> None:
    """Test that duplicate events within 30 seconds are rate limited."""
    mock_time = mocker.patch("lightly_train._events.tracker.time.time")

    mock_time.return_value = 0.0
    tracker.track_event(event_name="test_event", properties={"key": "value1"})

    mock_time.return_value = 10.0
    tracker.track_event(event_name="test_event", properties={"key": "value2"})

    mock_time.return_value = 31.0
    tracker.track_event(event_name="test_event", properties={"key": "value3"})

    assert len(tracker._events) == 2
    assert tracker._events[0]["properties"]["key"] == "value1"
    assert tracker._events[1]["properties"]["key"] == "value3"


def test_track_event__disabled(mock_events_disabled: None) -> None:
    """Test that events are not tracked when tracking is disabled."""
    tracker.track_event(event_name="test_event", properties={"key": "value"})

    assert len(tracker._events) == 0
    assert "test_event" not in tracker._last_event_time


def test_track_event__queue_size_limit(
    mock_events_enabled: None, mocker: MockerFixture
) -> None:
    """Test that queue drops new events when maximum size is reached."""
    mock_time = mocker.patch("lightly_train._events.tracker.time.time")

    for i in range(tracker._MAX_QUEUE_SIZE):
        mock_time.return_value = float(i * 100)
        tracker.track_event(event_name=f"event_{i}", properties={"index": i})

    assert len(tracker._events) == tracker._MAX_QUEUE_SIZE

    mock_time.return_value = float(tracker._MAX_QUEUE_SIZE * 100)
    tracker.track_event(
        event_name=f"event_{tracker._MAX_QUEUE_SIZE}",
        properties={"index": tracker._MAX_QUEUE_SIZE},
    )

    assert len(tracker._events) == tracker._MAX_QUEUE_SIZE


def test__get_system_info__structure() -> None:
    """Test that system info contains required fields."""
    info = tracker._get_system_info()

    assert "os" in info
    assert "gpu_name" in info
    assert isinstance(info["os"], str)


def test__get_system_info__cached(mocker: MockerFixture) -> None:
    """Test that system info is cached after first call."""
    mock_cuda = mocker.patch("torch.cuda.is_available", return_value=False)

    info1 = tracker._get_system_info()
    info2 = tracker._get_system_info()

    assert info1 is info2
    assert mock_cuda.call_count == 1


def test_session_id_consistent() -> None:
    """Test that session ID remains consistent across calls."""
    session_id = tracker._session_id

    assert isinstance(session_id, str)
    assert len(session_id) > 0
