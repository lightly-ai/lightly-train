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
    mocker.patch("lightly_train._events.config.EVENTS_DISABLED", True)


@pytest.fixture
def mock_events_enabled(mocker: MockerFixture) -> None:
    """Mock events as enabled and prevent background threads."""
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_EVENTS_DISABLED": "0"})
    mocker.patch("lightly_train._events.config.EVENTS_DISABLED", False)
    mocker.patch("threading.Thread")


@pytest.fixture(autouse=True)
def clear_tracker_state() -> None:
    """Clear tracker state before each test."""
    tracker._events.clear()
    tracker._last_event_time.clear()
    tracker._system_info = None


def test_track_event__disabled(mock_events_disabled: None) -> None:
    """Test that events are not tracked when disabled."""
    tracker.track_event(event_name="test_event", properties={"key": "value"})

    assert len(tracker._events) == 0
    assert "test_event" not in tracker._last_event_time


def test_track_event__rate_limited(
    mock_events_enabled: None, mocker: MockerFixture
) -> None:
    """Test that events within 30s are rate limited."""
    mock_time = mocker.patch("lightly_train._events.tracker.time.time")

    # First event at t=0
    mock_time.return_value = 0.0
    tracker.track_event(event_name="test_event", properties={"key": "value1"})

    # Second event at t=10 (within 30s) - should be ignored
    mock_time.return_value = 10.0
    tracker.track_event(event_name="test_event", properties={"key": "value2"})

    # Third event at t=31 (after 30s) - should be tracked
    mock_time.return_value = 31.0
    tracker.track_event(event_name="test_event", properties={"key": "value3"})

    assert len(tracker._events) == 2
    assert tracker._events[0]["properties"]["key"] == "value1"
    assert tracker._events[1]["properties"]["key"] == "value3"


def test_track_event__structure(
    mock_events_enabled: None, mocker: MockerFixture
) -> None:
    """Test that event data has correct structure."""
    mocker.patch("lightly_train._events.config.POSTHOG_API_KEY", "test_key")

    tracker.track_event(event_name="test_event", properties={"prop1": "value1"})

    assert len(tracker._events) == 1
    event_data = tracker._events[0]

    assert event_data["api_key"] == "test_key"
    assert event_data["event"] == "test_event"
    assert event_data["distinct_id"] == tracker._session_id
    assert "prop1" in event_data["properties"]
    assert event_data["properties"]["prop1"] == "value1"
    assert "os" in event_data["properties"]


def test__get_system_info__cached(mocker: MockerFixture) -> None:
    """Test that system info is cached."""
    mock_cuda = mocker.patch("torch.cuda.is_available", return_value=False)

    info1 = tracker._get_system_info()
    info2 = tracker._get_system_info()

    assert info1 is info2
    assert mock_cuda.call_count == 1


def test__get_system_info__structure() -> None:
    """Test that system info has correct structure."""
    info = tracker._get_system_info()

    assert "os" in info
    assert "gpu_name" in info
    assert isinstance(info["os"], str)


def test_session_id_consistent() -> None:
    """Test that session_id is consistent across events."""
    session_id = tracker._session_id
    assert isinstance(session_id, str)
    assert len(session_id) > 0
