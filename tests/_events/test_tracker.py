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
    mocker.patch("lightly_train._distributed.is_global_rank_zero", return_value=True)


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


def test__get_device_count__int() -> None:
    """Test that int devices returns the int directly."""
    assert tracker._get_device_count(4) == 4


def test__get_device_count__list_and_string() -> None:
    """Test that list returns length and string returns 1."""
    assert tracker._get_device_count([0, 1, 2]) == 3
    assert tracker._get_device_count("auto") == 1


def test_track_training_started__success(mock_events_enabled: None) -> None:
    """Test that training started events are tracked successfully."""
    tracker.track_training_started(
        task_type="ssl_pretraining",
        model="resnet50",
        method="simclr",
        batch_size=32,
        devices=2,
        epochs=100,
    )

    assert len(tracker._events) == 1
    assert tracker._events[0]["event"] == "training_started"
    props = tracker._events[0]["properties"]
    assert props["task_type"] == "ssl_pretraining"
    assert props["model_name"] == "resnet50"
    assert props["method"] == "simclr"
    assert props["batch_size"] == 32
    assert props["devices"] == 2
    assert props["epochs"] == 100


def test_track_training_started__with_model_instance(
    mock_events_enabled: None,
) -> None:
    """Test that training started events extract model name from instance."""

    class MyModel:
        pass

    tracker.track_training_started(
        task_type="object_detection",
        model=MyModel(),
        method="ltdetr",
        batch_size="auto",
        devices=[0, 1],
        steps=1000,
    )

    assert len(tracker._events) == 1
    props = tracker._events[0]["properties"]
    assert props["model_name"] == "MyModel"
    assert props["devices"] == 2  # len([0, 1]) = 2


def test_track_inference_started__success(mock_events_enabled: None) -> None:
    """Test that inference started events are tracked successfully."""
    tracker.track_inference_started(
        task_type="object_detection",
        model="DINOv3LTDETRObjectDetection",
        batch_size=16,
        devices=1,
    )

    assert len(tracker._events) == 1
    assert tracker._events[0]["event"] == "inference_started"
    props = tracker._events[0]["properties"]
    assert props["task_type"] == "object_detection"
    assert props["model_name"] == "DINOv3LTDETRObjectDetection"
    assert props["batch_size"] == 16
    assert props["devices"] == 1


def test_track_inference_started__with_model_instance(
    mock_events_enabled: None,
) -> None:
    """Test that inference started events extract model name from instance."""

    class DINOv3EoMTSemanticSegmentation:
        pass

    tracker.track_inference_started(
        task_type="semantic_segmentation",
        model=DINOv3EoMTSemanticSegmentation(),
    )

    assert len(tracker._events) == 1
    props = tracker._events[0]["properties"]
    assert props["model_name"] == "DINOv3EoMTSemanticSegmentation"
    assert props["devices"] == 1  # default
    assert "batch_size" not in props  # not provided


def test_track_inference_started__without_batch_size(
    mock_events_enabled: None,
) -> None:
    """Test that inference started events work without optional batch_size."""
    tracker.track_inference_started(
        task_type="embedding",
        model="EmbeddingModel",
    )

    assert len(tracker._events) == 1
    props = tracker._events[0]["properties"]
    assert props["task_type"] == "embedding"
    assert props["model_name"] == "EmbeddingModel"
    assert "batch_size" not in props
