#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
import uuid
from pathlib import Path

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
    tracker._user_id = None


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
    assert event_data["distinct_id"] == tracker._user_id
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


def test_track_event__disabled_does_not_load_user_id(
    mock_events_disabled: None, mocker: MockerFixture
) -> None:
    """Test that _load_user_id is not called when events are disabled.

    Users who opt out of analytics should not have userid.txt created or read.
    """
    mock_load = mocker.patch("lightly_train._events.tracker._load_user_id")
    mocker.patch("lightly_train._distributed.is_global_rank_zero", return_value=True)

    tracker.track_event(event_name="test_event", properties={"key": "value"})

    mock_load.assert_not_called()
    assert tracker._user_id is None


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
    assert "is_ci" in info
    assert "is_container" in info
    assert isinstance(info["os"], str)


def test__get_system_info__cached(mocker: MockerFixture) -> None:
    """Test that system info is cached after first call."""
    mock_cuda = mocker.patch("torch.cuda.is_available", return_value=False)

    info1 = tracker._get_system_info()
    info2 = tracker._get_system_info()

    assert info1 is info2
    assert mock_cuda.call_count == 1


def test__load_user_id__consistent() -> None:
    """Test that user ID remains consistent across calls."""
    first = tracker._load_user_id()
    second = tracker._load_user_id()
    assert first == second
    assert isinstance(first, str)
    assert len(first) > 0


def test__load_user_id__creates_file(lightly_train_cache_dir: Path) -> None:
    """Test that _load_user_id creates userid.txt when it does not exist."""
    userid_path = lightly_train_cache_dir / "userid.txt"
    assert not userid_path.exists()

    user_id = tracker._load_user_id()

    assert userid_path.exists()
    assert userid_path.read_text(encoding="utf-8").strip() == user_id
    assert isinstance(user_id, str)
    assert len(user_id) > 0


def test__load_user_id__reads_existing_file(lightly_train_cache_dir: Path) -> None:
    """Test that _load_user_id reads an existing userid.txt without creating a new one."""
    userid_path = lightly_train_cache_dir / "userid.txt"
    expected_id = str(uuid.uuid4())
    userid_path.write_text(expected_id, encoding="utf-8")

    user_id = tracker._load_user_id()

    assert user_id == expected_id


def test__load_user_id__empty_file_regenerates(lightly_train_cache_dir: Path) -> None:
    """Test that _load_user_id regenerates a UUID when userid.txt is empty or whitespace."""
    userid_path = lightly_train_cache_dir / "userid.txt"
    userid_path.write_text("   \n", encoding="utf-8")

    user_id = tracker._load_user_id()

    uuid.UUID(user_id)  # Raises ValueError if not a valid UUID.
    # The file should be overwritten with the new valid UUID.
    assert userid_path.read_text(encoding="utf-8").strip() == user_id


def test__load_user_id__invalid_content_regenerates(
    lightly_train_cache_dir: Path,
) -> None:
    """Test that _load_user_id regenerates a UUID when userid.txt contains invalid content."""
    userid_path = lightly_train_cache_dir / "userid.txt"
    userid_path.write_text("not-a-uuid", encoding="utf-8")

    user_id = tracker._load_user_id()

    uuid.UUID(user_id)  # Raises ValueError if not a valid UUID.
    # The file should be overwritten with the new valid UUID.
    assert userid_path.read_text(encoding="utf-8").strip() == user_id


def test__load_user_id__read_error_fallback(
    lightly_train_cache_dir: Path, mocker: MockerFixture
) -> None:
    """Test that _load_user_id falls back to a UUID when reading raises an unexpected exception."""
    userid_path = lightly_train_cache_dir / "userid.txt"
    userid_path.write_text("some-id", encoding="utf-8")
    mocker.patch("pathlib.Path.read_text", side_effect=OSError("Permission denied"))

    user_id = tracker._load_user_id()

    assert isinstance(user_id, str)
    assert len(user_id) > 0


def test__load_user_id__write_error_fallback(
    lightly_train_cache_dir: Path, mocker: MockerFixture
) -> None:
    """Test that _load_user_id returns a UUID if writing the file fails."""
    mocker.patch(
        "lightly_train._events.tracker.os.replace",
        side_effect=OSError("Permission denied"),
    )

    user_id = tracker._load_user_id()

    assert isinstance(user_id, str)
    assert len(user_id) > 0


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


def test__is_ci__true(mocker: MockerFixture) -> None:
    """Test that _is_ci returns True when CI environment variable is set."""
    mocker.patch.dict(os.environ, {"CI": "true"})

    assert tracker._is_ci() is True


def test__is_ci__empty_string(mocker: MockerFixture) -> None:
    """Test that _is_ci returns True when CI is set to empty string."""
    mocker.patch.dict(os.environ, {"CI": ""})

    assert tracker._is_ci() is True


def test__is_ci__false(mocker: MockerFixture) -> None:
    """Test that _is_ci returns False when CI environment variable is not set."""
    mocker.patch.dict(os.environ, {}, clear=True)

    assert tracker._is_ci() is False


def test__is_container__dockerenv(mocker: MockerFixture) -> None:
    """Test that _is_container returns True when /.dockerenv exists."""
    mocker.patch("os.path.isfile", side_effect=lambda path: path == "/.dockerenv")

    assert tracker._is_container() is True


def test__is_container__containerenv(mocker: MockerFixture) -> None:
    """Test that _is_container returns True when /run/.containerenv exists."""
    mocker.patch(
        "os.path.isfile",
        side_effect=lambda path: path == "/run/.containerenv",
    )

    assert tracker._is_container() is True


def test__is_container__singularity(mocker: MockerFixture) -> None:
    """Test that _is_container returns True when SINGULARITY_CONTAINER is set."""
    mocker.patch("os.path.isfile", return_value=False)
    mocker.patch.dict(os.environ, {"SINGULARITY_CONTAINER": "1"}, clear=True)

    assert tracker._is_container() is True


def test__is_container__apptainer(mocker: MockerFixture) -> None:
    """Test that _is_container returns True when APPTAINER_CONTAINER is set."""
    mocker.patch("os.path.isfile", return_value=False)
    mocker.patch.dict(os.environ, {"APPTAINER_CONTAINER": "1"}, clear=True)

    assert tracker._is_container() is True


def test__is_container__cgroup_docker(mocker: MockerFixture) -> None:
    """Test that _is_container returns True when /proc/1/cgroup contains docker."""
    mocker.patch("os.path.isfile", return_value=False)
    mocker.patch.dict(os.environ, {}, clear=True)
    mock_open = mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data="9:cpuset:/docker/abc123def456\n"),
    )

    assert tracker._is_container() is True
    mock_open.assert_called_with("/proc/self/cgroup", encoding="utf-8")


def test__is_container__cgroup_kubepods(mocker: MockerFixture) -> None:
    """Test that _is_container returns True when /proc/self/cgroup contains kubepods."""
    mocker.patch("os.path.isfile", return_value=False)
    mocker.patch.dict(os.environ, {}, clear=True)
    mock_open = mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data="9:cpuset:/kubepods/abc12345\n"),
    )

    assert tracker._is_container() is True
    mock_open.assert_called_with("/proc/self/cgroup", encoding="utf-8")


def test__is_container__cgroup_containerd(mocker: MockerFixture) -> None:
    """Test that _is_container returns True when /proc/self/cgroup contains containerd."""
    mocker.patch("os.path.isfile", return_value=False)
    mocker.patch.dict(os.environ, {}, clear=True)
    mock_open = mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data="9:cpuset:/containerd/abc12345def\n"),
    )

    assert tracker._is_container() is True
    mock_open.assert_called_with("/proc/self/cgroup", encoding="utf-8")


def test__is_container__false(mocker: MockerFixture) -> None:
    """Test that _is_container returns False when no container is detected."""
    mocker.patch("os.path.isfile", return_value=False)
    mocker.patch.dict(os.environ, {}, clear=True)
    mock_open = mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data="9:cpuset:/system.slice/user.slice\n"),
    )

    assert tracker._is_container() is False
    mock_open.assert_called_with("/proc/self/cgroup", encoding="utf-8")


def test__is_container__cgroup_file_not_found(mocker: MockerFixture) -> None:
    """Test that _is_container returns False when /proc/self/cgroup doesn't exist."""
    mocker.patch("os.path.isfile", return_value=False)
    mocker.patch.dict(os.environ, {}, clear=True)
    mocker.patch("builtins.open", side_effect=FileNotFoundError)

    assert tracker._is_container() is False


def test__is_container__cgroup_permission_error(mocker: MockerFixture) -> None:
    """Test that _is_container returns False when /proc/self/cgroup is unreadable."""
    mocker.patch("os.path.isfile", return_value=False)
    mocker.patch.dict(os.environ, {}, clear=True)
    mocker.patch("builtins.open", side_effect=PermissionError)

    assert tracker._is_container() is False
