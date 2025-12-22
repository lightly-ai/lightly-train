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
from lightly_train._task_models.task_model import TaskModel


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


class MockObjectDetectionModel(TaskModel):
    """Mock task model for testing object detection."""

    model_suffix = "test"

    def __init__(self, *, model_name: str = "dinov3/vits16-ltdetr") -> None:
        # Use locals() like real task models do.
        super().__init__(init_args=locals())
        self.model_name = model_name


class MockSemanticSegmentationModel(TaskModel):
    """Mock semantic segmentation model for testing."""

    model_suffix = "test"

    def __init__(self, *, model_name: str = "dinov3/vits16-eomt") -> None:
        # Use locals() like real task models do.
        super().__init__(init_args=locals())
        self.model_name = model_name


class MockUnknownModel(TaskModel):
    """Mock model with unknown task type for testing."""

    model_suffix = "test"

    def __init__(self) -> None:
        # Use locals() like real task models do.
        super().__init__(init_args=locals())


def test_track_inference__object_detection(mock_events_enabled: None) -> None:
    """Test that _track_inference correctly identifies object detection models."""
    model = MockObjectDetectionModel(model_name="dinov3/vits16-ltdetr-coco")
    model._track_inference()

    assert len(tracker._events) == 1
    props = tracker._events[0]["properties"]
    assert props["task_type"] == "object_detection"
    assert props["model_name"] == "dinov3/vits16-ltdetr-coco"


def test_track_inference__semantic_segmentation(mock_events_enabled: None) -> None:
    """Test that _track_inference correctly identifies semantic segmentation models."""
    model = MockSemanticSegmentationModel(model_name="dinov3/vits16-eomt-ade20k")
    model._track_inference()

    assert len(tracker._events) == 1
    props = tracker._events[0]["properties"]
    assert props["task_type"] == "semantic_segmentation"
    assert props["model_name"] == "dinov3/vits16-eomt-ade20k"


def test_track_inference__unknown_type(mock_events_enabled: None) -> None:
    """Test that _track_inference handles unknown model types gracefully."""
    model = MockUnknownModel()
    model._track_inference()

    assert len(tracker._events) == 1
    props = tracker._events[0]["properties"]
    assert props["task_type"] == "unknown"
    # Falls back to class name when model_name is not set.
    assert props["model_name"] == "MockUnknownModel"


def test_track_inference__never_crashes(
    mock_events_enabled: None, mocker: MockerFixture
) -> None:
    """Test that _track_inference never crashes even when tracking fails."""
    # Mock track_inference_started to raise an exception.
    mocker.patch.object(
        tracker, "track_inference_started", side_effect=Exception("Test error")
    )

    model = MockObjectDetectionModel()

    # This should NOT raise - it should silently catch the error.
    model._track_inference()

    # No events should be recorded since track_inference_started failed
    assert len(tracker._events) == 0
