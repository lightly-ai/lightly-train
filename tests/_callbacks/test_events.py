#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from unittest.mock import MagicMock

from pytest_mock import MockerFixture

from lightly_train._callbacks.events import EventsCallback
from lightly_train._events import tracker
from lightly_train._events.event_info import TrainingEventInfo


def test_on_train_start__rank_0(mocker: MockerFixture) -> None:
    """Test that events are tracked on rank 0."""
    mock_track_event = mocker.patch("lightly_train._events.tracker.track_event")
    tracker._events.clear()
    tracker._last_event_time.clear()

    event_info = TrainingEventInfo(
        method="simclr", model="resnet18", epochs=10, batch_size=64, devices=2
    )
    callback = EventsCallback(event_info=event_info)

    trainer = MagicMock()
    trainer.global_rank = 0
    pl_module = MagicMock()

    callback.on_train_start(trainer=trainer, pl_module=pl_module)

    mock_track_event.assert_called_once()
    assert mock_track_event.call_args[0][0] == "training_started"


def test_on_train_start__rank_1(mocker: MockerFixture) -> None:
    """Test that events are not tracked on rank 1."""
    mock_track_event = mocker.patch("lightly_train._events.tracker.track_event")
    tracker._events.clear()
    tracker._last_event_time.clear()

    event_info = TrainingEventInfo(
        method="simclr", model="resnet18", epochs=10, batch_size=64, devices=2
    )
    callback = EventsCallback(event_info=event_info)

    trainer = MagicMock()
    trainer.global_rank = 1
    pl_module = MagicMock()

    callback.on_train_start(trainer=trainer, pl_module=pl_module)

    mock_track_event.assert_not_called()
