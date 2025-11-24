#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
import platform
import threading
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

import torch

from lightly_train import _distributed
from lightly_train._env import Env

_RATE_LIMIT_SECONDS: float = 30.0
_MAX_QUEUE_SIZE: int = 100

_events: List[Dict[str, Any]] = []
_last_event_time: Dict[str, float] = {}
_last_flush: float = 0.0
_system_info: Optional[Dict[str, Any]] = None
_session_id: str = str(uuid.uuid4())


def _flush() -> None:
    """Flush events from queue with timeout."""
    end_time = time.time() + _RATE_LIMIT_SECONDS
    while _events and time.time() < end_time:
        event_data = _events.pop(0)
        try:
            request = urllib.request.Request(
                "https://eu.i.posthog.com/i/v0/e/",
                data=json.dumps(event_data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(request, timeout=10)
        except Exception:
            pass


def _get_system_info() -> Dict[str, Any]:
    """Collect minimal system info for analytics.

    This lightweight helper avoids the higher overhead of lightly_train._system.get_system_information()
    which triggers heavy metadata/git lookups.
    """
    global _system_info
    # TODO(Igor, 11/25): Check if we can use lightly_train._system.get_system_information() here.
    if _system_info is None:
        gpu_name = None
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        _system_info = {
            "os": platform.system(),
            "gpu_name": gpu_name,
        }
    return _system_info


def track_event(event_name: str, properties: Dict[str, Any]) -> None:
    """Track an event.

    Events are buffered so flushes can send them in batches. This limits how many
    threads we spawn and ensures no events are dropped while a flush runs.
    """
    if not _distributed.is_global_rank_zero():
        return

    global _last_flush

    current_time = time.time()
    if Env.LIGHTLY_TRAIN_EVENTS_DISABLED.value or (
        current_time - _last_event_time.get(event_name, -100.0) < _RATE_LIMIT_SECONDS
    ):
        return

    if len(_events) >= _MAX_QUEUE_SIZE:
        return

    _last_event_time[event_name] = current_time

    event_data = {
        "api_key": Env.LIGHTLY_TRAIN_POSTHOG_KEY.value,
        "event": event_name,
        "distinct_id": _session_id,
        "properties": {**properties, **_get_system_info()},
    }
    _events.append(event_data)

    if current_time - _last_flush >= _RATE_LIMIT_SECONDS:
        _last_flush = current_time
        threading.Thread(target=_flush, daemon=True).start()


def track_training_started(
    *,
    task_type: str,
    model: Any,
    method: str,
    batch_size: int | str,
    devices: int | str | list[int],
    epochs: Optional[int | str] = None,
    steps: Optional[int | str] = None,
) -> None:
    """Track training started event.

    Args:
        task_type: Type of task being trained (e.g., "ssl_pretraining", "instance_segmentation").
        model: Model instance or model name string.
        method: Training method (e.g., "simclr", "eomt", "ltdetr").
        batch_size: Global batch size (can be "auto").
        devices: Number or list of devices (can be "auto").
        epochs: Optional number of epochs (for pretraining tasks, can be "auto").
        steps: Optional number of steps (for task-specific training, can be "auto").
    """
    model_name = model if isinstance(model, str) else model.__class__.__name__
    device_count = devices if isinstance(devices, int) else 1

    properties = {
        "task_type": task_type,
        "model_name": model_name,
        "method": method,
        "batch_size": batch_size,
        "devices": device_count,
    }

    if epochs is not None:
        properties["epochs"] = epochs
    if steps is not None:
        properties["steps"] = steps

    track_event("training_started", properties)
