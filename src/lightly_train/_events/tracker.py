#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
import logging
import os
import platform
import threading
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

import torch

from lightly_train import _distributed
from lightly_train._env import Env

logger = logging.getLogger(__name__)

_RATE_LIMIT_SECONDS: float = 30.0
_MAX_QUEUE_SIZE: int = 100

_events: List[Dict[str, Any]] = []
_last_event_time: Dict[str, float] = {}
_last_flush: float = 0.0
_system_info: Optional[Dict[str, Any]] = None
_user_id: str | None = None


def _load_user_id() -> str:
    """Load or create a persistent user ID from the cache directory.

    Returns:
        A UUID string identifying this installation.
    """
    userid_path = Env.LIGHTLY_TRAIN_CACHE_DIR.value / "userid.txt"
    try:
        return userid_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        pass
    except Exception:
        logger.debug("Failed to read userid.txt, falling back to temporary user ID.")
        return str(uuid.uuid4())

    # File does not exist. Generate a new UUID and write it atomically.
    new_id = str(uuid.uuid4())
    tmp_path = userid_path.parent / new_id
    try:
        userid_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(new_id, encoding="utf-8")
        os.replace(tmp_path, userid_path)
    except Exception:
        logger.debug("Failed to write userid.txt, falling back to temporary user ID.")
        # Clean up temp file if it was created.
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return new_id

    # Read back to get the winner in case of a race condition.
    try:
        return userid_path.read_text(encoding="utf-8").strip()
    except Exception:
        return new_id


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


def _get_model_name(model: object) -> str:
    """Extract model name from model instance or string.

    Checks for model_name attribute first, then falls back to class name.
    Works with any object: str, torch.nn.Module, ModelWrapper, etc.
    """
    if isinstance(model, str):
        return model
    return getattr(model, "model_name", model.__class__.__name__)


def _get_device_count(devices: int | str | list[int]) -> int:
    """Extract device count from devices argument.

    Args:
        devices: Number of devices as int, list of device IDs, or "auto".

    Returns:
        The number of devices. Returns 1 for "auto" or other strings.
    """
    if isinstance(devices, int):
        return devices
    elif isinstance(devices, list):
        return len(devices)
    return 1


def track_event(event_name: str, properties: Dict[str, Any]) -> None:
    """Track an event.

    Events are buffered so flushes can send them in batches. This limits how many
    threads we spawn and ensures no events are dropped while a flush runs.
    """
    if not _distributed.is_global_rank_zero():
        return

    global _last_flush, _user_id
    if _user_id is None:
        _user_id = _load_user_id()

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
        "distinct_id": _user_id,
        "properties": {**properties, **_get_system_info()},
    }
    _events.append(event_data)

    if current_time - _last_flush >= _RATE_LIMIT_SECONDS:
        _last_flush = current_time
        threading.Thread(target=_flush, daemon=True).start()


def track_training_started(
    *,
    task_type: str,
    model: object,
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
    properties = {
        "task_type": task_type,
        "model_name": _get_model_name(model),
        "method": method,
        "batch_size": batch_size,
        "devices": _get_device_count(devices),
    }

    if epochs is not None:
        properties["epochs"] = epochs
    if steps is not None:
        properties["steps"] = steps

    track_event("training_started", properties)


def track_inference_started(
    *,
    task_type: str,
    model: object,
    batch_size: Optional[int] = None,
    devices: int | str | list[int] = 1,
) -> None:
    """Track inference started event.

    Args:
        task_type: Type of task being inferred (e.g., "embedding", "object_detection",
            "semantic_segmentation", "instance_segmentation", "panoptic_segmentation").
        model: Model instance or model name string.
        batch_size: Optional batch size.
        devices: Number or list of devices (can be "auto").
    """
    properties = {
        "task_type": task_type,
        "model_name": _get_model_name(model),
        "devices": _get_device_count(devices),
    }

    if batch_size is not None:
        properties["batch_size"] = batch_size

    track_event("inference_started", properties)
