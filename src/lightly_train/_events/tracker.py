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
from typing import Any, Dict, List, Optional, Set

import torch

from lightly_train._events import config

_events: List[Dict[str, Any]] = []
_sent_event_names: Set[str] = set()
_last_flush: float = 0.0
_system_info: Optional[Dict[str, Any]] = None
_session_id: str = str(uuid.uuid4())


def _flush() -> None:
    """Flush events from queue with 30s timeout."""
    end_time = time.time() + 30.0
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
    """Collect minimal system info."""
    global _system_info
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
    """Track an event."""
    global _last_flush

    if config.EVENTS_DISABLED or event_name in _sent_event_names:
        return

    _sent_event_names.add(event_name)

    event_data = {
        "api_key": config.POSTHOG_API_KEY,
        "event": event_name,
        "distinct_id": _session_id,
        "properties": {**properties, **_get_system_info()},
    }
    _events.append(event_data)

    if time.time() - _last_flush >= 30.0:
        _last_flush = time.time()
        threading.Thread(target=_flush, daemon=True).start()
