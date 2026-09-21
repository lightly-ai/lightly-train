#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    from lightly_train_api import db, settings

    monkeypatch.setenv("LIGHTLY_TRAIN_API_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("LIGHTLY_TRAIN_API_USE_HATCHET", "0")
    monkeypatch.setenv("LIGHTLY_TRAIN_API_DEVICE", "cpu")
    settings.get_settings.cache_clear()
    db.reset_engine()
    yield
    settings.get_settings.cache_clear()
    db.reset_engine()


@pytest.fixture
def client() -> Iterator[TestClient]:
    from lightly_train_api.app import _head_cache, app

    _head_cache.clear()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def image_bytes() -> Callable[[tuple[int, int, int]], bytes]:
    def make(color: tuple[int, int, int]) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), color).save(buffer, format="PNG")
        return buffer.getvalue()

    return make
