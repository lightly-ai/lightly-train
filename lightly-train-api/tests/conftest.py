#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Tiny architecture-only detector. Keeps tests offline and fast; predictions are
# meaningless, which the detection tests never rely on.
TEST_DETECTOR = "dinov2/_vittest14-ltdetrv2"
TEST_DETECTOR_IMAGE_SIZE = 224


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    from lightly_train._commands import train_api
    from lightly_train_api import db, encoder, settings

    monkeypatch.setenv("LIGHTLY_TRAIN_API_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("LIGHTLY_TRAIN_API_USE_HATCHET", "0")
    monkeypatch.setenv("LIGHTLY_TRAIN_API_DEVICE", "cpu")
    monkeypatch.setenv("LIGHTLY_TRAIN_API_DETECTION_MODEL_NAME", TEST_DETECTOR)
    monkeypatch.setenv(
        "LIGHTLY_TRAIN_API_DETECTION_IMAGE_SIZE", str(TEST_DETECTOR_IMAGE_SIZE)
    )
    monkeypatch.setenv("LIGHTLY_TRAIN_API_DETECTION_TRAIN_STEPS", "2")
    settings.get_settings.cache_clear()
    encoder.get_detector.cache_clear()
    train_api._load_checkpoint.cache_clear()
    db.reset_engine()
    yield
    settings.get_settings.cache_clear()
    encoder.get_detector.cache_clear()
    train_api._load_checkpoint.cache_clear()
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


@pytest.fixture
def annotation() -> Callable[[list[str]], str]:
    """Returns a JSON annotation with one box per label."""

    def make(labels: list[str]) -> str:
        boxes = [[5.0 + 10 * i, 5.0, 25.0 + 10 * i, 30.0] for i in range(len(labels))]
        return json.dumps({"boxes": boxes, "labels": labels})

    return make
