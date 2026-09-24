#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from lightly_train_api.models import RunStatus, TaskType


class ModelInfo(BaseModel):
    backbone: str
    feature_dim: int
    image_size: int
    detector: str
    detector_image_size: int


class HeadInfo(BaseModel):
    id: int
    task: TaskType
    class_names: list[str]
    num_samples: int
    train_loss: float
    train_accuracy: float | None
    created_at: datetime


class RunInfo(BaseModel):
    id: int
    dataset: str
    status: RunStatus
    head_id: int | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class DatasetInfo(BaseModel):
    user_id: str
    dataset: str
    task: TaskType
    class_names: list[str]
    num_samples: int
    samples_per_class: dict[str, int]
    head: HeadInfo | None
    latest_run: RunInfo | None


class Annotation(BaseModel):
    """Bounding boxes for one uploaded image, in image pixels."""

    boxes: list[tuple[float, float, float, float]]
    labels: list[str]


class SampleState(BaseModel):
    """Client-side state of one sample, as sent to `/samples/diff`.

    Carries the annotation rather than only the image hash, so that re-labeling an
    image whose bytes did not change is reported as `changed`.
    """

    key: str
    content_hash: str
    label: str | None = None
    annotation: Annotation | None = None


class DiffRequest(BaseModel):
    samples: list[SampleState]


class DiffResponse(BaseModel):
    """Which of the requested samples the server still needs."""

    new: list[str]
    changed: list[str]
    unchanged: list[str]


class IngestResponse(BaseModel):
    ingested: list[str]
    updated: list[str]
    unchanged: list[str]
    num_samples: int
    class_names: list[str]
    # None when nothing changed, because then no training run is started.
    run_id: int | None


class Prediction(BaseModel):
    label: str
    score: float
    probabilities: dict[str, float]


class Box(BaseModel):
    label: str
    score: float
    # Absolute xyxy in the coordinates of the uploaded image.
    box: tuple[float, float, float, float]


class Detection(BaseModel):
    boxes: list[Box]
