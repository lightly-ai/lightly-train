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
    status: RunStatus
    head_id: int | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class UserInfo(BaseModel):
    user_id: str
    task: TaskType
    class_names: list[str]
    num_samples: int
    samples_per_class: dict[str, int]
    head: HeadInfo | None
    latest_run: RunInfo | None


class UploadResponse(BaseModel):
    sample_ids: list[int]
    run_id: int


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


class Annotation(BaseModel):
    """Bounding boxes for one uploaded image, in image pixels."""

    boxes: list[tuple[float, float, float, float]]
    labels: list[str]
