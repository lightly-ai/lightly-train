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


class ModelInfo(BaseModel):
    backbone: str
    feature_dim: int
    image_size: int


class HeadInfo(BaseModel):
    id: int
    class_names: list[str]
    num_samples: int
    train_loss: float
    train_accuracy: float
    created_at: datetime


class RunInfo(BaseModel):
    id: int
    status: str
    head_id: int | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class UserInfo(BaseModel):
    user_id: str
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
