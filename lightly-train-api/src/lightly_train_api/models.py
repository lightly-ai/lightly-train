#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import JSON, Column
from sqlalchemy import Enum as SAEnum
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _enum_column(enum: type[Enum], **kwargs: Any) -> Column:  # type: ignore[type-arg]
    # Store the enum value, not its name.
    return Column(
        SAEnum(enum, values_callable=lambda members: [m.value for m in members]),
        nullable=False,
        **kwargs,
    )


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    task: TaskType = Field(
        default=TaskType.CLASSIFICATION, sa_column=_enum_column(TaskType)
    )
    # Index in this list is the class index the head is trained on.
    class_names: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_now)


class Sample(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    # Set for classification samples.
    label: str | None = None
    image: bytes
    # Classification: float32 (feature_dim,) raw bytes of the pooled features.
    embedding: bytes | None = None
    # Detection: float32 (3, H, W) raw bytes of the preprocessed image.
    tensor: bytes | None = None
    # Detection: {"boxes": [[x1, y1, x2, y2], ...], "labels": [...]} in image pixels.
    annotations: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    backbone: str
    created_at: datetime = Field(default_factory=_now)


class Head(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    task: TaskType = Field(
        default=TaskType.CLASSIFICATION, sa_column=_enum_column(TaskType)
    )
    class_names: list[str] = Field(sa_column=Column(JSON))
    backbone: str
    weights: bytes
    num_samples: int
    train_loss: float
    # Only defined for classification.
    train_accuracy: float | None = None
    created_at: datetime = Field(default_factory=_now)


class TrainingRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    status: RunStatus = Field(
        default=RunStatus.QUEUED, sa_column=_enum_column(RunStatus)
    )
    head_id: int | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
