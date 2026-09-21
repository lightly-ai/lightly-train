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

from sqlalchemy import JSON, Column
from sqlalchemy import Enum as SAEnum
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    # Index in this list is the class index the head is trained on.
    class_names: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_now)


class Sample(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    label: str
    image: bytes
    # float32 (feature_dim,) raw bytes.
    embedding: bytes
    backbone: str
    created_at: datetime = Field(default_factory=_now)


class Head(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    class_names: list[str] = Field(sa_column=Column(JSON))
    backbone: str
    weights: bytes
    num_samples: int
    train_loss: float
    train_accuracy: float
    created_at: datetime = Field(default_factory=_now)


class TrainingRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    # Stored as the enum value, not its name.
    status: RunStatus = Field(
        default=RunStatus.QUEUED,
        sa_column=Column(
            SAEnum(RunStatus, values_callable=lambda enum: [m.value for m in enum]),
            nullable=False,
        ),
    )
    head_id: int | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None
