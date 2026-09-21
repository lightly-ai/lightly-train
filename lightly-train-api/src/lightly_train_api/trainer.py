#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import io
import traceback
from datetime import datetime, timezone
from typing import TypeVar

import torch
from sqlmodel import Session, select
from torch import Tensor

from lightly_train._commands import train_api
from lightly_train._commands.train_api import FittedHead, HeadWeights, TrainMetrics
from lightly_train_api import encoder
from lightly_train_api.db import get_engine
from lightly_train_api.models import (
    Head,
    RunStatus,
    Sample,
    TaskType,
    TrainingRun,
    User,
)
from lightly_train_api.settings import get_settings

T = TypeVar("T")

__all__ = [
    "FittedHead",
    "HeadWeights",
    "TrainMetrics",
    "dump_weights",
    "load_weights",
    "retrain_user",
]


def retrain_user(user_id: str, run_id: int) -> TrainMetrics:
    """Retrains the head of a user from scratch on all their samples."""
    with Session(get_engine()) as session:
        run = session.get(TrainingRun, run_id)
        if run is None:
            raise ValueError(f"Unknown training run {run_id}.")
        run.status = RunStatus.RUNNING
        session.add(run)
        session.commit()

        try:
            metrics = _retrain(session=session, user_id=user_id, run=run)
        except Exception:
            session.rollback()
            run = session.get(TrainingRun, run_id)
            assert run is not None
            run.status = RunStatus.FAILED
            run.error = traceback.format_exc()
            run.finished_at = datetime.now(timezone.utc)
            session.add(run)
            session.commit()
            raise
        return metrics


def _retrain(session: Session, user_id: str, run: TrainingRun) -> TrainMetrics:
    user = session.get(User, user_id)
    if user is None:
        raise ValueError(f"Unknown user {user_id}.")

    samples = list(session.exec(select(Sample).where(Sample.user_id == user_id)).all())
    if not samples:
        raise ValueError(f"User {user_id} has no samples.")

    settings = get_settings()
    if user.task is TaskType.DETECTION:
        fitted = _fit_detection(user=user, samples=samples)
        backbone = settings.detection_model_name
    else:
        fitted = _fit_classification(user=user, samples=samples)
        backbone = settings.model_name

    head = Head(
        user_id=user_id,
        task=user.task,
        class_names=list(user.class_names),
        backbone=backbone,
        weights=dump_weights(fitted.weights),
        num_samples=len(samples),
        train_loss=fitted.metrics.train_loss,
        train_accuracy=fitted.metrics.train_accuracy,
    )
    session.add(head)
    session.commit()
    session.refresh(head)

    run.status = RunStatus.SUCCEEDED
    run.head_id = head.id
    run.finished_at = datetime.now(timezone.utc)
    session.add(run)
    session.commit()
    return fitted.metrics


def _fit_classification(user: User, samples: list[Sample]) -> FittedHead:
    settings = get_settings()
    class_to_index = {name: index for index, name in enumerate(user.class_names)}
    features = torch.stack(
        [encoder.blob_to_feature(_require(sample.embedding)) for sample in samples]
    )
    labels = torch.tensor(
        [class_to_index[_require(sample.label)] for sample in samples]
    )
    return train_api.fit_classification_head(
        features=features,
        labels=labels,
        num_classes=len(user.class_names),
        steps=settings.train_steps,
        lr=settings.train_lr,
        weight_decay=settings.train_weight_decay,
        max_seconds=settings.train_max_seconds,
    )


def _fit_detection(user: User, samples: list[Sample]) -> FittedHead:
    settings = get_settings()
    class_names = list(user.class_names)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    model = encoder.get_detector(tuple(class_names))
    size = settings.detection_image_size

    images = torch.stack(
        [
            encoder.blob_to_tensor(_require(sample.tensor), (3, size, size))
            for sample in samples
        ]
    ).to(next(model.parameters()).device)
    targets = []
    for sample in samples:
        annotations = _require(sample.annotations)
        targets.append(
            train_api.boxes_to_target(
                boxes=annotations["boxes"],
                labels=[class_to_index[label] for label in annotations["labels"]],
                # Boxes are stored in the coordinates of the uploaded image.
                size=(annotations["height"], annotations["width"]),
            )
        )

    return train_api.fit_detection_head(
        model=model,
        images=images,
        targets=targets,
        steps=settings.detection_train_steps,
        lr=settings.detection_train_lr,
        weight_decay=settings.detection_train_weight_decay,
        max_seconds=settings.detection_train_max_seconds,
    )


def _require(value: T | None) -> T:
    if value is None:
        raise ValueError("Sample is missing data required by its task.")
    return value


def dump_weights(weights: HeadWeights) -> bytes:
    buffer = io.BytesIO()
    torch.save(weights, buffer)
    return buffer.getvalue()


def load_weights(blob: bytes) -> HeadWeights:
    weights: dict[str, Tensor] = torch.load(
        io.BytesIO(blob), map_location="cpu", weights_only=True
    )
    return weights
