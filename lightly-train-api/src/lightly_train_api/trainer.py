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
import time
import traceback
from datetime import datetime, timezone

import torch
from sqlmodel import Session, select
from torch import Tensor
from torch.nn import Linear
from torch.nn import functional as F

from lightly_train_api import encoder
from lightly_train_api.db import get_engine
from lightly_train_api.models import Head, Sample, TrainingRun, User
from lightly_train_api.settings import get_settings


def fit_linear_head(
    features: Tensor, labels: Tensor, num_classes: int
) -> tuple[dict[str, Tensor], dict[str, float]]:
    """Fits a randomly initialized linear head on all features. Full batch."""
    settings = get_settings()
    features = encoder.normalize_features(features)

    head = Linear(features.shape[-1], num_classes)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=settings.train_lr,
        weight_decay=settings.train_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.train_steps
    )

    deadline = time.monotonic() + settings.train_max_seconds
    loss = torch.zeros(())
    for _ in range(settings.train_steps):
        optimizer.zero_grad()
        logits = head(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        scheduler.step()
        if time.monotonic() > deadline:
            break

    with torch.no_grad():
        logits = head(features)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()

    state_dict = {"weight": head.weight.detach(), "bias": head.bias.detach()}
    metrics = {"train_loss": float(loss.detach()), "train_accuracy": float(accuracy)}
    return state_dict, metrics


def retrain_user(user_id: str, run_id: int) -> dict[str, float]:
    """Retrains the head of a user from scratch on all their samples."""
    with Session(get_engine()) as session:
        run = session.get(TrainingRun, run_id)
        if run is None:
            raise ValueError(f"Unknown training run {run_id}.")
        run.status = "running"
        session.add(run)
        session.commit()

        try:
            metrics = _retrain(session=session, user_id=user_id, run=run)
        except Exception:
            session.rollback()
            run = session.get(TrainingRun, run_id)
            assert run is not None
            run.status = "failed"
            run.error = traceback.format_exc()
            run.finished_at = datetime.now(timezone.utc)
            session.add(run)
            session.commit()
            raise
        return metrics


def _retrain(session: Session, user_id: str, run: TrainingRun) -> dict[str, float]:
    user = session.get(User, user_id)
    if user is None:
        raise ValueError(f"Unknown user {user_id}.")

    class_names: list[str] = json.loads(user.class_names)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    samples = list(session.exec(select(Sample).where(Sample.user_id == user_id)).all())
    if not samples:
        raise ValueError(f"User {user_id} has no samples.")

    features = torch.stack(
        [encoder.blob_to_feature(sample.embedding) for sample in samples]
    )
    labels = torch.tensor([class_to_index[sample.label] for sample in samples])
    state_dict, metrics = fit_linear_head(
        features=features, labels=labels, num_classes=len(class_names)
    )

    head = Head(
        user_id=user_id,
        class_names=user.class_names,
        backbone=get_settings().model_name,
        weights=dump_weights(state_dict),
        num_samples=len(samples),
        train_loss=metrics["train_loss"],
        train_accuracy=metrics["train_accuracy"],
    )
    session.add(head)
    session.commit()
    session.refresh(head)

    run.status = "succeeded"
    run.head_id = head.id
    run.finished_at = datetime.now(timezone.utc)
    session.add(run)
    session.commit()
    return metrics


def dump_weights(state_dict: dict[str, Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def load_weights(weights: bytes) -> dict[str, Tensor]:
    loaded: dict[str, Tensor] = torch.load(
        io.BytesIO(weights), map_location="cpu", weights_only=True
    )
    return loaded
