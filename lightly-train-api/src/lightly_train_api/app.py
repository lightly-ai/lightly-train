#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from sqlmodel import Session, col, func, select
from torch.nn import functional as F

from lightly_train_api import encoder, schemas, tasks, trainer
from lightly_train_api.db import get_session, init_db
from lightly_train_api.models import Head, Sample, TrainingRun, User
from lightly_train_api.settings import get_settings


@dataclass(frozen=True)
class CachedHead:
    """Head weights kept in memory to avoid reloading them on every prediction."""

    head_id: int
    weights: trainer.HeadWeights
    class_names: list[str]


_head_cache: dict[str, CachedHead] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    encoder.get_encoder()
    if get_settings().use_hatchet:
        # Worker.start() creates its own event loop and blocks, so it needs a thread.
        worker = tasks.create_worker()
        threading.Thread(target=worker.start, daemon=True).start()
    yield


app = FastAPI(title="lightly-train-api", lifespan=lifespan)

SessionDep = Annotated[Session, Depends(get_session)]


def get_current_user(session: SessionDep, x_user_id: Annotated[str, Header()]) -> User:
    user = session.get(User, x_user_id)
    if user is None:
        user = User(id=x_user_id)
        session.add(user)
        session.commit()
        session.refresh(user)
    return user


UserDep = Annotated[User, Depends(get_current_user)]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model")
def model_info() -> schemas.ModelInfo:
    settings = get_settings()
    return schemas.ModelInfo(
        backbone=settings.model_name,
        feature_dim=encoder.feature_dim(),
        image_size=settings.image_size,
    )


@app.get("/me")
def me(session: SessionDep, user: UserDep) -> schemas.UserInfo:
    counts = session.exec(
        select(Sample.label, func.count())
        .where(Sample.user_id == user.id)
        .group_by(col(Sample.label))
    ).all()
    head = _latest_head(session=session, user_id=user.id)
    run: TrainingRun | None = session.exec(
        select(TrainingRun)
        .where(TrainingRun.user_id == user.id)
        .order_by(col(TrainingRun.id).desc())
    ).first()
    return schemas.UserInfo(
        user_id=user.id,
        class_names=user.class_names,
        num_samples=sum(count for _, count in counts),
        samples_per_class={label: count for label, count in counts},
        head=_head_info(head),
        latest_run=_run_info(run),
    )


@app.post("/samples")
async def upload_samples(
    session: SessionDep,
    user: UserDep,
    files: Annotated[list[UploadFile], File()],
    labels: Annotated[list[str], Form()],
    class_names: Annotated[list[str] | None, Form()] = None,
) -> schemas.UploadResponse:
    if len(files) != len(labels):
        raise HTTPException(400, "Number of files and labels must match.")

    known = list(user.class_names)
    for name in [*(class_names or []), *labels]:
        if name not in known:
            known.append(name)
    user.class_names = known
    session.add(user)

    images_data = [await file.read() for file in files]
    images = [encoder.decode_image(data) for data in images_data]
    features = encoder.encode(images)

    samples = [
        Sample(
            user_id=user.id,
            label=label,
            image=data,
            embedding=encoder.feature_to_blob(feature),
            backbone=get_settings().model_name,
        )
        for data, label, feature in zip(images_data, labels, features)
    ]
    run = TrainingRun(user_id=user.id)
    session.add_all([*samples, run])
    session.commit()

    for sample in samples:
        session.refresh(sample)
    session.refresh(run)
    assert run.id is not None

    await tasks.enqueue_retrain(user_id=user.id, run_id=run.id)
    return schemas.UploadResponse(
        sample_ids=[sample.id for sample in samples if sample.id is not None],
        run_id=run.id,
    )


@app.get("/runs/{run_id}")
def get_run(session: SessionDep, user: UserDep, run_id: int) -> schemas.RunInfo:
    run = session.get(TrainingRun, run_id)
    if run is None or run.user_id != user.id:
        raise HTTPException(404, f"Unknown run {run_id}.")
    info = _run_info(run)
    assert info is not None
    return info


@app.post("/predict")
async def predict(
    session: SessionDep,
    user: UserDep,
    files: Annotated[list[UploadFile], File()],
) -> list[schemas.Prediction]:
    head = _get_cached_head(session=session, user_id=user.id)
    if head is None:
        raise HTTPException(409, "No trained head available yet.")

    images = [encoder.decode_image(await file.read()) for file in files]
    features = encoder.normalize_features(encoder.encode(images))
    logits = F.linear(features, head.weights.weight, head.weights.bias)
    probabilities = F.softmax(logits, dim=-1)
    scores, indices = probabilities.max(dim=-1)

    return [
        schemas.Prediction(
            label=head.class_names[index],
            score=float(score),
            probabilities=dict(zip(head.class_names, row.tolist())),
        )
        for index, score, row in zip(indices, scores, probabilities)
    ]


def _latest_head(session: Session, user_id: str) -> Head | None:
    head: Head | None = session.exec(
        select(Head).where(Head.user_id == user_id).order_by(col(Head.id).desc())
    ).first()
    return head


def _get_cached_head(session: Session, user_id: str) -> CachedHead | None:
    head = _latest_head(session=session, user_id=user_id)
    if head is None or head.id is None:
        return None
    cached = _head_cache.get(user_id)
    if cached is None or cached.head_id != head.id:
        cached = CachedHead(
            head_id=head.id,
            weights=trainer.load_weights(head.weights),
            class_names=list(head.class_names),
        )
        _head_cache[user_id] = cached
    return cached


def _head_info(head: Head | None) -> schemas.HeadInfo | None:
    if head is None or head.id is None:
        return None
    return schemas.HeadInfo(
        id=head.id,
        class_names=head.class_names,
        num_samples=head.num_samples,
        train_loss=head.train_loss,
        train_accuracy=head.train_accuracy,
        created_at=head.created_at,
    )


def _run_info(run: TrainingRun | None) -> schemas.RunInfo | None:
    if run is None or run.id is None:
        return None
    return schemas.RunInfo(
        id=run.id,
        status=run.status,
        head_id=run.head_id,
        error=run.error,
        created_at=run.created_at,
        finished_at=run.finished_at,
    )
