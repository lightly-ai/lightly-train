#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL.Image import Image
from pydantic import ValidationError
from sqlmodel import Session, col, func, select
from torch.nn import functional as F

from lightly_train_api import encoder, schemas, tasks, trainer
from lightly_train_api.db import get_session, init_db
from lightly_train_api.models import Head, Sample, TaskType, TrainingRun, User
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
        detector=settings.detection_model_name,
        detector_image_size=settings.detection_image_size,
    )


@app.get("/me")
def me(session: SessionDep, user: UserDep) -> schemas.UserInfo:
    counts = session.exec(
        select(Sample.label, func.count())
        .where(Sample.user_id == user.id)
        .group_by(col(Sample.label))
    ).all()
    num_samples = session.exec(
        select(func.count()).select_from(Sample).where(Sample.user_id == user.id)
    ).one()
    head = _latest_head(session=session, user_id=user.id)
    run: TrainingRun | None = session.exec(
        select(TrainingRun)
        .where(TrainingRun.user_id == user.id)
        .order_by(col(TrainingRun.id).desc())
    ).first()
    return schemas.UserInfo(
        user_id=user.id,
        task=user.task,
        class_names=user.class_names,
        num_samples=num_samples,
        samples_per_class={
            label: count for label, count in counts if label is not None
        },
        head=_head_info(head),
        latest_run=_run_info(run),
    )


@app.post("/samples")
async def upload_samples(
    session: SessionDep,
    user: UserDep,
    files: Annotated[list[UploadFile], File()],
    labels: Annotated[list[str] | None, Form()] = None,
    annotations: Annotated[list[str] | None, Form()] = None,
    class_names: Annotated[list[str] | None, Form()] = None,
) -> schemas.UploadResponse:
    if labels and annotations:
        raise HTTPException(400, "Provide either labels or annotations, not both.")
    task = TaskType.DETECTION if annotations else TaskType.CLASSIFICATION
    if user.class_names and user.task is not task:
        raise HTTPException(
            400, f"User {user.id} already has samples for task '{user.task.value}'."
        )

    per_file = annotations if task is TaskType.DETECTION else labels
    if per_file is None or len(files) != len(per_file):
        raise HTTPException(400, "Number of files and annotations must match.")

    images_data = [await file.read() for file in files]
    images = [encoder.decode_image(data) for data in images_data]

    if task is TaskType.DETECTION:
        parsed = [_parse_annotation(value) for value in annotations or []]
        new_names = [label for item in parsed for label in item.labels]
    else:
        parsed = []
        new_names = list(labels or [])

    user.task = task
    user.class_names = _extend_class_names(
        known=user.class_names, new=[*(class_names or []), *new_names]
    )
    session.add(user)

    if task is TaskType.DETECTION:
        samples = _detection_samples(
            user=user, images=images, images_data=images_data, parsed=parsed
        )
    else:
        samples = _classification_samples(
            user=user, images=images, images_data=images_data, labels=labels or []
        )

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
) -> list[schemas.Prediction] | list[schemas.Detection]:
    head = _get_cached_head(session=session, user_id=user.id)
    if head is None:
        raise HTTPException(409, "No trained head available yet.")

    images = [encoder.decode_image(await file.read()) for file in files]
    if user.task is TaskType.DETECTION:
        return _predict_detection(head=head, images=images)
    return _predict_classification(head=head, images=images)


def _predict_classification(
    head: CachedHead, images: Sequence[Image]
) -> list[schemas.Prediction]:
    features = encoder.normalize_features(encoder.encode(images))
    logits = F.linear(features, head.weights["weight"], head.weights["bias"])
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


def _predict_detection(
    head: CachedHead, images: Sequence[Image]
) -> list[schemas.Detection]:
    model = encoder.get_detector(tuple(head.class_names))
    # The detector is shared across users with the same class set, so the head is
    # loaded on every request rather than once at cache fill.
    model.load_state_dict(head.weights, strict=False)
    device = next(model.parameters()).device

    batch = torch.stack(
        [encoder.preprocess_for_detection(image, head.class_names) for image in images]
    ).to(device)
    metadata = [{"orig_h": image.height, "orig_w": image.width} for image in images]
    with torch.inference_mode():
        raw = model(model.preprocessor.preprocess_batch(batch))
    predictions = model.postprocess(
        raw_outputs=raw,
        metadata=metadata,
        threshold=get_settings().detection_predict_threshold,
    )

    return [
        schemas.Detection(
            boxes=[
                schemas.Box(
                    label=head.class_names[int(label)],
                    score=float(score),
                    box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
                for label, score, box in zip(
                    prediction["labels"], prediction["scores"], prediction["bboxes"]
                )
            ]
        )
        for prediction in predictions
    ]


def _parse_annotation(value: str) -> schemas.Annotation:
    try:
        return schemas.Annotation.model_validate_json(value)
    except ValidationError as error:
        raise HTTPException(400, f"Invalid annotation: {error}.")


def _extend_class_names(known: Sequence[str], new: Sequence[str]) -> list[str]:
    """Appends unseen class names. Existing indices must stay stable."""
    names = list(known)
    for name in new:
        if name not in names:
            names.append(name)
    return names


def _classification_samples(
    user: User,
    images: Sequence[Image],
    images_data: Sequence[bytes],
    labels: Sequence[str],
) -> list[Sample]:
    features = encoder.encode(images)
    return [
        Sample(
            user_id=user.id,
            label=label,
            image=data,
            embedding=encoder.feature_to_blob(feature),
            backbone=get_settings().model_name,
        )
        for data, label, feature in zip(images_data, labels, features)
    ]


def _detection_samples(
    user: User,
    images: Sequence[Image],
    images_data: Sequence[bytes],
    parsed: Sequence[schemas.Annotation],
) -> list[Sample]:
    class_names = tuple(user.class_names)
    return [
        Sample(
            user_id=user.id,
            image=data,
            tensor=encoder.tensor_to_blob(
                encoder.preprocess_for_detection(image, class_names)
            ),
            annotations={
                "boxes": [list(box) for box in annotation.boxes],
                "labels": list(annotation.labels),
                "height": image.height,
                "width": image.width,
            },
            backbone=get_settings().detection_model_name,
        )
        for data, image, annotation in zip(images_data, images, parsed)
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
        task=head.task,
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
