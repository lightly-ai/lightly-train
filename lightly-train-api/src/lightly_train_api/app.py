#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import hashlib
import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any

import torch
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    UploadFile,
)
from PIL.Image import Image
from pydantic import ValidationError
from sqlmodel import Session, col, func, select
from torch.nn import functional as F

from lightly_train_api import encoder, schemas, tasks, trainer
from lightly_train_api.db import get_session, init_db
from lightly_train_api.models import (
    Dataset,
    Head,
    RunStatus,
    Sample,
    TaskType,
    TrainingRun,
    User,
)
from lightly_train_api.settings import get_settings


@dataclass(frozen=True)
class CachedHead:
    """Head weights kept in memory to avoid reloading them on every prediction."""

    head_id: int
    weights: trainer.HeadWeights
    class_names: list[str]


_head_cache: dict[int, CachedHead] = {}


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
DatasetName = Annotated[str, Path(min_length=1)]


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


@app.get("/datasets")
def list_datasets(session: SessionDep, user: UserDep) -> list[str]:
    names = session.exec(
        select(Dataset.name)
        .where(Dataset.user_id == user.id)
        .order_by(col(Dataset.name))
    ).all()
    return list(names)


@app.get("/datasets/{dataset_name}")
def get_dataset(
    session: SessionDep, user: UserDep, dataset_name: DatasetName
) -> schemas.DatasetInfo:
    dataset = _require_dataset(session=session, user=user, name=dataset_name)
    assert dataset.id is not None

    counts = session.exec(
        select(Sample.label, func.count())
        .where(Sample.dataset_id == dataset.id)
        .group_by(col(Sample.label))
    ).all()
    num_samples = session.exec(
        select(func.count()).select_from(Sample).where(Sample.dataset_id == dataset.id)
    ).one()
    run: TrainingRun | None = session.exec(
        select(TrainingRun)
        .where(TrainingRun.dataset_id == dataset.id)
        .order_by(col(TrainingRun.id).desc())
    ).first()
    return schemas.DatasetInfo(
        user_id=user.id,
        dataset=dataset.name,
        task=dataset.task,
        class_names=dataset.class_names,
        num_samples=num_samples,
        samples_per_class={
            label: count for label, count in counts if label is not None
        },
        head=_head_info(_latest_head(session=session, dataset_id=dataset.id)),
        latest_run=_run_info(session=session, run=run),
    )


@app.post("/datasets/{dataset_name}/samples/diff")
def diff_samples(
    session: SessionDep,
    user: UserDep,
    dataset_name: DatasetName,
    request: schemas.DiffRequest,
) -> schemas.DiffResponse:
    """Reports which samples the server is missing, so only those are uploaded.

    Applies the same comparison as the ingest endpoint, so a sample reported as
    `unchanged` would be a no-op to upload.
    """
    dataset = session.exec(
        select(Dataset).where(Dataset.user_id == user.id, Dataset.name == dataset_name)
    ).first()
    keys = [sample.key for sample in request.samples]
    known: dict[str, Sample] = {}
    if dataset is not None:
        known = {
            sample.key: sample
            for sample in session.exec(
                select(Sample).where(
                    Sample.dataset_id == dataset.id, col(Sample.key).in_(keys)
                )
            ).all()
        }

    response = schemas.DiffResponse(new=[], changed=[], unchanged=[])
    for state in request.samples:
        sample = known.get(state.key)
        if sample is None:
            response.new.append(state.key)
        elif _is_unchanged(
            sample=sample,
            content_hash=state.content_hash,
            label=state.label,
            annotation=state.annotation,
        ):
            response.unchanged.append(state.key)
        else:
            response.changed.append(state.key)
    return response


@app.post("/datasets/{dataset_name}/samples")
async def upload_samples(
    session: SessionDep,
    user: UserDep,
    dataset_name: DatasetName,
    files: Annotated[list[UploadFile], File()],
    keys: Annotated[list[str] | None, Form()] = None,
    labels: Annotated[list[str] | None, Form()] = None,
    annotations: Annotated[list[str] | None, Form()] = None,
    class_names: Annotated[list[str] | None, Form()] = None,
) -> schemas.IngestResponse:
    if labels and annotations:
        raise HTTPException(400, "Provide either labels or annotations, not both.")
    task = TaskType.DETECTION if annotations else TaskType.CLASSIFICATION

    per_file = annotations if task is TaskType.DETECTION else labels
    if per_file is None or len(files) != len(per_file):
        raise HTTPException(400, "Number of files and annotations must match.")
    sample_keys = _resolve_keys(files=files, keys=keys)

    dataset = _get_or_create_dataset(session=session, user=user, name=dataset_name)
    if dataset.class_names and dataset.task is not task:
        raise HTTPException(
            400,
            f"Dataset '{dataset.name}' already has samples for task "
            f"'{dataset.task.value}'.",
        )
    assert dataset.id is not None

    parsed = (
        [_parse_annotation(value) for value in annotations or []]
        if task is TaskType.DETECTION
        else []
    )
    new_names = (
        [label for item in parsed for label in item.labels]
        if task is TaskType.DETECTION
        else list(labels or [])
    )

    dataset.task = task
    dataset.class_names = _extend_class_names(
        known=dataset.class_names, new=[*(class_names or []), *new_names]
    )
    session.add(dataset)

    images_data = [await file.read() for file in files]
    hashes = [hashlib.sha256(data).hexdigest() for data in images_data]
    existing = {
        sample.key: sample
        for sample in session.exec(
            select(Sample).where(
                Sample.dataset_id == dataset.id, col(Sample.key).in_(sample_keys)
            )
        ).all()
    }

    response = schemas.IngestResponse(
        ingested=[],
        updated=[],
        unchanged=[],
        num_samples=0,
        class_names=list(dataset.class_names),
        run_id=None,
    )
    todo: list[int] = []
    for index, key in enumerate(sample_keys):
        sample = existing.get(key)
        if sample is None:
            response.ingested.append(key)
            todo.append(index)
        elif _is_unchanged(
            sample=sample,
            content_hash=hashes[index],
            label=None if task is TaskType.DETECTION else (labels or [])[index],
            annotation=parsed[index] if task is TaskType.DETECTION else None,
        ):
            response.unchanged.append(key)
        else:
            response.updated.append(key)
            todo.append(index)

    if todo:
        images = [encoder.decode_image(images_data[index]) for index in todo]
        payloads = (
            _detection_payloads(
                dataset=dataset,
                images=images,
                parsed=[parsed[index] for index in todo],
            )
            if task is TaskType.DETECTION
            else _classification_payloads(
                images=images, labels=[(labels or [])[index] for index in todo]
            )
        )
        for index, payload in zip(todo, payloads):
            key = sample_keys[index]
            sample = existing.get(key) or Sample(
                dataset_id=dataset.id, key=key, content_hash="", image=b"", backbone=""
            )
            sample.content_hash = hashes[index]
            sample.image = images_data[index]
            sample.updated_at = datetime.now(timezone.utc)
            for field, value in payload.items():
                setattr(sample, field, value)
            session.add(sample)

    session.commit()
    response.num_samples = session.exec(
        select(func.count()).select_from(Sample).where(Sample.dataset_id == dataset.id)
    ).one()
    response.class_names = list(dataset.class_names)

    if todo:
        run_id = _enqueue_run(session=session, dataset_id=dataset.id)
        response.run_id = run_id
        await tasks.enqueue_retrain(dataset_id=dataset.id, run_id=run_id)
    return response


@app.get("/runs/{run_id}")
def get_run(session: SessionDep, user: UserDep, run_id: int) -> schemas.RunInfo:
    run = session.get(TrainingRun, run_id)
    dataset = None if run is None else session.get(Dataset, run.dataset_id)
    if run is None or dataset is None or dataset.user_id != user.id:
        raise HTTPException(404, f"Unknown run {run_id}.")
    info = _run_info(session=session, run=run)
    assert info is not None
    return info


@app.post("/datasets/{dataset_name}/predict")
async def predict(
    session: SessionDep,
    user: UserDep,
    dataset_name: DatasetName,
    files: Annotated[list[UploadFile], File()],
    threshold: Annotated[float | None, Form()] = None,
) -> list[schemas.Prediction] | list[schemas.Detection]:
    dataset = _require_dataset(session=session, user=user, name=dataset_name)
    assert dataset.id is not None
    head = _get_cached_head(session=session, dataset_id=dataset.id)
    if head is None:
        raise HTTPException(409, "No trained head available yet.")

    images = [encoder.decode_image(await file.read()) for file in files]
    if dataset.task is TaskType.DETECTION:
        return _predict_detection(head=head, images=images, threshold=threshold)
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
    head: CachedHead, images: Sequence[Image], threshold: float | None
) -> list[schemas.Detection]:
    model = encoder.get_detector(tuple(head.class_names))
    # The detector is shared across datasets with the same class set, so the head is
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
        threshold=(
            get_settings().detection_predict_threshold
            if threshold is None
            else threshold
        ),
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


def _resolve_keys(files: Sequence[UploadFile], keys: Sequence[str] | None) -> list[str]:
    """Returns one identity per file, defaulting to the uploaded filename."""
    if keys is None:
        resolved = [file.filename or "" for file in files]
        if not all(resolved):
            raise HTTPException(400, "Every file needs a key or a filename.")
    elif len(keys) != len(files):
        raise HTTPException(400, "Number of files and keys must match.")
    else:
        resolved = list(keys)
    if len(set(resolved)) != len(resolved):
        raise HTTPException(400, "Keys must be unique within one request.")
    return resolved


def _is_unchanged(
    sample: Sample,
    content_hash: str,
    label: str | None,
    annotation: schemas.Annotation | None,
) -> bool:
    """Whether re-ingesting would not change anything about the stored sample."""
    if sample.content_hash != content_hash:
        return False
    if annotation is None:
        return sample.label == label
    stored = sample.annotations or {}
    return stored.get("boxes") == [
        list(box) for box in annotation.boxes
    ] and stored.get("labels") == list(annotation.labels)


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


def _classification_payloads(
    images: Sequence[Image], labels: Sequence[str]
) -> list[dict[str, Any]]:
    features = encoder.encode(images)
    return [
        {
            "label": label,
            "embedding": encoder.feature_to_blob(feature),
            "tensor": None,
            "annotations": None,
            "backbone": get_settings().model_name,
        }
        for label, feature in zip(labels, features)
    ]


def _detection_payloads(
    dataset: Dataset,
    images: Sequence[Image],
    parsed: Sequence[schemas.Annotation],
) -> list[dict[str, Any]]:
    class_names = tuple(dataset.class_names)
    return [
        {
            "label": None,
            "embedding": None,
            "tensor": encoder.tensor_to_blob(
                encoder.preprocess_for_detection(image, class_names)
            ),
            "annotations": {
                "boxes": [list(box) for box in annotation.boxes],
                "labels": list(annotation.labels),
                "height": image.height,
                "width": image.width,
            },
            "backbone": get_settings().detection_model_name,
        }
        for image, annotation in zip(images, parsed)
    ]


def _get_or_create_dataset(session: Session, user: User, name: str) -> Dataset:
    dataset = session.exec(
        select(Dataset).where(Dataset.user_id == user.id, Dataset.name == name)
    ).first()
    if dataset is None:
        dataset = Dataset(user_id=user.id, name=name)
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
    return dataset


def _require_dataset(session: Session, user: User, name: str) -> Dataset:
    dataset = session.exec(
        select(Dataset).where(Dataset.user_id == user.id, Dataset.name == name)
    ).first()
    if dataset is None:
        raise HTTPException(404, f"Unknown dataset '{name}'.")
    return dataset


def _enqueue_run(session: Session, dataset_id: int) -> int:
    """Returns the run that will train on the current samples.

    A queued run has not read the samples yet, so it is reused instead of stacking a
    second run behind it when an upload arrives in several batches.
    """
    queued: TrainingRun | None = session.exec(
        select(TrainingRun)
        .where(
            TrainingRun.dataset_id == dataset_id,
            TrainingRun.status == RunStatus.QUEUED,
        )
        .order_by(col(TrainingRun.id).desc())
    ).first()
    if queued is not None and queued.id is not None:
        return queued.id
    run = TrainingRun(dataset_id=dataset_id)
    session.add(run)
    session.commit()
    session.refresh(run)
    assert run.id is not None
    return run.id


def _latest_head(session: Session, dataset_id: int) -> Head | None:
    head: Head | None = session.exec(
        select(Head).where(Head.dataset_id == dataset_id).order_by(col(Head.id).desc())
    ).first()
    return head


def _get_cached_head(session: Session, dataset_id: int) -> CachedHead | None:
    head = _latest_head(session=session, dataset_id=dataset_id)
    if head is None or head.id is None:
        return None
    cached = _head_cache.get(dataset_id)
    if cached is None or cached.head_id != head.id:
        cached = CachedHead(
            head_id=head.id,
            weights=trainer.load_weights(head.weights),
            class_names=list(head.class_names),
        )
        _head_cache[dataset_id] = cached
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


def _run_info(session: Session, run: TrainingRun | None) -> schemas.RunInfo | None:
    if run is None or run.id is None:
        return None
    dataset = session.get(Dataset, run.dataset_id)
    assert dataset is not None
    return schemas.RunInfo(
        id=run.id,
        dataset=dataset.name,
        status=run.status,
        head_id=run.head_id,
        error=run.error,
        created_at=run.created_at,
        finished_at=run.finished_at,
    )
