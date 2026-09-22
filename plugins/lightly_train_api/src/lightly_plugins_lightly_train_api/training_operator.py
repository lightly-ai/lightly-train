"""Trains a model on the LightlyTrain API from the annotations in LightlyStudio."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence
from uuid import UUID

from lightly_studio.models.annotation.annotation_base import AnnotationType
from lightly_studio.plugins.base_operator import BaseOperator, OperatorResult
from lightly_studio.plugins.operator_context import ExecutionContext, OperatorScope
from lightly_studio.plugins.parameter import (
    BaseParameter,
    BoolParameter,
    IntParameter,
    StringParameter,
)
from lightly_studio.resolvers import collection_resolver
from sqlmodel import Session

from lightly_plugins_lightly_train_api import studio
from lightly_plugins_lightly_train_api.client import (
    FAILED,
    FINISHED,
    ApiClient,
    ApiError,
)

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_USER_ID = "lightly-studio"
DEFAULT_TIMEOUT_S = 900

PARAM_API_URL = "api_url"
PARAM_USER_ID = "user_id"
PARAM_DATASET = "dataset"
PARAM_ANNOTATION_SOURCE = "annotation_source"
PARAM_WAIT = "wait_for_training"
PARAM_TIMEOUT_S = "timeout_s"


@dataclass
class Sample:
    """One annotated image, ready to be sent to the API."""

    key: str
    path: str
    content_hash: str
    label: str | None = None
    boxes: list[list[float]] | None = None
    labels: list[str] | None = None

    def state(self) -> dict[str, Any]:
        """The sample as `/samples/diff` expects it, without the image bytes."""
        state: dict[str, Any] = {"key": self.key, "content_hash": self.content_hash}
        if self.boxes is None:
            state["label"] = self.label
        else:
            state["annotation"] = {"boxes": self.boxes, "labels": self.labels}
        return state

    def annotation(self) -> str:
        return json.dumps({"boxes": self.boxes, "labels": self.labels})


@dataclass
class LightlyTrainApiTrainingOperator(BaseOperator):
    """Sends the annotated images of the current view to the LightlyTrain API."""

    name: str = "LightlyTrain API training"
    description: str = (
        "Uploads the annotated images of the current view to the LightlyTrain API and "
        "retrains the model of that dataset."
    )

    @property
    def parameters(self) -> list[BaseParameter]:
        """Return the list of parameters this operator expects."""
        return [
            StringParameter(
                name=PARAM_API_URL,
                required=True,
                default=DEFAULT_API_URL,
                description="Base URL of the LightlyTrain API service.",
            ),
            StringParameter(
                name=PARAM_USER_ID,
                required=True,
                default=DEFAULT_USER_ID,
                description="Identifies the owner of the dataset on the API.",
            ),
            StringParameter(
                name=PARAM_DATASET,
                required=False,
                default="",
                description=(
                    "Dataset to train on the API. Defaults to the name of the current "
                    "collection."
                ),
            ),
            StringParameter(
                name=PARAM_ANNOTATION_SOURCE,
                required=False,
                default="",
                description=(
                    "Only train on annotations from this source. Empty uses every "
                    "annotation of the view."
                ),
            ),
            BoolParameter(
                name=PARAM_WAIT,
                required=False,
                default=True,
                description="Wait for the training run to finish before returning.",
            ),
            IntParameter(
                name=PARAM_TIMEOUT_S,
                required=False,
                default=DEFAULT_TIMEOUT_S,
                description="How long to wait for the training run, in seconds.",
            ),
        ]

    @property
    def supported_scopes(self) -> list[OperatorScope]:
        """Return the list of scopes this operator can be triggered from."""
        return [OperatorScope.IMAGE]

    def execute(
        self,
        *,
        session: Session,
        context: ExecutionContext,
        parameters: dict[str, Any],
    ) -> OperatorResult:
        """Execute the operator with the given parameters."""
        client = ApiClient(
            url=str(parameters.get(PARAM_API_URL, DEFAULT_API_URL)),
            user_id=str(parameters.get(PARAM_USER_ID, DEFAULT_USER_ID)),
        )
        source = str(parameters.get(PARAM_ANNOTATION_SOURCE, "")).strip()
        dataset = str(
            parameters.get(PARAM_DATASET, "")
        ).strip() or studio.collection_name(
            session=session, collection_id=context.collection_id
        )

        images = studio.images_in_view(session=session, context=context)
        if not images:
            return OperatorResult(
                success=True, message="No samples found for current view."
            )

        try:
            samples, task, skipped = _collect_samples(
                session=session, images=images, source=source
            )
        except ValueError as error:
            return OperatorResult(success=False, message=str(error))
        if not samples:
            return OperatorResult(
                success=False,
                message=(
                    "None of the images in the current view carry annotations"
                    f"{f' from source {source!r}' if source else ''}."
                ),
            )

        try:
            return _train(
                client=client,
                dataset=dataset,
                samples=samples,
                task=task,
                skipped=skipped,
                wait=bool(parameters.get(PARAM_WAIT, True)),
                timeout_s=float(parameters.get(PARAM_TIMEOUT_S, DEFAULT_TIMEOUT_S)),
            )
        except ApiError as error:
            logger.exception("LightlyTrain API training failed")
            return OperatorResult(success=False, message=str(error))


def _train(
    client: ApiClient,
    dataset: str,
    samples: list[Sample],
    task: AnnotationType,
    skipped: int,
    wait: bool,
    timeout_s: float,
) -> OperatorResult:
    diff = client.diff(dataset=dataset, samples=[sample.state() for sample in samples])
    outdated = set(diff["new"]) | set(diff["changed"])
    todo = [sample for sample in samples if sample.key in outdated]
    if not todo:
        return OperatorResult(
            success=True,
            message=(
                f"Dataset '{dataset}' is already up to date with "
                f"{len(samples)} annotated samples."
            ),
        )

    run_id: int | None = None
    for start in range(0, len(todo), studio.BATCH_SIZE):
        batch = todo[start : start + studio.BATCH_SIZE]
        files = [
            ("files", (sample.key, studio.read_image(sample.path))) for sample in batch
        ]
        keys = [sample.key for sample in batch]
        if task is AnnotationType.OBJECT_DETECTION:
            response = client.upload(
                dataset=dataset,
                files=files,
                keys=keys,
                annotations=[sample.annotation() for sample in batch],
            )
        else:
            response = client.upload(
                dataset=dataset,
                files=files,
                keys=keys,
                labels=[sample.label or "" for sample in batch],
            )
        run_id = response.get("run_id") or run_id

    ingested = f"Ingested {len(todo)} of {len(samples)} annotated samples"
    detail = f" ({skipped} unannotated images skipped)" if skipped else ""
    if run_id is None:
        return OperatorResult(success=True, message=f"{ingested}{detail}.")
    if not wait:
        return OperatorResult(
            success=True, message=f"{ingested}{detail}. Training run {run_id} started."
        )

    run = client.wait_for_run(run_id=run_id, timeout_s=timeout_s)
    if run["status"] == FAILED:
        return OperatorResult(
            success=False, message=f"Training run {run_id} failed: {run['error']}"
        )
    if run["status"] != FINISHED:
        return OperatorResult(
            success=True,
            message=(
                f"{ingested}{detail}. Training run {run_id} is still "
                f"{run['status']} after {timeout_s:.0f}s."
            ),
        )
    return OperatorResult(
        success=True, message=f"{ingested}{detail}. Training run {run_id} succeeded."
    )


def _collect_samples(
    session: Session, images: Sequence[Any], source: str
) -> tuple[list[Sample], AnnotationType, int]:
    """Builds one API sample per annotated image and decides the task.

    An image carrying at least one bounding box makes the whole run a detection run,
    because the API trains one task per dataset.
    """
    source_names = _annotation_source_names(session=session, images=images)
    per_image = [
        [
            annotation
            for annotation in image.sample.annotations
            if not source
            or source_names.get(annotation.annotation_collection_id) == source
        ]
        for image in images
    ]
    task = (
        AnnotationType.OBJECT_DETECTION
        if any(
            annotation.annotation_type is AnnotationType.OBJECT_DETECTION
            and annotation.object_detection_details is not None
            for image_annotations in per_image
            for annotation in image_annotations
        )
        else AnnotationType.CLASSIFICATION
    )

    samples: list[Sample] = []
    skipped = 0
    for image, image_annotations in zip(images, per_image):
        sample = (
            _detection_sample(image=image, annotations=image_annotations)
            if task is AnnotationType.OBJECT_DETECTION
            else _classification_sample(image=image, annotations=image_annotations)
        )
        if sample is None:
            skipped += 1
        else:
            samples.append(sample)
    return samples, task, skipped


def _detection_sample(image: Any, annotations: Sequence[Any]) -> Sample | None:
    boxes: list[list[float]] = []
    labels: list[str] = []
    for annotation in annotations:
        details = annotation.object_detection_details
        if (
            annotation.annotation_type is not AnnotationType.OBJECT_DETECTION
            or details is None
        ):
            continue
        boxes.append(
            [
                float(details.x),
                float(details.y),
                float(details.x) + float(details.width),
                float(details.y) + float(details.height),
            ]
        )
        labels.append(annotation.annotation_label.annotation_label_name)
    if not boxes:
        return None
    return Sample(
        key=image.file_path_abs,
        path=image.file_path_abs,
        content_hash=studio.content_hash(studio.read_image(image.file_path_abs)),
        boxes=boxes,
        labels=labels,
    )


def _classification_sample(image: Any, annotations: Sequence[Any]) -> Sample | None:
    """Uses the most confident classification, because the API trains one label."""
    candidates = [
        annotation
        for annotation in annotations
        if annotation.annotation_type is AnnotationType.CLASSIFICATION
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda annotation: annotation.confidence or 0.0)
    return Sample(
        key=image.file_path_abs,
        path=image.file_path_abs,
        content_hash=studio.content_hash(studio.read_image(image.file_path_abs)),
        label=best.annotation_label.annotation_label_name,
    )


def _annotation_source_names(
    session: Session, images: Sequence[Any]
) -> dict[UUID, str]:
    """Maps annotation collection ids to their names, to filter by source."""
    ids = {
        annotation.annotation_collection_id
        for image in images
        for annotation in image.sample.annotations
    }
    if not ids:
        return {}
    names: dict[UUID, str] = collection_resolver.get_names_by_ids(
        session=session, collection_ids=list(ids)
    )
    return names
