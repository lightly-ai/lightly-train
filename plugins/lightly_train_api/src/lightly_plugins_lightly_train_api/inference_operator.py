"""Auto-labels images in LightlyStudio with a model served by the LightlyTrain API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from lightly_studio.models.annotation.annotation_base import (
    AnnotationCreate,
    AnnotationType,
)
from lightly_studio.plugins.base_operator import BaseOperator, OperatorResult
from lightly_studio.plugins.operator_context import ExecutionContext, OperatorScope
from lightly_studio.plugins.parameter import (
    BaseParameter,
    FloatParameter,
    StringParameter,
)
from lightly_studio.resolvers import annotation_resolver
from sqlmodel import Session

from lightly_plugins_lightly_train_api import studio
from lightly_plugins_lightly_train_api.client import ApiClient, ApiError

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_USER_ID = "lightly-studio"
DEFAULT_SCORE_THRESHOLD = 0.5

PARAM_API_URL = "api_url"
PARAM_USER_ID = "user_id"
PARAM_DATASET = "dataset"
PARAM_SCORE_THRESHOLD = "score_threshold"
PARAM_ANNOTATION_SOURCE = "annotation_source"


@dataclass
class LightlyTrainApiInferenceOperator(BaseOperator):
    """Runs inference against the LightlyTrain API and writes back annotations."""

    name: str = "LightlyTrain API inference"
    description: str = (
        "Predicts the images of the current view with the model the LightlyTrain API "
        "serves for a dataset, and adds the predictions as annotations."
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
                    "Dataset whose model to predict with. Defaults to the name of the "
                    "current collection."
                ),
            ),
            FloatParameter(
                name=PARAM_SCORE_THRESHOLD,
                required=False,
                default=DEFAULT_SCORE_THRESHOLD,
                description=(
                    "Minimum score for keeping a prediction. Applied by the API."
                ),
            ),
            StringParameter(
                name=PARAM_ANNOTATION_SOURCE,
                required=False,
                default="",
                description=(
                    "Annotation source the predictions are written to. Defaults to "
                    "the dataset name of the API."
                ),
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
        threshold = float(
            parameters.get(PARAM_SCORE_THRESHOLD, DEFAULT_SCORE_THRESHOLD)
        )
        if threshold < 0.0 or threshold > 1.0:
            return OperatorResult(
                success=False, message="score_threshold must be between 0 and 1"
            )

        client = ApiClient(
            url=str(parameters.get(PARAM_API_URL, DEFAULT_API_URL)),
            user_id=str(parameters.get(PARAM_USER_ID, DEFAULT_USER_ID)),
        )
        dataset = str(
            parameters.get(PARAM_DATASET, "")
        ).strip() or studio.collection_name(
            session=session, collection_id=context.collection_id
        )
        source = (
            str(parameters.get(PARAM_ANNOTATION_SOURCE, "")).strip()
            or f"lightly_train_api__{dataset}"
        )

        images = studio.images_in_view(session=session, context=context)
        if not images:
            return OperatorResult(
                success=True, message="No samples found for current view."
            )

        try:
            info = client.get_dataset(dataset)
            if info is None:
                return OperatorResult(
                    success=False, message=f"The API does not know dataset '{dataset}'."
                )
            if info["head"] is None:
                return OperatorResult(
                    success=False,
                    message=f"Dataset '{dataset}' has no trained model yet.",
                )
            label_ids = studio.get_or_create_label_ids(
                session=session,
                collection_id=context.collection_id,
                names=list(info["class_names"]),
            )
            annotations = _predict(
                client=client,
                dataset=dataset,
                images=images,
                task=str(info["task"]),
                threshold=threshold,
                label_ids=label_ids,
            )
        except ApiError as error:
            logger.exception("LightlyTrain API inference failed")
            return OperatorResult(success=False, message=str(error))

        if annotations:
            annotation_resolver.create_many(
                session=session,
                parent_collection_id=context.collection_id,
                annotations=annotations,
                collection_name=source,
            )
        return OperatorResult(
            success=True,
            message=(
                f"Added {len(annotations)} annotations to {len(images)} samples "
                f"from dataset '{dataset}'."
            ),
        )


def _predict(
    client: ApiClient,
    dataset: str,
    images: list[Any],
    task: str,
    threshold: float,
    label_ids: dict[str, Any],
) -> list[AnnotationCreate]:
    annotations: list[AnnotationCreate] = []
    for start in range(0, len(images), studio.BATCH_SIZE):
        batch = images[start : start + studio.BATCH_SIZE]
        files = [
            ("files", (image.file_path_abs, studio.read_image(image.file_path_abs)))
            for image in batch
        ]
        predictions = client.predict(dataset=dataset, files=files, threshold=threshold)
        for image, prediction in zip(batch, predictions):
            annotations.extend(
                _detections(
                    image=image,
                    prediction=prediction,
                    threshold=threshold,
                    label_ids=label_ids,
                )
                if task == "detection"
                else _classification(
                    image=image,
                    prediction=prediction,
                    threshold=threshold,
                    label_ids=label_ids,
                )
            )
    return annotations


def _classification(
    image: Any, prediction: dict[str, Any], threshold: float, label_ids: dict[str, Any]
) -> list[AnnotationCreate]:
    score = float(prediction["score"])
    label_id = label_ids.get(prediction["label"])
    if score < threshold or label_id is None:
        return []
    return [
        AnnotationCreate(
            annotation_label_id=label_id,
            annotation_type=AnnotationType.CLASSIFICATION,
            parent_sample_id=image.sample_id,
            confidence=score,
        )
    ]


def _detections(
    image: Any, prediction: dict[str, Any], threshold: float, label_ids: dict[str, Any]
) -> list[AnnotationCreate]:
    """Converts the absolute xyxy boxes of the API to Studio's xywh boxes."""
    annotations: list[AnnotationCreate] = []
    for box in prediction["boxes"]:
        score = float(box["score"])
        label_id = label_ids.get(box["label"])
        if score < threshold or label_id is None:
            continue
        x1, y1, x2, y2 = (float(value) for value in box["box"])
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        # Studio stores integer pixels and rejects boxes reaching outside the image.
        left = min(max(round(x1), 0), image.width)
        top = min(max(round(y1), 0), image.height)
        width = min(max(round(x2), 0), image.width) - left
        height = min(max(round(y2), 0), image.height) - top
        if width <= 0 or height <= 0:
            continue
        annotations.append(
            AnnotationCreate(
                annotation_label_id=label_id,
                annotation_type=AnnotationType.OBJECT_DETECTION,
                parent_sample_id=image.sample_id,
                x=left,
                y=top,
                width=width,
                height=height,
                confidence=score,
            )
        )
    return annotations
