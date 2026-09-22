"""Helpers shared by the training and the inference operator."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from lightly_studio.models.annotation_label import AnnotationLabelCreate
from lightly_studio.plugins.operator_context import ExecutionContext
from lightly_studio.resolvers import (
    annotation_label_resolver,
    collection_resolver,
    image_resolver,
)
from lightly_studio.resolvers.image_filter import ImageFilter
from lightly_studio.resolvers.sample_resolver.sample_filter import SampleFilter
from sqlmodel import Session

# Images sent to the API in one request.
BATCH_SIZE = 16


def as_image_filter(context_filter: Any) -> ImageFilter | None:
    """Reduces the operator's context filter to the image filter the resolver takes."""
    if isinstance(context_filter, ImageFilter):
        return context_filter
    if isinstance(context_filter, SampleFilter):
        return ImageFilter(sample_filter=context_filter)
    return None


def images_in_view(session: Session, context: ExecutionContext) -> list[Any]:
    """Returns the images of the current view, in collection order."""
    result = image_resolver.get_all_by_collection_id(
        session=session,
        collection_id=context.collection_id,
        filters=as_image_filter(context.context_filter),
    )
    return list(result.samples)


def collection_name(session: Session, collection_id: UUID) -> str:
    """Name of the Studio collection, used as the default API dataset name."""
    collection = collection_resolver.get_by_id(
        session=session, collection_id=collection_id
    )
    if collection is None:
        raise ValueError(f"Collection {collection_id} doesn't exist")
    return str(collection.name)


def get_or_create_label_ids(
    session: Session, collection_id: UUID, names: list[str]
) -> dict[str, UUID]:
    """Ensures a Studio label exists for every class name of the API model."""
    collection = collection_resolver.get_by_id(
        session=session, collection_id=collection_id
    )
    if collection is None:
        raise ValueError(f"Collection {collection_id} doesn't exist")

    label_ids: dict[str, UUID] = {}
    for name in names:
        label = annotation_label_resolver.get_by_label_name(
            session=session, dataset_id=collection.dataset_id, label_name=name
        )
        if label is None:
            label = annotation_label_resolver.create(
                session=session,
                label=AnnotationLabelCreate(
                    dataset_id=collection.dataset_id, annotation_label_name=name
                ),
            )
        label_ids[name] = label.annotation_label_id
    return label_ids


def read_image(path: str) -> bytes:
    with open(path, "rb") as file:
        return file.read()


def content_hash(data: bytes) -> str:
    """Must match the hash the API computes over the uploaded bytes."""
    return hashlib.sha256(data).hexdigest()
