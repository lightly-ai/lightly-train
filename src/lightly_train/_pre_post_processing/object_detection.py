#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import Module
from torchvision.ops import box_convert
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train._data import file_helpers
from lightly_train._task_models.object_detection_components import tiling_utils
from lightly_train._task_models.task_model_io import (
    BaseModelOutput,
    RowIndexableOutput,
)
from lightly_train.types import PathLike


@dataclass
class ObjectDetectionOutput(BaseModelOutput):
    """Raw object detection output: logits and normalized ``cxcywh`` boxes.

    Attributes:
        logits: Shape ``(B, num_queries, num_classes)``. Raw (pre-sigmoid) per-query
            class logits.
        boxes: Shape ``(B, num_queries, 4)``. Normalized ``cxcywh`` boxes (values in
            ``[0, 1]``) relative to the model input size.
    """

    logits: Tensor
    boxes: Tensor


@dataclass
class ObjectDetectionPrediction(RowIndexableOutput):
    """Predictions for one image in original-image ``xyxy`` coordinates.

    Attributes:
        labels: Shape ``(N,)``, ``int64``. Predicted class index for each of the
            ``N`` kept detections.
        bboxes: Shape ``(N, 4)``. ``xyxy`` boxes in original-image pixel coordinates.
        scores: Shape ``(N,)``. Confidence score in ``[0, 1]`` for each detection.

    Supports row filtering, which always returns a new prediction::

        kept = prediction[prediction.scores > 0.5]
        cats = prediction[prediction.labels == 17]

    Note:
        ``len(prediction)`` returns the number of fields (3) and iterating yields
        field names, because predictions also behave like a ``Mapping``. Use
        :attr:`num_detections` to count detections.
    """

    labels: Tensor
    bboxes: Tensor
    scores: Tensor

    @property
    def num_detections(self) -> int:
        """Number of detections ``N``.

        Note that ``len(prediction)`` returns the number of fields instead, because
        predictions also behave like a ``Mapping``.
        """
        return self.num_rows

    def to_torchmetrics(self) -> dict[str, Tensor]:
        """Convert to a format compatible with TorchMetrics.

        Returns:
            A dictionary with keys ``"boxes"``, ``"scores"``, and ``"labels"``.
        """
        return {
            "boxes": self.bboxes,
            "scores": self.scores,
            "labels": self.labels,
        }


@dataclass
class ObjectDetectionMetadata:
    orig_h: int
    orig_w: int
    tile_coordinates: Tensor | None = None


@dataclass(frozen=True)
class ObjectDetectionSahiConfig:
    overlap: float
    nms_iou_threshold: float
    global_local_iou_threshold: float


def _sahi_tile_coordinates(metadata: Sequence[ObjectDetectionMetadata]) -> Tensor:
    """Return the tile coordinates of the single image that SAHI postprocesses."""
    if len(metadata) != 1 or metadata[0].tile_coordinates is None:
        raise ValueError("SAHI postprocessing expects metadata for one image.")
    return metadata[0].tile_coordinates


def decode_object_detection_output(
    *,
    logits: Tensor,
    boxes: Tensor,
    target_sizes: Tensor,
    num_top_queries: int,
    internal_class_to_class: Tensor,
) -> list[ObjectDetectionPrediction]:
    """Decode raw detection outputs into one prediction per image.

    Args:
        logits: Shape ``(B, num_queries, num_classes)``. Raw (pre-sigmoid) logits.
        boxes: Shape ``(B, num_queries, 4)``. Normalized ``cxcywh`` boxes.
        target_sizes: Shape ``(B, 2)``. ``(width, height)`` the boxes are scaled to.
        num_top_queries: Number of ``(query, class)`` pairs kept per image.
        internal_class_to_class:
            Shape ``(num_classes,)``. Maps internal contiguous class indices to
            user-facing class ids.

    Returns:
        A list of ``B`` predictions, each holding ``num_top_queries`` detections with
        ``xyxy`` boxes in ``target_sizes`` coordinates. No score thresholding is
        applied.
    """
    scores = logits.sigmoid()
    num_classes = scores.shape[-1]
    scores, index = scores.flatten(1).topk(num_top_queries, dim=-1)
    labels = internal_class_to_class[index % num_classes]
    query_index = index // num_classes
    boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    boxes = boxes.gather(1, query_index.unsqueeze(-1).expand(-1, -1, 4))
    boxes = boxes * target_sizes.repeat(1, 2).unsqueeze(1)
    return [
        ObjectDetectionPrediction(labels=labels_i, bboxes=boxes_i, scores=scores_i)
        for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
    ]


class ObjectDetectionPreprocessor(Module):
    """Host image preparation plus dense, batch-friendly preprocessing."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        image_normalize: dict[str, tuple[float, ...]] | None,
        expected_input_channels: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.image_normalize = image_normalize
        self.expected_input_channels = expected_input_channels

    def preprocess_image(
        self,
        image: PathLike | PILImage | Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        sahi_config: ObjectDetectionSahiConfig | None = None,
    ) -> tuple[Tensor, ObjectDetectionMetadata]:
        """Prepare a single image for the model.

        Args:
            image: Input image as a path, PIL image, or ``(C, H, W)`` tensor.
            device: Device the returned tensor is placed on.
            dtype: Floating point dtype the image is converted to.
            sahi_config:
                If ``None``, the image is resized to ``image_size`` and returned as a
                ``(C, H, W)`` tensor. Otherwise the image is tiled and returned as a
                ``(1 + num_tiles, C, H, W)`` stack whose first entry is the resized
                global image, with the tile coordinates set on the metadata.
        """
        image_tensor = file_helpers.as_image_tensor(image).to(device)
        orig_h, orig_w = image_tensor.shape[-2:]
        image_tensor = self._validate_channels(image_tensor)
        image_tensor = transforms_functional.to_dtype(
            image_tensor, dtype=dtype, scale=True
        )
        metadata = ObjectDetectionMetadata(orig_h=orig_h, orig_w=orig_w)
        if sahi_config is None:
            return transforms_functional.resize(image_tensor, self.image_size), metadata

        tiles, coordinates = tiling_utils.tile_image(
            image=image_tensor,
            overlap=sahi_config.overlap,
            tile_size=self.image_size,
        )
        global_image = transforms_functional.resize(
            image_tensor, self.image_size
        ).unsqueeze(0)
        metadata.tile_coordinates = coordinates
        return torch.cat([global_image, tiles]), metadata

    def _validate_channels(self, image: Tensor) -> Tensor:
        if image.shape[-3] == 1 and self.expected_input_channels > 1:
            return image.expand(self.expected_input_channels, -1, -1)
        if image.shape[-3] != self.expected_input_channels:
            raise ValueError(
                f"Image has {image.shape[-3]} channels but model expects {self.expected_input_channels}."
            )
        return image

    def preprocess_batch(self, batch: Tensor) -> Tensor:
        if self.image_normalize is None:
            return batch
        return cast(
            Tensor,
            transforms_functional.normalize(
                batch,
                mean=list(self.image_normalize["mean"]),
                std=list(self.image_normalize["std"]),
            ),
        )


class ObjectDetectionPostprocessor(Module):
    """Decode raw object detection outputs into typed per-image predictions."""

    internal_class_to_class: Tensor

    def __init__(
        self,
        *,
        num_classes: int,
        num_top_queries: int,
        internal_class_to_class: Tensor,
        image_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.image_size = image_size
        self.register_buffer(
            "internal_class_to_class", internal_class_to_class, persistent=False
        )

    def postprocess(
        self,
        raw: ObjectDetectionOutput,
        metadata: Sequence[ObjectDetectionMetadata],
        threshold: float,
        *,
        sahi_config: ObjectDetectionSahiConfig | None = None,
    ) -> list[ObjectDetectionPrediction]:
        """Decode raw outputs into one prediction per image.

        Args:
            raw: Raw model output for all images in the batch.
            metadata: Per-image metadata as returned by the preprocessor.
            threshold: Detections with a score <= threshold are discarded.
            sahi_config:
                If given, ``raw`` and ``metadata`` must describe a single tiled image
                and the global and tile predictions are merged into one prediction.
        """
        if sahi_config is not None:
            tile_coordinates = _sahi_tile_coordinates(metadata)
            target_sizes = self._target_sizes(
                metadata, device=raw.boxes.device, tile_coordinates=tile_coordinates
            )
            predictions = decode_object_detection_output(
                logits=raw.logits,
                boxes=raw.boxes,
                target_sizes=target_sizes,
                num_top_queries=self.num_top_queries,
                internal_class_to_class=self.internal_class_to_class,
            )
            return [
                tiling_utils.combine_sahi_object_detection_predictions(
                    predictions=predictions,
                    tile_coordinates=tile_coordinates,
                    threshold=threshold,
                    nms_iou_threshold=sahi_config.nms_iou_threshold,
                    global_local_iou_threshold=sahi_config.global_local_iou_threshold,
                )
            ]

        target_sizes = self._target_sizes(metadata, device=raw.boxes.device)
        predictions = decode_object_detection_output(
            logits=raw.logits,
            boxes=raw.boxes,
            target_sizes=target_sizes,
            num_top_queries=self.num_top_queries,
            internal_class_to_class=self.internal_class_to_class,
        )
        return [prediction[prediction.scores > threshold] for prediction in predictions]

    def _target_sizes(
        self,
        metadata: Sequence[ObjectDetectionMetadata],
        *,
        device: torch.device,
        tile_coordinates: Tensor | None = None,
    ) -> Tensor:
        if tile_coordinates is None:
            sizes = [[item.orig_w, item.orig_h] for item in metadata]
        else:
            tile_h, tile_w = self.image_size
            sizes = [
                [metadata[0].orig_w, metadata[0].orig_h],
                *[[tile_w, tile_h] for _ in tile_coordinates],
            ]
        return torch.tensor(sizes, dtype=torch.int64, device=device)
