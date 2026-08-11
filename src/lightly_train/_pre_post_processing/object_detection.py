#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import Module
from torchvision.ops import batched_nms, box_convert, box_iou
from torchvision.transforms.v2 import functional as transforms_functional

from lightly_train._data import file_helpers
from lightly_train._pre_post_processing import tiling
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


def yolo_to_xyxy(batch_boxes: Sequence[Tensor]) -> list[Tensor]:
    """Convert boxes from normalized ``cxcywh`` to normalized ``xyxy``.

    Args:
        batch_boxes: Per-image boxes of shape ``(N, 4)`` with values in ``[0, 1]``.

    Returns:
        Per-image boxes in normalized ``xyxy`` format.
    """
    converted_boxes = []
    for sample_boxes in batch_boxes:
        cxcywh = sample_boxes
        if cxcywh.ndim == 1:
            cxcywh = cxcywh.reshape(-1, 4)
        converted_boxes.append(box_convert(cxcywh, in_fmt="cxcywh", out_fmt="xyxy"))
    return converted_boxes


def denormalize_xyxy_boxes(
    boxes: Sequence[Tensor],
    sizes: Sequence[tuple[int, int]],
) -> list[Tensor]:
    """Scale normalized ``xyxy`` boxes to pixel coordinates.

    Args:
        boxes: Per-image boxes of shape ``(N, 4)`` with values in ``[0, 1]``.
        sizes: Per-image ``(width, height)`` to scale by.

    Returns:
        Per-image boxes in ``xyxy`` pixel coordinates.
    """
    denormalized_boxes = []
    for sample_boxes, (width, height) in zip(boxes, sizes):
        scale = sample_boxes.new_tensor([width, height, width, height])
        denormalized_boxes.append(sample_boxes * scale)
    return denormalized_boxes


def targets_to_torchmetrics(
    *,
    bboxes: Sequence[Tensor],
    classes: Sequence[Tensor],
    original_sizes: Sequence[tuple[int, int]],
) -> list[dict[str, Tensor]]:
    """Convert ground truth boxes into a format compatible with TorchMetrics.

    This is the ground truth counterpart to
    :meth:`ObjectDetectionPrediction.to_torchmetrics`: it brings targets into the
    same ``xyxy`` original-image pixel coordinates the predictions use.

    Args:
        bboxes:
            Per-image ground truth boxes of shape ``(N, 4)`` in YOLO format
            (normalized ``cxcywh``).
        classes: Per-image ground truth class ids of shape ``(N,)``.
        original_sizes: Per-image ``(width, height)`` of the original image.

    Returns:
        A list with one dictionary per image, with keys ``"boxes"`` and ``"labels"``.
    """
    boxes_xyxy = yolo_to_xyxy(bboxes)
    boxes_denormalized = denormalize_xyxy_boxes(boxes_xyxy, original_sizes)
    return [
        {"boxes": boxes, "labels": labels}
        for boxes, labels in zip(boxes_denormalized, classes)
    ]


def combine_object_detection_tiles(
    pred_global: Mapping[str, Tensor],
    pred_tiles: Mapping[str, Tensor],
    nms_iou_threshold: float = 0.2,
    global_local_iou_threshold: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Combine predictions from the global view (full image) and local views (image tiles).

    Args:
        pred_global: Mapping with keys "labels", "bboxes", "scores". An
            :class:`ObjectDetectionPrediction` satisfies this.
        pred_tiles: Mapping with keys "labels", "bboxes", "scores".
        nms_iou_threshold: IoU used in NMS of tiles predictions.
        global_local_iou_threshold: IoU above which a tile box is removed if it matches a global box of same label.

    Returns:
        Filtered labels, boxes, scores as a tuple.
    """
    # Get tiles and global predictions.
    labels_global = pred_global["labels"]
    boxes_global = pred_global["bboxes"]
    scores_global = pred_global["scores"]
    labels_tiles = pred_tiles["labels"]
    boxes_tiles = pred_tiles["bboxes"]
    scores_tiles = pred_tiles["scores"]

    # NMS on tiles predictions is needed due overlapping tiles. Suppression is
    # class-aware so a high-confidence prediction cannot hide another class.
    if boxes_tiles.numel() > 0:
        keep = batched_nms(boxes_tiles, scores_tiles, labels_tiles, nms_iou_threshold)
        labels_tiles = labels_tiles[keep]
        boxes_tiles = boxes_tiles[keep]
        scores_tiles = scores_tiles[keep]

    # Drop tile boxes that overlap global boxes of same class
    if boxes_global.numel() > 0 and boxes_tiles.numel() > 0:
        # Compute overlap between tiles and global predictions.
        ious = box_iou(boxes_tiles, boxes_global)

        # Only keep tiles predictions that do not overlap above the threshold with
        # any global prediction of the same class. The same-label check must be
        # applied before reducing over global predictions: reducing first (e.g.
        # via the single max-IoU global box) would miss a same-label overlap that
        # is not the strongest one.
        same_label = labels_tiles[:, None] == labels_global[None, :]
        overlaps_same_label = (same_label & (ious > global_local_iou_threshold)).any(
            dim=1
        )
        keep = ~overlaps_same_label
        labels_tiles = labels_tiles[keep]
        boxes_tiles = boxes_tiles[keep]
        scores_tiles = scores_tiles[keep]

    # Concatenate the global and tiles predictions
    labels = torch.cat([labels_global, labels_tiles], dim=0)
    boxes = torch.cat([boxes_global, boxes_tiles], dim=0)
    scores = torch.cat([scores_global, scores_tiles], dim=0)

    return labels, boxes, scores


def combine_sahi_object_detection_predictions(
    *,
    predictions: Sequence[ObjectDetectionPrediction],
    tile_coordinates: Tensor,
    threshold: float,
    nms_iou_threshold: float,
    global_local_iou_threshold: float,
) -> ObjectDetectionPrediction:
    """Offset, filter, and merge decoded global/tile predictions for one image."""
    global_prediction = predictions[0]
    tile_prediction = ObjectDetectionPrediction(
        labels=torch.cat([prediction.labels for prediction in predictions[1:]]),
        bboxes=torch.cat(
            [
                prediction.bboxes + coordinates.repeat(2)
                for prediction, coordinates in zip(
                    predictions[1:],
                    tile_coordinates.to(global_prediction.bboxes.device),
                )
            ]
        ),
        scores=torch.cat([prediction.scores for prediction in predictions[1:]]),
    )
    global_prediction = global_prediction[global_prediction.scores > threshold]
    tile_prediction = tile_prediction[tile_prediction.scores > threshold]

    labels, bboxes, scores = combine_object_detection_tiles(
        pred_global=global_prediction,
        pred_tiles=tile_prediction,
        nms_iou_threshold=nms_iou_threshold,
        global_local_iou_threshold=global_local_iou_threshold,
    )
    return ObjectDetectionPrediction(labels=labels, bboxes=bboxes, scores=scores)


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

        tiles, coordinates = tiling.tile_image(
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
                combine_sahi_object_detection_predictions(
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
