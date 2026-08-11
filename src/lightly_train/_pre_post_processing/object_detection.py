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
class ObjectDetectionBatchOutput(BaseModelOutput):
    """Raw object detection output for a whole batch of model inputs.

    This is what the model's ``forward`` returns: per-query class logits and normalized
    ``cxcywh`` boxes, before any decoding.

    Attributes:
        logits: Shape ``(B, num_queries, num_classes)``. Raw (pre-sigmoid) per-query
            class logits.
        boxes: Shape ``(B, num_queries, 4)``. Normalized ``cxcywh`` boxes (values in
            ``[0, 1]``) relative to the model input size.

    Note that ``B`` counts model input rows, not input images: with SAHI a single
    image contributes one global row plus one row per tile.
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
class ObjectDetectionBatchPrediction(BaseModelOutput):
    """Dense, unfiltered predictions for a whole batch of model inputs.

    This is what the batched half of postprocessing produces: every row keeps exactly
    ``num_top_queries`` detections, so all fields stay rectangular and the stage is
    free of data-dependent control flow. Score thresholding, which makes the number of
    detections differ per image, happens in the per-image half instead.

    Attributes:
        labels: Shape ``(B, K)``, ``int64``. Predicted class id per kept detection.
        bboxes: Shape ``(B, K, 4)``. ``xyxy`` boxes in target-size coordinates.
        scores: Shape ``(B, K)``. Confidence score in ``[0, 1]``.

    Note that ``B`` counts model input rows, not input images: with SAHI a single
    image contributes one global row plus one row per tile.
    """

    labels: Tensor
    bboxes: Tensor
    scores: Tensor

    def select_rows(self, start: int, end: int) -> ObjectDetectionBatchPrediction:
        """Return the rows in ``[start, end)``, for example the rows of one image."""
        return ObjectDetectionBatchPrediction(
            labels=self.labels[start:end],
            bboxes=self.bboxes[start:end],
            scores=self.scores[start:end],
        )

    def to_predictions(self) -> list[ObjectDetectionPrediction]:
        """Return one :class:`ObjectDetectionPrediction` per row."""
        return [
            ObjectDetectionPrediction(labels=labels, bboxes=bboxes, scores=scores)
            for labels, bboxes, scores in zip(self.labels, self.bboxes, self.scores)
        ]


@dataclass
class ObjectDetectionMetadata:
    orig_h: int
    orig_w: int
    tile_coordinates: Tensor | None = None

    @property
    def num_rows(self) -> int:
        """Number of model input rows this image occupies.

        One row for the image itself, plus one row per SAHI tile.
        """
        if self.tile_coordinates is None:
            return 1
        return 1 + len(self.tile_coordinates)


@dataclass(frozen=True)
class ObjectDetectionSAHIConfig:
    overlap: float
    nms_iou_threshold: float
    global_local_iou_threshold: float


def decode_object_detection_output(
    *,
    raw: ObjectDetectionBatchOutput,
    target_sizes: Tensor,
    num_top_queries: int,
    internal_class_to_class: Tensor,
) -> ObjectDetectionBatchPrediction:
    """Decode raw detection outputs into dense per-row predictions.

    Args:
        raw:
            The model's :class:`ObjectDetectionBatchOutput`, holding raw (pre-sigmoid)
            logits of shape ``(B, num_queries, num_classes)`` and normalized ``cxcywh``
            boxes of shape ``(B, num_queries, 4)``. Note that ``B`` counts model input
            rows, not input images: with SAHI a single image contributes one global row
            plus one row per tile.
        target_sizes: Shape ``(B, 2)``. ``(width, height)`` the boxes are scaled to.
        num_top_queries: Number of ``(query, class)`` pairs kept per row.
        internal_class_to_class:
            Shape ``(num_classes,)``. Maps internal contiguous class indices to
            user-facing class ids.

    Returns:
        An :class:`ObjectDetectionBatchPrediction` holding ``num_top_queries``
        detections per row with ``xyxy`` boxes in ``target_sizes`` coordinates. No
        score thresholding is applied.
    """
    scores = raw.logits.sigmoid()
    num_classes = scores.shape[-1]
    scores, index = scores.flatten(1).topk(num_top_queries, dim=-1)
    labels = internal_class_to_class[index % num_classes]
    query_index = index // num_classes
    boxes = box_convert(raw.boxes, in_fmt="cxcywh", out_fmt="xyxy")
    boxes = boxes.gather(1, query_index.unsqueeze(-1).expand(-1, -1, 4))
    boxes = boxes * target_sizes.repeat(1, 2).unsqueeze(1)
    return ObjectDetectionBatchPrediction(labels=labels, bboxes=boxes, scores=scores)


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
    batch_prediction: ObjectDetectionBatchPrediction,
    tile_coordinates: Tensor,
    threshold: float,
    nms_iou_threshold: float,
    global_local_iou_threshold: float,
) -> ObjectDetectionPrediction:
    """Offset, filter, and merge the decoded rows of one tiled image.

    Args:
        batch_prediction:
            The decoded rows belonging to a single image: the global image first,
            followed by one row per tile.
        tile_coordinates: Shape ``(num_tiles, 2)``. Top-left ``(x, y)`` of each tile.
        threshold: Detections with a score <= threshold are discarded.
        nms_iou_threshold: IoU used in NMS of the tile predictions.
        global_local_iou_threshold:
            IoU above which a tile box is removed if it matches a global box of the
            same label.
    """
    global_prediction = ObjectDetectionPrediction(
        labels=batch_prediction.labels[0],
        bboxes=batch_prediction.bboxes[0],
        scores=batch_prediction.scores[0],
    )
    # Tile boxes are in tile coordinates, so shift them by the tile's top-left corner
    # to bring them into original-image coordinates. The offset is per tile, hence the
    # query dimension is broadcast over.
    offsets = tile_coordinates.to(batch_prediction.bboxes.device).repeat(1, 2)
    tile_prediction = ObjectDetectionPrediction(
        labels=batch_prediction.labels[1:].flatten(),
        bboxes=(batch_prediction.bboxes[1:] + offsets.unsqueeze(1)).flatten(0, 1),
        scores=batch_prediction.scores[1:].flatten(),
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
    """Host image preparation plus dense, batch-friendly preprocessing.

    The work is split in two stages, mirrored by
    :class:`ObjectDetectionPostprocessor`:

    - :meth:`preprocess_image` runs per image on the host and is ragged: it decodes the
      input, validates channels, resizes, and optionally tiles.
    - :meth:`preprocess_batch` runs on a stacked batch and is dense and vectorized.

    :meth:`preprocess` spans both for callers that just want to hand over a list of
    images.
    """

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
        sahi_config: ObjectDetectionSAHIConfig | None = None,
    ) -> tuple[Tensor, ObjectDetectionMetadata]:
        """Prepare a single image for the model.

        Args:
            image: Input image as a path, PIL image, or ``(C, H, W)`` tensor.
            device: Device the returned tensor is placed on.
            dtype: Floating point dtype the image is converted to.
            sahi_config:
                If ``None``, the image is resized to ``image_size``. Otherwise the
                image is tiled and the resized global image is prepended to the tiles,
                with the tile coordinates set on the metadata.

        Returns:
            A ``(num_rows, C, H, W)`` stack and the image's metadata. Without
            ``sahi_config`` there is a single row, otherwise there is one global row
            followed by one row per tile. The row count is always
            ``metadata.num_rows``.
        """
        image_tensor = file_helpers.as_image_tensor(image).to(device)
        orig_h, orig_w = image_tensor.shape[-2:]
        image_tensor = self._to_expected_channels(image_tensor)
        image_tensor = transforms_functional.to_dtype(
            image_tensor, dtype=dtype, scale=True
        )
        metadata = ObjectDetectionMetadata(orig_h=orig_h, orig_w=orig_w)
        if sahi_config is None:
            resized = transforms_functional.resize(image_tensor, self.image_size)
            return resized.unsqueeze(0), metadata

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

    def _to_expected_channels(self, image: Tensor) -> Tensor:
        """Expand a grayscale image to ``expected_input_channels``.

        Raises:
            ValueError: If the image cannot be brought to the expected channel count.
        """
        if image.shape[-3] == 1 and self.expected_input_channels > 1:
            return image.expand(self.expected_input_channels, -1, -1)
        if image.shape[-3] != self.expected_input_channels:
            raise ValueError(
                f"Image has {image.shape[-3]} channels but model expects {self.expected_input_channels}."
            )
        return image

    def preprocess_batch(self, batch: Tensor) -> Tensor:
        """Normalize a stacked ``(B, C, H, W)`` batch.

        Kept separate from :meth:`preprocess_image` so this stage can run on the
        device, or be baked into an exported graph, while the host-side per-image work
        runs elsewhere (for example in dataloader workers).
        """
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

    def preprocess(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        sahi_config: ObjectDetectionSAHIConfig | None = None,
    ) -> tuple[Tensor, list[ObjectDetectionMetadata]]:
        """Prepare a batch of images for the model.

        Convenience wrapper that runs :meth:`preprocess_image` per image and
        :meth:`preprocess_batch` on the stacked result.

        Args:
            images: Input images as paths, PIL images, or ``(C, H, W)`` tensors.
            device: Device the returned batch is placed on.
            dtype: Floating point dtype the images are converted to.
            sahi_config: If given, every image is tiled.

        Returns:
            The model input batch and one metadata entry per *image*. With SAHI an
            image contributes ``metadata.num_rows`` rows to the batch, so the batch
            size and the number of metadata entries differ.

        Raises:
            ValueError: If ``images`` is empty.
        """
        if not images:
            raise ValueError("images must contain at least one image.")
        prepared = [
            self.preprocess_image(
                image, device=device, dtype=dtype, sahi_config=sahi_config
            )
            for image in images
        ]
        batch = self.preprocess_batch(torch.cat([rows for rows, _ in prepared]))
        return batch, [metadata for _, metadata in prepared]


class ObjectDetectionPostprocessor(Module):
    """Decode raw object detection outputs into typed per-image predictions.

    Mirrors the two stages of :class:`ObjectDetectionPreprocessor`:

    - :meth:`postprocess_batch` runs on the whole batch and is dense and vectorized:
      top-k selection, box conversion, and rescaling.
    - :meth:`postprocess_image` runs per image on the host and is ragged: score
      thresholding and, for tiled images, merging the tiles back together.

    :meth:`postprocess` spans both and returns one prediction per image.
    """

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

    def postprocess_batch(
        self,
        raw: ObjectDetectionBatchOutput,
        metadata: Sequence[ObjectDetectionMetadata],
    ) -> ObjectDetectionBatchPrediction:
        """Decode raw outputs into dense per-row predictions.

        Kept separate from :meth:`postprocess_image` so this stage can run on the
        device, or be baked into an exported graph: it is fully vectorized and keeps
        the same number of detections for every row. Score thresholding, which is
        data-dependent, happens in :meth:`postprocess_image` instead.

        Args:
            raw: Raw model output for all rows in the batch.
            metadata: Per-image metadata as returned by the preprocessor.
        """
        target_sizes = self._target_sizes(metadata, device=raw.boxes.device)
        return decode_object_detection_output(
            raw=raw,
            target_sizes=target_sizes,
            num_top_queries=self.num_top_queries,
            internal_class_to_class=self.internal_class_to_class,
        )

    def postprocess_image(
        self,
        batch_prediction: ObjectDetectionBatchPrediction,
        metadata: ObjectDetectionMetadata,
        threshold: float,
        *,
        sahi_config: ObjectDetectionSAHIConfig | None = None,
    ) -> ObjectDetectionPrediction:
        """Turn the decoded rows of one image into its final prediction.

        Args:
            batch_prediction:
                The ``metadata.num_rows`` rows belonging to a single image, as
                returned by :meth:`ObjectDetectionBatchPrediction.select_rows`.
            metadata: The image's metadata as returned by the preprocessor.
            threshold: Detections with a score <= threshold are discarded.
            sahi_config:
                Merge settings, required when the image was tiled.

        Raises:
            ValueError: If the image was tiled but no ``sahi_config`` is given.
        """
        if metadata.tile_coordinates is None:
            prediction = ObjectDetectionPrediction(
                labels=batch_prediction.labels[0],
                bboxes=batch_prediction.bboxes[0],
                scores=batch_prediction.scores[0],
            )
            return prediction[prediction.scores > threshold]

        if sahi_config is None:
            raise ValueError(
                "Metadata contains tile coordinates but no sahi_config was given to "
                "merge the tile predictions."
            )
        return combine_sahi_object_detection_predictions(
            batch_prediction=batch_prediction,
            tile_coordinates=metadata.tile_coordinates,
            threshold=threshold,
            nms_iou_threshold=sahi_config.nms_iou_threshold,
            global_local_iou_threshold=sahi_config.global_local_iou_threshold,
        )

    def postprocess(
        self,
        raw: ObjectDetectionBatchOutput,
        metadata: Sequence[ObjectDetectionMetadata],
        threshold: float,
        *,
        sahi_config: ObjectDetectionSAHIConfig | None = None,
    ) -> list[ObjectDetectionPrediction]:
        """Decode raw outputs into one prediction per image.

        Convenience wrapper that runs :meth:`postprocess_batch` once and then
        :meth:`postprocess_image` on each image's slice of the result.

        Args:
            raw: Raw model output for all rows in the batch.
            metadata: Per-image metadata as returned by the preprocessor.
            threshold: Detections with a score <= threshold are discarded.
            sahi_config:
                Merge settings, required when any image in ``metadata`` was tiled.
        """
        batch_prediction = self.postprocess_batch(raw, metadata)
        predictions = []
        start = 0
        for item in metadata:
            end = start + item.num_rows
            predictions.append(
                self.postprocess_image(
                    batch_prediction.select_rows(start, end),
                    item,
                    threshold,
                    sahi_config=sahi_config,
                )
            )
            start = end
        return predictions

    def _target_sizes(
        self,
        metadata: Sequence[ObjectDetectionMetadata],
        *,
        device: torch.device,
    ) -> Tensor:
        """Return the ``(width, height)`` each row's boxes are scaled to.

        The row for the image itself is scaled back to the original image, while tile
        rows stay in tile coordinates and are offset when the tiles are merged.
        """
        tile_h, tile_w = self.image_size
        sizes: list[list[int]] = []
        for item in metadata:
            sizes.append([item.orig_w, item.orig_h])
            if item.tile_coordinates is not None:
                sizes.extend(
                    [tile_w, tile_h] for _ in range(len(item.tile_coordinates))
                )
        return torch.tensor(sizes, dtype=torch.int64, device=device)
