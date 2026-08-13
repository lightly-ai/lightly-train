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
from typing import TypedDict, cast

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import Module
from torchvision.ops import (
    batched_nms,
    box_convert,
    box_iou,
    clip_boxes_to_image,
    remove_small_boxes,
)
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import Self

from lightly_train._data import file_helpers
from lightly_train._pre_post_processing.tiling import tile_image
from lightly_train._task_models.task_model_io import (
    BaseModelOutput,
    RowIndexableOutput,
)
from lightly_train.types import PathLike


class ObjectDetectionTorchmetricsPrediction(TypedDict):
    """Single-image prediction in the format expected by TorchMetrics detection
    metrics (for example ``MeanAveragePrecision``).

    Attributes:
        boxes: Shape ``(N, 4)``. ``xyxy`` boxes in original-image pixel coordinates.
        scores: Shape ``(N,)``. Confidence score in ``[0, 1]`` for each detection.
        labels: Shape ``(N,)``, ``int64``. Predicted class id for each detection.
    """

    boxes: Tensor
    scores: Tensor
    labels: Tensor


class ObjectDetectionTorchmetricsTarget(TypedDict):
    """Single-image ground truth in the format expected by TorchMetrics detection
    metrics (for example ``MeanAveragePrecision``).

    Attributes:
        boxes: Shape ``(N, 4)``. ``xyxy`` boxes in original-image pixel coordinates.
        labels: Shape ``(N,)``, ``int64``. Ground truth class id for each box.
    """

    boxes: Tensor
    labels: Tensor


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

    def to_torchmetrics(self) -> ObjectDetectionTorchmetricsPrediction:
        """Convert to a format compatible with TorchMetrics.

        Returns:
            A dictionary with keys ``"boxes"``, ``"scores"``, and ``"labels"``.
        """
        return {
            "boxes": self.bboxes,
            "scores": self.scores,
            "labels": self.labels,
        }

    def offset_boxes(self, xy: Tensor) -> Self:
        """Return a prediction with the boxes shifted by ``xy``.

        Args:
            xy: Shape ``(N, 2)`` or ``(2,)``. ``(x, y)`` offset added to every corner.
        """
        xy = xy.to(self.bboxes.device)
        return type(self)(
            labels=self.labels,
            bboxes=self.bboxes + torch.cat([xy, xy], dim=-1),
            scores=self.scores,
        )

    def clip_to_image(self, *, height: int, width: int) -> Self:
        """Return the detections clipped to the image rectangle, dropping empty boxes.

        Boxes are clamped into ``[0, width] x [0, height]``, and detections whose
        clipped box has zero or negative area are dropped: a box lying entirely outside
        the rectangle carries no information, and a zero-area box would survive NMS (its
        IoU with everything is 0) and pollute the output.

        SAHI needs this because a tile is zero padded when the image is smaller than a
        tile in a dimension (see
        :func:`~lightly_train._pre_post_processing.tiling.tile_image`), so a tile
        detection can sit partly or wholly inside that padding, outside the image.

        Args:
            height: Height of the image in pixels.
            width: Width of the image in pixels.
        """
        limits = self.bboxes.new_tensor([width, height, width, height])
        bboxes = self.bboxes.clamp(min=0.0).minimum(limits)
        clipped = type(self)(labels=self.labels, bboxes=bboxes, scores=self.scores)
        keep = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
        return clipped[keep]

    def apply_nms(self, iou_threshold: float) -> Self:
        """Return the detections surviving class-aware non-maximum suppression.

        Suppression is class-aware so that a high-confidence detection cannot hide a
        detection of another class.
        """
        keep: Tensor = batched_nms(self.bboxes, self.scores, self.labels, iou_threshold)
        return self[keep]

    def drop_overlapping_predictions(
        self, other: ObjectDetectionPrediction, iou_threshold: float
    ) -> Self:
        """Drop detections that overlap a same-label detection in ``other``.

        Args:
            other: The detections to compare against. Never modified.
            iou_threshold:
                IoU above which a detection is dropped if ``other`` holds a detection
                of the same label.
        """
        ious: Tensor = box_iou(self.bboxes, other.bboxes)
        # The same-label check must be applied before reducing over ``other``:
        # reducing first (e.g. via the single max-IoU box) would miss a same-label
        # overlap that is not the strongest one.
        same_label = self.labels[:, None] == other.labels[None, :]
        overlaps_same_label = (same_label & (ious > iou_threshold)).any(dim=1)
        return self[~overlaps_same_label]

    def drop_contained_predictions(self, regions: Tensor) -> Self:
        """Drop detections whose box lies entirely inside one of ``regions``.

        Purely geometric: unlike :meth:`drop_overlapping_predictions` this ignores
        labels and scores. Used to drop the global-view boxes of objects that fit
        inside a single SAHI tile, because the tiles saw those objects at a higher
        resolution and localize them better.

        Args:
            regions:
                Shape ``(M, 4)``. ``xyxy`` regions in the same coordinates as
                :attr:`bboxes`. May contain infinities.
        """
        regions = regions.to(self.bboxes.device)
        # Corner comparisons rather than intersection areas, so infinite regions stay
        # well defined. Index 0 of the last dimension holds both x conditions and index
        # 1 both y conditions, so ``all`` over it requires all four corners to be in.
        inside = (self.bboxes[:, None, :2] >= regions[None, :, :2]) & (
            self.bboxes[:, None, 2:] <= regions[None, :, 2:]
        )
        contained = inside.all(dim=-1).any(dim=1)
        return self[~contained]

    def map_labels(self, mapping: Tensor) -> Self:
        """Return a prediction with each label replaced by ``mapping[label]``.

        Used to translate internal, contiguous class ids (``0..num_classes - 1``,
        the model's output space) to the user-facing class ids stored in the task
        checkpoint, via :attr:`ObjectDetectionPostprocessor.internal_class_to_class`.

        Args:
            mapping:
                Shape ``(num_classes,)``, ``int64``. ``mapping[i]`` is the
                user-facing class id for internal class ``i``.
        """
        return type(self)(
            labels=mapping[self.labels], bboxes=self.bboxes, scores=self.scores
        )

    @classmethod
    def concat(cls, predictions: Sequence[ObjectDetectionPrediction]) -> Self:
        """Concatenate several single-image predictions into one.

        Used to merge the global-view and tile detections of a tiled (SAHI) image
        back into a single prediction, after they have already been filtered and
        deduplicated. Detections are kept in the given order; nothing else is
        deduplicated or sorted here.

        Args:
            predictions: Predictions to concatenate, all for the same image.
        """
        return cls(
            labels=torch.cat([prediction.labels for prediction in predictions], dim=0),
            bboxes=torch.cat([prediction.bboxes for prediction in predictions], dim=0),
            scores=torch.cat([prediction.scores for prediction in predictions], dim=0),
        )


@dataclass
class ObjectDetectionBatchPrediction(RowIndexableOutput):
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
    image contributes one global row plus one row per tile. Row selection
    (``batch_prediction[1:]``) therefore selects model input rows, not detections.
    """

    labels: Tensor
    bboxes: Tensor
    scores: Tensor

    def split(self, num_rows: Sequence[int]) -> list[ObjectDetectionBatchPrediction]:
        """Split rows into consecutive groups of the given sizes.

        Used to regroup the batch per image: an image consumes
        :attr:`ObjectDetectionMetadata.num_rows` consecutive rows, one for the image
        itself plus one per SAHI tile.
        """
        predictions = []
        start = 0
        for rows in num_rows:
            end = start + rows
            predictions.append(
                ObjectDetectionBatchPrediction(
                    labels=self.labels[start:end],
                    bboxes=self.bboxes[start:end],
                    scores=self.scores[start:end],
                )
            )
            start = end
        return predictions

    def row(self, index: int) -> ObjectDetectionPrediction:
        """Return a single row as a flat :class:`ObjectDetectionPrediction`."""
        return ObjectDetectionPrediction(
            labels=self.labels[index],
            bboxes=self.bboxes[index],
            scores=self.scores[index],
        )

    def offset_rows(self, xy: Tensor) -> Self:
        """Return a prediction with each row's boxes shifted by its own offset.

        Args:
            xy: Shape ``(B, 2)``. ``(x, y)`` offset for each row, broadcast over ``K``.
        """
        xy = xy.to(self.bboxes.device)
        offsets = torch.cat([xy, xy], dim=-1).unsqueeze(1)
        return type(self)(
            labels=self.labels, bboxes=self.bboxes + offsets, scores=self.scores
        )

    def flatten(self) -> ObjectDetectionPrediction:
        """Collapse the row dimension: ``(B, K, ...)`` becomes ``(B * K, ...)``."""
        return ObjectDetectionPrediction(
            labels=self.labels.flatten(),
            bboxes=self.bboxes.flatten(0, 1),
            scores=self.scores.flatten(),
        )

    def merge_tiles(
        self,
        tiling: ObjectDetectionTiling,
        *,
        threshold: float,
        orig_h: int,
        orig_w: int,
    ) -> ObjectDetectionPrediction:
        """Merge the rows of one tiled image into a single prediction.

        Row 0 holds the global view in original-image coordinates, rows ``1:`` hold the
        tiles. A tile's boxes are decoded against the size of the image region that tile
        covers (:attr:`ObjectDetectionTiling.tile_size`), so shifting them by the tile's
        top-left corner is the whole transform and no scale is involved even when the
        tiles were magnified. Both halves are score filtered and clipped to the original
        image, and then the two views divide the work by object size:

        - The global view is responsible only for objects that no single tile saw whole.
          Every global box that fits inside some tile is dropped, because for those
          objects the tiles saw the same object at a higher resolution and localize it
          far better, while the downscaled global view tends to return a coarse box with
          a high score that would otherwise evict the accurate tile box.
        - The tiles are responsible for everything else. Overlapping tile boxes are
          deduplicated with class-aware NMS (needed because tiles overlap), and a tile
          box that repeats a surviving global box of the same label is dropped: those
          global boxes belong to objects larger than a tile, so an overlapping tile box
          is a fragment of one rather than a competing detection.

        Clipping happens before the NMS and before dropping fragments so that every IoU
        is computed on in-image boxes. A tile is zero padded when the image is smaller
        than a tile in a dimension, and a detection reaching into that padding covers
        area that exists nowhere in the image: comparing it un-clipped would let it
        suppress the real detection it overlaps.

        Args:
            tiling: The tiling record from the image's metadata.
            threshold: Detections with a score <= threshold are discarded.
            orig_h: Height of the original image in pixels.
            orig_w: Width of the original image in pixels.
        """
        global_row = self.row(0)
        global_prediction = global_row[global_row.scores > threshold].clip_to_image(
            height=orig_h, width=orig_w
        )
        if tiling.num_tiles > 1:
            # A single tile spans the whole image, so it has no field-of-view advantage
            # over the global view and there is nothing to arbitrate: both views are
            # entitled to every object. The preprocessor never produces a single tile
            # (it skips tiling entirely in that case), but merge_tiles should not depend
            # on that to stay correct.
            global_prediction = global_prediction.drop_contained_predictions(
                tiling.tile_boxes
            )
        tile_flat = self[1:].offset_rows(tiling.coordinates).flatten()
        tile_prediction = tile_flat[tile_flat.scores > threshold].clip_to_image(
            height=orig_h, width=orig_w
        )
        tile_prediction = tile_prediction.apply_nms(
            tiling.nms_iou_threshold
        ).drop_overlapping_predictions(
            global_prediction, tiling.global_local_iou_threshold
        )
        merged = ObjectDetectionPrediction.concat([global_prediction, tile_prediction])
        # Untiled predictions come out of top-k in descending score order; sorting here
        # gives the tiled path the same guarantee instead of "global first, then tiles".
        return merged[torch.argsort(merged.scores, descending=True, stable=True)]

    def to_predictions(self) -> list[ObjectDetectionPrediction]:
        """Return one :class:`ObjectDetectionPrediction` per row."""
        return [
            ObjectDetectionPrediction(labels=labels, bboxes=bboxes, scores=scores)
            for labels, bboxes, scores in zip(self.labels, self.bboxes, self.scores)
        ]

    def to_torchmetrics_list(self) -> list[ObjectDetectionTorchmetricsPrediction]:
        """Return one TorchMetrics-compatible dict per row.

        Convenience for callers (for example metrics computation) that want
        :meth:`to_predictions` and :meth:`ObjectDetectionPrediction.to_torchmetrics`
        chained together.
        """
        return [prediction.to_torchmetrics() for prediction in self.to_predictions()]


@dataclass(frozen=True)
class ObjectDetectionSAHIConfig:
    """User-facing SAHI settings, consumed by :class:`ObjectDetectionPreprocessor`.

    Only :attr:`tile_size` and :attr:`overlap` affect tiling itself; the two IoU
    thresholds describe how the tiles are merged again and are recorded on the metadata
    the preprocessor returns, so the postprocessor does not need this config.

    Attributes:
        overlap: Fractional overlap between neighbouring tiles, in ``[0, 1)``.
        nms_iou_threshold: IoU used in the class-aware NMS of the tile predictions.
        global_local_iou_threshold:
            IoU above which a tile box is dropped as a fragment of a same-label global
            box. Global boxes only survive the merge for objects too large for a tile,
            so this only ever removes fragments of those.
        tile_size:
            ``(height, width)`` of the image region each tile covers, in original-image
            pixels. Tiles are cut at this size and then resized to the model's input
            size, so a value below the model input magnifies the tiles, which is what
            makes small objects detectable. ``None`` means half the model input size.
    """

    overlap: float
    nms_iou_threshold: float
    global_local_iou_threshold: float
    tile_size: tuple[int, int] | None = None


@dataclass(frozen=True)
class ObjectDetectionTiling:
    """Record of how the preprocessor tiled one image, and how to merge it back.

    Attributes:
        coordinates:
            Shape ``(num_tiles, 2)``. Top-left ``(x, y)`` of each tile, in
            original-image pixels.
        tile_size:
            ``(height, width)`` of the image region each tile covers, in original-image
            pixels. This is the size the tile's boxes are decoded against, which is what
            lets tiles be magnified: a tile may be fed to the model at a different
            resolution, but its boxes always scale back to the region it covers.
        nms_iou_threshold: IoU used in NMS of the tile predictions.
        global_local_iou_threshold:
            IoU above which a tile box is dropped if it matches a global box of the
            same label.
    """

    coordinates: Tensor
    tile_size: tuple[int, int]
    nms_iou_threshold: float
    global_local_iou_threshold: float

    @property
    def num_tiles(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def tile_boxes(self) -> Tensor:
        """Shape ``(num_tiles, 4)``. The ``xyxy`` region each tile covers.

        In original-image coordinates, so these are directly comparable with the boxes
        :meth:`ObjectDetectionBatchPrediction.merge_tiles` works on.

        Edges lying on the image border are pushed out to infinity. Predicted boxes may
        stick out of the image, and a box that only leaves the image through the border
        is still an object the tile saw whole; without this a detection at the image
        edge would look like it crosses a tile seam.
        """
        tile_h, tile_w = self.tile_size
        top_left = self.coordinates.to(torch.float32)
        bottom_right = top_left + top_left.new_tensor([tile_w, tile_h])
        # amin/amax are per coordinate over tiles, so a tile's left edge is extended only
        # if that tile is in the first column, its top edge only if it is in the first
        # row, and so on. Interior seams are left untouched.
        top_left = torch.where(top_left <= top_left.amin(dim=0), -torch.inf, top_left)
        bottom_right = torch.where(
            bottom_right >= bottom_right.amax(dim=0), torch.inf, bottom_right
        )
        return torch.cat([top_left, bottom_right], dim=-1)


@dataclass(frozen=True)
class ObjectDetectionMetadata:
    """What the preprocessor did to one image, so the postprocessor can undo it.

    Attributes:
        orig_h: Height of the original image in pixels.
        orig_w: Width of the original image in pixels.
        tiling:
            The tiling record, or ``None`` if the image was not tiled -- either because
            no SAHI config was given, or because the image fits inside a single tile.
    """

    orig_h: int
    orig_w: int
    tiling: ObjectDetectionTiling | None = None

    @property
    def num_rows(self) -> int:
        """Number of model input rows this image occupies.

        One row for the image itself, plus one row per SAHI tile.
        """
        if self.tiling is None:
            return 1
        return 1 + self.tiling.num_tiles

    @property
    def row_sizes(self) -> list[tuple[int, int]]:
        """The ``(width, height)`` of the image region each of this image's rows covers.

        This is what the decode multiplies a row's normalized boxes by, so it answers the
        only question postprocessing has per row: which rectangle of the original image
        does this model input show? Row 0 is the global row and covers the whole image.
        A tile row covers ``tiling.tile_size`` pixels of the original image, which is
        independent of the resolution the tile was fed to the model at -- that is what
        lets tiles be magnified -- and is offset by the tile's corner when the tiles are
        merged.
        """
        sizes = [(self.orig_w, self.orig_h)]
        if self.tiling is not None:
            tile_h, tile_w = self.tiling.tile_size
            sizes += [(tile_w, tile_h)] * self.tiling.num_tiles
        return sizes


def decode_object_detection_output(
    *,
    raw: ObjectDetectionBatchOutput,
    target_sizes: Tensor,
    num_top_queries: int,
) -> ObjectDetectionBatchPrediction:
    """Decode raw detection outputs into dense per-row predictions.

    Purely geometric: selects the top-scoring ``(query, class)`` pairs and converts
    their boxes to pixel coordinates. Labels stay in the internal, contiguous class-id
    space (positional index into the classes the model was built with); mapping them
    to user-facing class ids is the caller's responsibility, done once on the final,
    filtered prediction (see :meth:`ObjectDetectionPostprocessor.postprocess_image`).

    Args:
        raw:
            The model's :class:`ObjectDetectionBatchOutput`, holding raw (pre-sigmoid)
            logits of shape ``(B, num_queries, num_classes)`` and normalized ``cxcywh``
            boxes of shape ``(B, num_queries, 4)``. Note that ``B`` counts model input
            rows, not input images: with SAHI a single image contributes one global row
            plus one row per tile.
        target_sizes: Shape ``(B, 2)``. ``(width, height)`` the boxes are scaled to.
        num_top_queries: Number of ``(query, class)`` pairs kept per row.

    Returns:
        An :class:`ObjectDetectionBatchPrediction` holding ``num_top_queries``
        detections per row with ``xyxy`` boxes in ``target_sizes`` coordinates and
        internal-id ``labels``. No score thresholding is applied.
    """
    scores = raw.logits.sigmoid()
    num_classes = scores.shape[-1]
    scores, index = scores.flatten(1).topk(num_top_queries, dim=-1)
    labels = index % num_classes
    query_index = index // num_classes
    boxes = box_convert(raw.boxes, in_fmt="cxcywh", out_fmt="xyxy")
    boxes = boxes.gather(1, query_index.unsqueeze(-1).expand(-1, -1, 4))
    boxes = boxes * target_sizes.repeat(1, 2).unsqueeze(1)
    return ObjectDetectionBatchPrediction(labels=labels, bboxes=boxes, scores=scores)


def _target_sizes(
    metadata: Sequence[ObjectDetectionMetadata], *, device: torch.device
) -> Tensor:
    """Return the ``(width, height)`` each row in the batch is scaled to.

    Concatenates :attr:`ObjectDetectionMetadata.row_sizes` over all images, so the
    result has one row per model input row.
    """
    sizes = [size for item in metadata for size in item.row_sizes]
    return torch.tensor(sizes, dtype=torch.int64, device=device)


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
) -> list[ObjectDetectionTorchmetricsTarget]:
    """Convert ground truth boxes into a format compatible with TorchMetrics.

    This is the ground truth counterpart to
    :meth:`ObjectDetectionPrediction.to_torchmetrics`: it brings targets into the
    same ``xyxy`` original-image pixel coordinates the predictions use. Boxes are
    also clipped to the image canvas and degenerate (zero-area) boxes are dropped,
    matching the ``BboxParams(clip=True, filter_invalid_bboxes=True)`` behavior of
    the training/val transforms.

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

    targets = []
    for boxes, labels, (width, height) in zip(
        boxes_denormalized, classes, original_sizes
    ):
        boxes = clip_boxes_to_image(boxes, size=(height, width))
        keep = remove_small_boxes(boxes, min_size=1e-6)
        targets.append({"boxes": boxes[keep], "labels": labels[keep]})
    return targets


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
                If ``None``, the image is resized to ``image_size``. Otherwise the image
                is tiled and the resized global image is prepended to the tiles, with an
                :class:`ObjectDetectionTiling` recorded on the metadata. The config is
                not needed again afterwards: everything the postprocessor needs to merge
                the tiles is on that record. Images that fit inside a single tile are not
                tiled even when a config is given: the metadata then has ``tiling=None``
                and a single row, see :meth:`_fits_in_one_tile`.

        Returns:
            A ``(num_rows, C, H, W)`` stack and the image's metadata. Without
            ``sahi_config`` there is a single row, otherwise there is one global row
            followed by one row per tile, or a single row if the image fits inside one
            tile. The row count is always ``metadata.num_rows``.
        """
        image_tensor = file_helpers.as_image_tensor(image).to(device)
        orig_h, orig_w = image_tensor.shape[-2:]
        image_tensor = self._to_expected_channels(image_tensor)
        image_tensor = transforms_functional.to_dtype(
            image_tensor, dtype=dtype, scale=True
        )
        global_image = transforms_functional.resize(
            image_tensor, self.image_size
        ).unsqueeze(0)

        rows = global_image
        image_tiling: ObjectDetectionTiling | None = None
        if sahi_config is not None:
            tile_size = self._tile_size(sahi_config)
            if not self._fits_in_one_tile(tile_size, orig_h=orig_h, orig_w=orig_w):
                tiles, coordinates = tile_image(
                    image=image_tensor,
                    overlap=sahi_config.overlap,
                    tile_size=tile_size,
                )
                # The tiles cover ``tile_size`` pixels of the original image but the
                # model takes ``image_size``, so resizing them here is what magnifies the
                # tiles when ``tile_size`` is smaller than the model input. Their boxes
                # are decoded against ``tile_size`` (recorded below and reported by
                # ``ObjectDetectionMetadata.row_sizes``), so the scale is undone for free.
                if tile_size != self.image_size:
                    tiles = transforms_functional.resize(tiles, list(self.image_size))
                rows = torch.cat([global_image, tiles])
                image_tiling = ObjectDetectionTiling(
                    coordinates=coordinates,
                    tile_size=tile_size,
                    nms_iou_threshold=sahi_config.nms_iou_threshold,
                    global_local_iou_threshold=sahi_config.global_local_iou_threshold,
                )

        return rows, ObjectDetectionMetadata(
            orig_h=orig_h, orig_w=orig_w, tiling=image_tiling
        )

    def _tile_size(self, sahi_config: ObjectDetectionSAHIConfig) -> tuple[int, int]:
        """Resolve the SAHI tile size in original-image pixels.

        ``None`` means half the model input size, so tiles are magnified 2x by default:
        cutting tiles at the model input size makes them native-resolution crops, which
        magnifies nothing and leaves small objects exactly as small as they were.
        """
        if sahi_config.tile_size is not None:
            return sahi_config.tile_size
        tile_h, tile_w = self.image_size
        return max(1, tile_h // 2), max(1, tile_w // 2)

    def _fits_in_one_tile(
        self, tile_size: tuple[int, int], *, orig_h: int, orig_w: int
    ) -> bool:
        """Whether tiling would add nothing over the global row.

        An image no larger than one tile yields a single tile: the zero-padded original
        at native resolution. The global row already carries the same pixels, resized to
        the size the model was trained on, so the tile magnifies nothing -- it shows the
        same content at a *smaller* apparent object size, which is the opposite of what
        SAHI is for. It would only cost a model input row plus a near-duplicate detection
        to deduplicate, and when the image is exactly one tile the two rows are identical.
        """
        tile_h, tile_w = tile_size
        return orig_h <= tile_h and orig_w <= tile_w

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
            sahi_config: If given, every image larger than a single tile is tiled.

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
        num_top_queries: int,
        internal_class_to_class: Tensor,
    ) -> None:
        super().__init__()
        self.num_top_queries = num_top_queries
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

        Returned labels are internal, contiguous class ids; :meth:`postprocess_image`
        remaps them to user-facing class ids on the final, filtered prediction.

        Args:
            raw: Raw model output for all rows in the batch.
            metadata: Per-image metadata as returned by the preprocessor.
        """
        target_sizes = _target_sizes(metadata, device=raw.boxes.device)
        return decode_object_detection_output(
            raw=raw,
            target_sizes=target_sizes,
            num_top_queries=self.num_top_queries,
        )

    def postprocess_image(
        self,
        batch_prediction: ObjectDetectionBatchPrediction,
        metadata: ObjectDetectionMetadata,
        threshold: float,
    ) -> ObjectDetectionPrediction:
        """Turn the decoded rows of one image into its final prediction.

        The metadata alone says whether the image was tiled and, if so, how to merge
        the tiles again, so no separate SAHI config is needed here.

        Args:
            batch_prediction:
                The ``metadata.num_rows`` rows belonging to a single image, as
                returned by :meth:`ObjectDetectionBatchPrediction.split`.
            metadata: The image's metadata as returned by the preprocessor.
            threshold: Detections with a score <= threshold are discarded.
        """
        if metadata.tiling is None:
            row = batch_prediction.row(0)
            prediction = row[row.scores > threshold]
        else:
            prediction = batch_prediction.merge_tiles(
                metadata.tiling,
                threshold=threshold,
                orig_h=metadata.orig_h,
                orig_w=metadata.orig_w,
            )

        # This is the only place internal class ids are mapped to user-facing class
        # ids: doing it last, on the filtered (and, for SAHI, merged) prediction,
        # keeps the dense batch stage and the tile-merging same-label comparisons
        # (which only need consistency, not any particular id space) working in the
        # cheaper internal-id space.
        return prediction.map_labels(self.internal_class_to_class)

    def postprocess(
        self,
        raw: ObjectDetectionBatchOutput,
        metadata: Sequence[ObjectDetectionMetadata],
        threshold: float,
    ) -> list[ObjectDetectionPrediction]:
        """Decode raw outputs into one prediction per image.

        Convenience wrapper that runs :meth:`postprocess_batch` once and then
        :meth:`postprocess_image` on each image's slice of the result.

        Args:
            raw: Raw model output for all rows in the batch.
            metadata: Per-image metadata as returned by the preprocessor.
            threshold: Detections with a score <= threshold are discarded.
        """
        batch_prediction = self.postprocess_batch(raw, metadata)
        num_rows = [item.num_rows for item in metadata]
        return [
            self.postprocess_image(item_prediction, item, threshold)
            for item_prediction, item in zip(batch_prediction.split(num_rows), metadata)
        ]
