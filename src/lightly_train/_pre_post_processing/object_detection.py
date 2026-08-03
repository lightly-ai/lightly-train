#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn.functional as F
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import Module
from torchvision.ops import box_convert
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import NotRequired

from lightly_train._data import file_helpers
from lightly_train._task_models.object_detection_components import tiling_utils
from lightly_train._task_models.task_model_io import BaseModelOutput
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
class ObjectDetectionPreprocessedBatch(BaseModelOutput):
    """A dense model-input batch and its ``(width, height)`` target sizes.

    Attributes:
        images: Shape ``(B, C, H, W)``. Resized, normalized model-input batch, ready
            to pass to the model's forward pass.
        target_sizes: Shape ``(B, 2)``, ``int64``. Original ``(width, height)`` of
            each image in pixels, used to rescale decoded boxes back to
            original-image coordinates.
    """

    images: Tensor
    target_sizes: Tensor


@dataclass
class ObjectDetectionSahiPreprocessedBatch(BaseModelOutput):
    """Flattened global/tile inputs and tensor-only SAHI reconstruction data.

    For each input image this holds one "global" (resized full image) entry followed
    by one entry per tile, all flattened into a single leading dimension ``N`` (the
    total number of global + tile entries across every image in the batch).

    Attributes:
        images: Shape ``(N, C, tile_h, tile_w)``. Resized, normalized model input for
            every global view and tile, flattened across all images.
        target_sizes: Shape ``(N, 2)``, ``int64``. Per-entry ``(width, height)``
            rescale target: the original image size for the global entry, and the
            tile size ``(tile_w, tile_h)`` for tile entries.
        tile_offsets: Shape ``(num_images + 1,)``, ``int64``. CSR-style row pointers
            into the ``N`` dimension: entries ``tile_offsets[i]:tile_offsets[i + 1]``
            belong to image ``i`` (its global view first, followed by its tiles).
        tile_coordinates: Shape ``(N, 2)``, ``int64``. Per-entry ``(x, y)`` pixel
            offset of the tile's top-left corner within the (possibly upscaled)
            original image; ``(0, 0)`` for each image's global entry.
    """

    images: Tensor
    target_sizes: Tensor
    tile_offsets: Tensor
    tile_coordinates: Tensor


@dataclass
class ObjectDetectionDecodedBatch(BaseModelOutput):
    """Dense top-k detections before score filtering or SAHI merging.

    Unlike :class:`ObjectDetectionPrediction`, this is a fixed-size, unfiltered dense
    batch: every entry has exactly ``num_top_queries`` detections, including
    low-confidence ones. ``B`` is whatever leading batch dimension the raw model
    output had when passed to :meth:`ObjectDetectionPostprocessor.decode` — this is
    the number of images for a plain batch, or the number of flattened global+tile
    entries for a SAHI batch.

    Attributes:
        labels: Shape ``(B, num_top_queries)``, ``int64``. Predicted class indices
            (already mapped through ``internal_class_to_class``) for the top-k
            queries of each batch entry.
        bboxes: Shape ``(B, num_top_queries, 4)``. ``xyxy`` boxes rescaled into the
            coordinate space given by the ``target_sizes`` passed to ``decode``.
        scores: Shape ``(B, num_top_queries)``. Confidence score (post-sigmoid) for
            each of the top-k queries.
    """

    labels: Tensor
    bboxes: Tensor
    scores: Tensor


@dataclass
class ObjectDetectionPrediction(BaseModelOutput):
    """Predictions for one image in original-image ``xyxy`` coordinates.

    Unlike :class:`ObjectDetectionDecodedBatch`, this is per-image and already
    filtered by score threshold (and, for SAHI, NMS-merged across global/tile
    detections), so ``N`` varies per image and can be smaller than
    ``num_top_queries``.

    Attributes:
        labels: Shape ``(N,)``, ``int64``. Predicted class index for each of the
            ``N`` kept detections.
        bboxes: Shape ``(N, 4)``. ``xyxy`` boxes in original-image pixel coordinates.
        scores: Shape ``(N,)``. Confidence score in ``[0, 1]`` for each detection.
    """

    labels: Tensor
    bboxes: Tensor
    scores: Tensor


class ObjectDetectionMetadata(TypedDict):
    orig_h: int
    orig_w: int
    tiles_coordinates: NotRequired[Tensor]


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
        self.expected_input_channels = expected_input_channels
        if image_normalize is None:
            self.register_buffer("image_mean", None, persistent=False)
            self.register_buffer("image_std", None, persistent=False)
        else:
            self.register_buffer(
                "image_mean", torch.tensor(image_normalize["mean"]), persistent=False
            )
            self.register_buffer(
                "image_std", torch.tensor(image_normalize["std"]), persistent=False
            )

    def preprocess_images(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ObjectDetectionPreprocessedBatch:
        """Load, bucket, resize, and normalize images while preserving order."""
        if not images:
            raise ValueError("images must contain at least one image.")
        tensors = [
            self._validate_channels(file_helpers.as_image_tensor(image).to(device))
            for image in images
        ]
        target_sizes = torch.tensor(
            [[tensor.shape[-1], tensor.shape[-2]] for tensor in tensors],
            dtype=torch.int64,
            device=device,
        )
        processed: list[Tensor | None] = [None] * len(tensors)
        for indices in self._bucket_indices(tensors).values():
            batch = torch.stack([tensors[index] for index in indices])
            batch = transforms_functional.to_dtype(batch, dtype=dtype, scale=True)
            batch = transforms_functional.resize(batch, self.image_size)
            for index, image in zip(indices, batch):
                processed[index] = image
        return ObjectDetectionPreprocessedBatch(
            images=self.preprocess_batch(
                torch.stack([image for image in processed if image is not None])
            ),
            target_sizes=target_sizes,
        )

    def preprocess_sahi_images(
        self,
        images: Sequence[PathLike | PILImage | Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        overlap: float,
    ) -> ObjectDetectionSahiPreprocessedBatch:
        """Prepare every global view and tile in one flattened model-input batch."""
        if not images:
            raise ValueError("images must contain at least one image.")
        tensors = [
            self._validate_channels(file_helpers.as_image_tensor(image).to(device))
            for image in images
        ]
        prepared: list[tuple[Tensor, Tensor, Tensor, Tensor] | None] = [None] * len(
            tensors
        )
        for indices in self._bucket_indices(tensors).values():
            batch = torch.stack([tensors[index] for index in indices])
            batch = transforms_functional.to_dtype(batch, dtype=dtype, scale=True)
            group_image_batches, group_target_sizes, group_coordinates = (
                self._tile_batch(batch, overlap)
            )
            for position, index in enumerate(indices):
                prepared[index] = (
                    group_image_batches[position],
                    group_target_sizes[position],
                    group_coordinates,
                    torch.tensor(group_image_batches[position].shape[0], device=device),
                )

        image_batches: list[Tensor] = []
        target_sizes: list[Tensor] = []
        coordinates: list[Tensor] = []
        offsets = [0]
        for item in prepared:
            assert item is not None
            image_batch, sizes, coords, count = item
            image_batches.append(image_batch)
            target_sizes.append(sizes)
            coordinates.append(coords)
            offsets.append(offsets[-1] + int(count))
        return ObjectDetectionSahiPreprocessedBatch(
            images=self.preprocess_sahi_batch(torch.cat(image_batches)),
            target_sizes=torch.cat(target_sizes),
            tile_offsets=torch.tensor(offsets, dtype=torch.int64, device=device),
            tile_coordinates=torch.cat(coordinates),
        )

    # Compatibility helpers for external callers of the recently extracted class.
    def preprocess_image(
        self,
        image: PathLike | PILImage | Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, ObjectDetectionMetadata]:
        output = self.preprocess_images([image], device=device, dtype=dtype)
        width, height = output.target_sizes[0].tolist()
        return output.images[0], {"orig_h": height, "orig_w": width}

    def preprocess_sahi_image(
        self,
        image: PathLike | PILImage | Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        overlap: float,
    ) -> tuple[Tensor, ObjectDetectionMetadata]:
        output = self.preprocess_sahi_images(
            [image], device=device, dtype=dtype, overlap=overlap
        )
        width, height = output.target_sizes[0].tolist()
        return output.images, {
            "orig_h": height,
            "orig_w": width,
            "tiles_coordinates": output.tile_coordinates[1:],
        }

    def preprocess_sahi_batch(self, batch: Tensor) -> Tensor:
        return self.preprocess_batch(batch)

    def _validate_channels(self, image: Tensor) -> Tensor:
        if image.shape[-3] == 1 and self.expected_input_channels > 1:
            return image.expand(self.expected_input_channels, -1, -1)
        if image.shape[-3] != self.expected_input_channels:
            raise ValueError(
                f"Image has {image.shape[-3]} channels but model expects {self.expected_input_channels}."
            )
        return image

    @staticmethod
    def _bucket_indices(
        tensors: Sequence[Tensor],
    ) -> dict[tuple[object, ...], list[int]]:
        buckets: dict[tuple[object, ...], list[int]] = defaultdict(list)
        for index, tensor in enumerate(tensors):
            buckets[(tensor.device, tensor.dtype, *tensor.shape)].append(index)
        return buckets

    def _tile_batch(
        self, batch: Tensor, overlap: float
    ) -> tuple[list[Tensor], list[Tensor], Tensor]:
        if not 0.0 <= overlap < 1.0:
            raise ValueError("overlap must be in the range [0.0, 1.0).")
        tile_h, tile_w = self.image_size
        _, _, height, width = batch.shape
        original_height, original_width = height, width
        tile_source = batch
        if height < tile_h or width < tile_w:
            scale = max(tile_h / height, tile_w / width)
            tile_source = F.interpolate(
                batch,
                size=(math.ceil(height * scale), math.ceil(width * scale)),
                mode="bilinear",
                align_corners=False,
            )
            _, _, height, width = tile_source.shape
        h_step = max(1, int((1.0 - overlap) * tile_h))
        w_step = max(1, int((1.0 - overlap) * tile_w))
        h_starts = self._tile_starts(height, tile_h, h_step)
        w_starts = self._tile_starts(width, tile_w, w_step)
        tiles = torch.stack(
            [
                tile_source[:, :, h : h + tile_h, w : w + tile_w]
                for h in h_starts
                for w in w_starts
            ],
            dim=1,
        )
        global_images = transforms_functional.resize(batch, self.image_size)
        image_batches = [
            torch.cat([global_image.unsqueeze(0), image_tiles])
            for global_image, image_tiles in zip(global_images, tiles)
        ]
        original_sizes = torch.tensor(
            [[original_width, original_height] for _ in range(batch.shape[0])],
            dtype=torch.int64,
            device=batch.device,
        )
        local_sizes = torch.tensor(
            [tile_w, tile_h], dtype=torch.int64, device=batch.device
        ).expand(tiles.shape[1], -1)
        target_sizes = [
            torch.cat([size.unsqueeze(0), local_sizes]) for size in original_sizes
        ]
        local_coordinates = torch.tensor(
            [[w, h] for h in h_starts for w in w_starts],
            dtype=torch.int64,
            device=batch.device,
        )
        coordinates = torch.cat(
            [
                torch.zeros(1, 2, dtype=torch.int64, device=batch.device),
                local_coordinates,
            ]
        )
        return image_batches, target_sizes, coordinates

    @staticmethod
    def _tile_starts(size: int, tile_size: int, step: int) -> list[int]:
        if size <= tile_size:
            return [0]
        last_start = size - tile_size
        starts = list(range(0, last_start + 1, step))
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def preprocess_batch(self, batch: Tensor) -> Tensor:
        mean = self.get_buffer("image_mean")
        std = self.get_buffer("image_std")
        if mean is None or std is None:
            return batch
        mean = mean.to(dtype=batch.dtype).view(1, -1, 1, 1)
        std = std.to(dtype=batch.dtype).view(1, -1, 1, 1)
        return (batch - mean) / std


class ObjectDetectionPostprocessor(Module):
    """Decode raw object detection outputs into typed per-image predictions."""

    internal_class_to_class: Tensor

    def __init__(
        self, *, num_classes: int, num_top_queries: int, internal_class_to_class: Tensor
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.register_buffer(
            "internal_class_to_class", internal_class_to_class, persistent=False
        )

    def decode(
        self, raw: ObjectDetectionOutput, target_sizes: Tensor
    ) -> ObjectDetectionDecodedBatch:
        scores = raw.logits.sigmoid()
        num_classes = scores.shape[-1]
        scores, index = scores.flatten(1).topk(self.num_top_queries, dim=-1)
        labels = index % num_classes
        query_index = index // num_classes
        boxes = box_convert(raw.boxes, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = boxes.gather(1, query_index.unsqueeze(-1).expand(-1, -1, 4))
        boxes = boxes * target_sizes.repeat(1, 2).unsqueeze(1)
        return ObjectDetectionDecodedBatch(
            self.internal_class_to_class[labels], boxes, scores
        )

    def postprocess(
        self, raw: ObjectDetectionOutput, target_sizes: Tensor, threshold: float
    ) -> list[ObjectDetectionPrediction]:
        decoded = self.decode(raw, target_sizes)
        return self._filter(decoded, threshold)

    def postprocess_metadata(
        self,
        raw: ObjectDetectionOutput,
        metadata: Sequence[ObjectDetectionMetadata],
        threshold: float,
    ) -> list[ObjectDetectionPrediction]:
        target_sizes = torch.tensor(
            [[item["orig_w"], item["orig_h"]] for item in metadata],
            dtype=torch.int64,
            device=self.internal_class_to_class.device,
        )
        return self.postprocess(raw, target_sizes, threshold)

    def postprocess_sahi(
        self,
        raw: ObjectDetectionOutput,
        batch: ObjectDetectionSahiPreprocessedBatch,
        *,
        threshold: float,
        nms_iou_threshold: float,
        global_local_iou_threshold: float,
    ) -> list[ObjectDetectionPrediction]:
        decoded = self.decode(raw, batch.target_sizes)
        predictions: list[ObjectDetectionPrediction] = []
        for image_index in range(batch.tile_offsets.numel() - 1):
            start = int(batch.tile_offsets[image_index])
            end = int(batch.tile_offsets[image_index + 1])
            labels = decoded.labels[start:end]
            boxes = decoded.bboxes[start:end].clone()
            scores = decoded.scores[start:end]
            offsets = batch.tile_coordinates[start:end].repeat(1, 2).unsqueeze(1)
            boxes[1:] += offsets[1:]
            global_prediction = self._prediction(
                labels[0], boxes[0], scores[0], threshold
            )
            tile_prediction = self._prediction(
                labels[1:].flatten(),
                boxes[1:].flatten(0, 1),
                scores[1:].flatten(),
                threshold,
            )
            labels_out, boxes_out, scores_out = (
                tiling_utils.combine_object_detection_tiles(
                    pred_global={
                        "labels": global_prediction.labels,
                        "bboxes": global_prediction.bboxes,
                        "scores": global_prediction.scores,
                    },
                    pred_tiles={
                        "labels": tile_prediction.labels,
                        "bboxes": tile_prediction.bboxes,
                        "scores": tile_prediction.scores,
                    },
                    nms_iou_threshold=nms_iou_threshold,
                    global_local_iou_threshold=global_local_iou_threshold,
                )
            )
            predictions.append(
                ObjectDetectionPrediction(labels_out, boxes_out, scores_out)
            )
        return predictions

    @staticmethod
    def _prediction(
        labels: Tensor, boxes: Tensor, scores: Tensor, threshold: float
    ) -> ObjectDetectionPrediction:
        keep = scores > threshold
        return ObjectDetectionPrediction(labels[keep], boxes[keep], scores[keep])

    def _filter(
        self, decoded: ObjectDetectionDecodedBatch, threshold: float
    ) -> list[ObjectDetectionPrediction]:
        return [
            self._prediction(labels, boxes, scores, threshold)
            for labels, boxes, scores in zip(
                decoded.labels, decoded.bboxes, decoded.scores
            )
        ]
