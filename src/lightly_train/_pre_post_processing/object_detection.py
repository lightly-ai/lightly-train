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
class ObjectDetectionPrediction(BaseModelOutput):
    """Predictions for one image in original-image ``xyxy`` coordinates.

    Attributes:
        labels: Shape ``(N,)``, ``int64``. Predicted class index for each of the
            ``N`` kept detections.
        bboxes: Shape ``(N, 4)``. ``xyxy`` boxes in original-image pixel coordinates.
        scores: Shape ``(N,)``. Confidence score in ``[0, 1]`` for each detection.
    """

    labels: Tensor
    bboxes: Tensor
    scores: Tensor

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


class ObjectDetectionPreprocessor(Module):
    """Host image preparation plus dense, batch-friendly preprocessing."""

    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        image_normalize: dict[str, tuple[float, ...]] | None,
        expected_input_channels: int,
        sahi_config: ObjectDetectionSahiConfig | None = None,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.image_normalize = image_normalize
        self.expected_input_channels = expected_input_channels
        self.sahi_config = sahi_config

    def preprocess_image(
        self,
        image: PathLike | PILImage | Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, ObjectDetectionMetadata]:
        image_tensor = file_helpers.as_image_tensor(image).to(device)
        orig_h, orig_w = image_tensor.shape[-2:]
        image_tensor = self._validate_channels(image_tensor)
        image_tensor = transforms_functional.to_dtype(
            image_tensor, dtype=dtype, scale=True
        )
        metadata = ObjectDetectionMetadata(orig_h=orig_h, orig_w=orig_w)
        if self.sahi_config is None:
            return transforms_functional.resize(image_tensor, self.image_size), metadata

        tiles, coordinates = tiling_utils.tile_image(
            image=image_tensor,
            overlap=self.sahi_config.overlap,
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
        sahi_config: ObjectDetectionSahiConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.image_size = image_size
        self.sahi_config = sahi_config
        self.register_buffer(
            "internal_class_to_class", internal_class_to_class, persistent=False
        )

    def postprocess(
        self,
        raw: ObjectDetectionOutput,
        metadata: Sequence[ObjectDetectionMetadata],
        threshold: float,
    ) -> list[ObjectDetectionPrediction]:
        target_sizes = self._target_sizes(metadata, device=raw.boxes.device)
        scores = raw.logits.sigmoid()
        num_classes = scores.shape[-1]
        scores, index = scores.flatten(1).topk(self.num_top_queries, dim=-1)
        labels = index % num_classes
        query_index = index // num_classes
        boxes = box_convert(raw.boxes, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = boxes.gather(1, query_index.unsqueeze(-1).expand(-1, -1, 4))
        boxes = boxes * target_sizes.repeat(1, 2).unsqueeze(1)
        labels = self.internal_class_to_class[labels]
        if self.sahi_config is not None:
            if len(metadata) != 1 or metadata[0].tile_coordinates is None:
                raise ValueError("SAHI postprocessing expects metadata for one image.")
            labels_out, boxes_out, scores_out = (
                tiling_utils.combine_sahi_object_detection_predictions(
                    labels=labels,
                    boxes=boxes,
                    scores=scores,
                    tile_coordinates=metadata[0].tile_coordinates,
                    threshold=threshold,
                    nms_iou_threshold=self.sahi_config.nms_iou_threshold,
                    global_local_iou_threshold=(
                        self.sahi_config.global_local_iou_threshold
                    ),
                )
            )
            return [ObjectDetectionPrediction(labels_out, boxes_out, scores_out)]

        return [
            ObjectDetectionPrediction(
                labels_i[scores_i > threshold],
                boxes_i[scores_i > threshold],
                scores_i[scores_i > threshold],
            )
            for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
        ]

    def _target_sizes(
        self, metadata: Sequence[ObjectDetectionMetadata], device: torch.device
    ) -> Tensor:
        if self.sahi_config is None:
            sizes = [[item.orig_w, item.orig_h] for item in metadata]
        else:
            if len(metadata) != 1 or metadata[0].tile_coordinates is None:
                raise ValueError("SAHI postprocessing expects metadata for one image.")
            tile_h, tile_w = self.image_size
            sizes = [
                [metadata[0].orig_w, metadata[0].orig_h],
                *[[tile_w, tile_h] for _ in metadata[0].tile_coordinates],
            ]
        return torch.tensor(sizes, dtype=torch.int64, device=device)
