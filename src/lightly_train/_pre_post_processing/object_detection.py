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
from typing import TypedDict

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import Module
from torchvision.ops import batched_nms, box_convert, box_iou
from torchvision.transforms.v2 import functional as transforms_functional
from typing_extensions import NotRequired

from lightly_train._data import file_helpers
from lightly_train._pre_post_processing import tiling
from lightly_train._task_models.task_model_io import BaseModelOutput
from lightly_train.types import PathLike


@dataclass
class ObjectDetectionOutput(BaseModelOutput):
    """Raw object detection output: ``(logits, normalized cxcywh boxes)``."""

    logits: Tensor
    boxes: Tensor


class ObjectDetectionMetadata(TypedDict):
    orig_h: int
    orig_w: int
    tiles_coordinates: NotRequired[Tensor]


class ObjectDetectionPreprocessor(Module):
    """Per-image and batch preprocessing for object detection task models."""

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
    ) -> tuple[Tensor, ObjectDetectionMetadata]:
        x = file_helpers.as_image_tensor(image).to(device)
        image_h, image_w = x.shape[-2:]
        x = self._validate_channels(x)
        x = transforms_functional.to_dtype(x, dtype=dtype, scale=True)
        x = transforms_functional.resize(x, self.image_size)
        return x, {"orig_h": image_h, "orig_w": image_w}

    def preprocess_sahi_image(
        self,
        image: PathLike | PILImage | Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        overlap: float,
    ) -> tuple[Tensor, ObjectDetectionMetadata]:
        x = file_helpers.as_image_tensor(image).to(device)
        orig_h, orig_w = x.shape[-2:]
        x = self._validate_channels(x)
        x = transforms_functional.to_dtype(x, dtype=dtype, scale=True)
        tiles, tiles_coordinates = tiling.tile_image(
            x, overlap=overlap, tile_size=self.image_size
        )
        x_global = transforms_functional.resize(x, self.image_size).unsqueeze(0)
        return torch.cat([x_global, tiles], dim=0), {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "tiles_coordinates": tiles_coordinates,
        }

    def preprocess_sahi_batch(self, batch: Tensor) -> Tensor:
        return self.preprocess_batch(batch)

    def _validate_channels(self, x: Tensor) -> Tensor:
        expected_c = self.expected_input_channels
        if x.shape[-3] == 1 and expected_c > 1:
            x = x.expand(expected_c, -1, -1)
        elif x.shape[-3] != expected_c:
            raise ValueError(
                f"Image has {x.shape[-3]} channels but model expects {expected_c}."
            )
        return x

    def preprocess_batch(self, batch: Tensor) -> Tensor:
        if self.image_normalize is not None:
            batch = transforms_functional.normalize(
                batch,
                mean=list(self.image_normalize["mean"]),
                std=list(self.image_normalize["std"]),
            )
        return batch


class ObjectDetectionPostprocessor(Module):
    """Decode raw object detection outputs into per-image predictions."""

    internal_class_to_class: Tensor

    def __init__(
        self,
        *,
        num_classes: int,
        num_top_queries: int,
        internal_class_to_class: Tensor,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.register_buffer(
            "internal_class_to_class", internal_class_to_class, persistent=False
        )

    def decode(
        self, raw: ObjectDetectionOutput, orig_target_sizes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        labels, boxes, scores = self.decode_internal(raw, orig_target_sizes)
        return self.internal_class_to_class[labels], boxes, scores

    def decode_internal(
        self, raw: ObjectDetectionOutput, orig_target_sizes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Like decode, but returns internal class ids in [0, num_classes - 1]."""
        scores = raw.logits.sigmoid()
        num_classes = scores.shape[-1]
        scores, index = scores.flatten(1).topk(self.num_top_queries, dim=-1)
        labels = index % num_classes
        query_index = index // num_classes
        boxes = box_convert(raw.boxes, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = boxes.gather(1, query_index.unsqueeze(-1).expand(-1, -1, 4))
        boxes = boxes * orig_target_sizes.repeat(1, 2).unsqueeze(1)
        return labels, boxes, scores

    def postprocess(
        self,
        raw: ObjectDetectionOutput,
        metadata: Sequence[ObjectDetectionMetadata],
        threshold: float,
    ) -> list[dict[str, Tensor]]:
        device = self.internal_class_to_class.device
        orig_target_size = torch.tensor(
            [[m["orig_w"], m["orig_h"]] for m in metadata],
            dtype=torch.int64,
            device=device,
        )
        labels_batch, boxes_batch, scores_batch = self.decode(raw, orig_target_size)
        out: list[dict[str, Tensor]] = []
        for i in range(len(metadata)):
            keep = scores_batch[i] > threshold
            out.append(
                {
                    "labels": labels_batch[i][keep],
                    "bboxes": boxes_batch[i][keep],
                    "scores": scores_batch[i][keep],
                }
            )
        return out

    def postprocess_sahi(
        self,
        raw: ObjectDetectionOutput,
        metadata: ObjectDetectionMetadata,
        *,
        threshold: float,
        nms_iou_threshold: float,
        global_local_iou_threshold: float,
        tile_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        device = self.internal_class_to_class.device
        tile_h, tile_w = tile_size
        tiles_coordinates = metadata["tiles_coordinates"].to(device)
        orig_target_sizes = torch.tensor(
            [
                [int(metadata["orig_w"]), int(metadata["orig_h"])],
                *[[tile_w, tile_h] for _ in tiles_coordinates],
            ],
            dtype=torch.int64,
            device=device,
        )
        labels, boxes, scores = self.decode(raw, orig_target_sizes)
        offsets = tiles_coordinates.repeat(1, 2).unsqueeze(1)
        boxes[1:] += offsets.expand(-1, boxes.shape[1], -1)

        boxes_global, boxes_tiles = boxes[0].view(-1, 4), boxes[1:].view(-1, 4)
        labels_global, labels_tiles = labels[0].flatten(), labels[1:].flatten()
        scores_global, scores_tiles = scores[0].flatten(), scores[1:].flatten()
        keep_global = scores_global > threshold
        keep_tiles = scores_tiles > threshold
        labels, boxes, scores = combine_object_detection_tiles(
            pred_global={
                "labels": labels_global[keep_global],
                "bboxes": boxes_global[keep_global],
                "scores": scores_global[keep_global],
            },
            pred_tiles={
                "labels": labels_tiles[keep_tiles],
                "bboxes": boxes_tiles[keep_tiles],
                "scores": scores_tiles[keep_tiles],
            },
            nms_iou_threshold=nms_iou_threshold,
            global_local_iou_threshold=global_local_iou_threshold,
        )
        return {"labels": labels, "bboxes": boxes, "scores": scores}


def combine_object_detection_tiles(
    pred_global: dict[str, Tensor],
    pred_tiles: dict[str, Tensor],
    nms_iou_threshold: float = 0.2,
    global_local_iou_threshold: float = 0.1,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Combine predictions from the global view (full image) and local views (image tiles).

    Args:
        pred_global: dict with keys "labels", "bboxes", "scores".
        pred_tiles: dict with keys "labels", "bboxes", "scores".
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

    # NMS on tiles predictions is needed due overlapping tiles. Class-aware so a
    # detection cannot suppress an overlapping detection of another class.
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
