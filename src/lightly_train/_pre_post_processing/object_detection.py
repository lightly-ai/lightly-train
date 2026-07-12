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
from typing import Any

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
    """Raw object detection outputs returned by the model's ``forward``.

    The field order defines the ONNX output names (``logits``, ``boxes``); see
    ``task_model_io.output_names_from_model_output``.
    """

    logits: Tensor  # (B, num_queries, num_classes)
    boxes: Tensor  # (B, num_queries, 4) in normalized cxcywh format


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
    ) -> tuple[Tensor, dict[str, Any]]:
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
    ) -> tuple[Tensor, dict[str, Any]]:
        x = file_helpers.as_image_tensor(image).to(device)
        orig_h, orig_w = x.shape[-2:]

        x = self._validate_channels(x)
        x = transforms_functional.to_dtype(x, dtype=dtype, scale=True)

        tiles, tiles_coordinates = tiling_utils.tile_image(
            x, overlap=overlap, tile_size=self.image_size
        )

        x_global = transforms_functional.resize(x, self.image_size).unsqueeze(0)
        batch = torch.cat([x_global, tiles], dim=0)
        batch = self.preprocess_batch(batch)

        return batch, {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "tiles_coordinates": tiles_coordinates,
        }

    def _validate_channels(self, x: Tensor) -> Tensor:
        # Expand grayscale to the expected channel count so images can be stacked.
        # TODO(Nauryzbay, 05/26): Revisit grayscale handling — the implicit
        # 1-channel expansion is a convenience inherited from RGB-only models.
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
    """Decodes raw object detection outputs into per-image predictions.

    Performs top-k selection, box decoding and rescaling, and owns the
    internal-to-user class-id mapping. This is the RT-DETR focal-loss decoding
    (sigmoid scores + flat top-k over queries and classes), reimplemented in plain
    eager PyTorch — postprocessing is no longer part of the exported graph, so it
    needs no deploy/TensorRT-compatible variant.
    """

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
        # Registered as buffer to be automatically moved to the correct device.
        self.register_buffer(
            "internal_class_to_class",
            internal_class_to_class,
            persistent=False,  # No need to save it in the state dict.
        )

    def decode(
        self, raw: ObjectDetectionOutput, orig_target_sizes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-image ``(labels, boxes, scores)`` with class ids remapped.

        Args:
            raw:
                Raw model outputs (``logits`` and normalized ``cxcywh`` boxes).
            orig_target_sizes:
                Tensor of shape (B, 2) giving the ``(width, height)`` each image's
                boxes should be rescaled to.

        Returns:
            ``(labels, boxes, scores)``, each of shape (B, num_top_queries[, 4]).
            Boxes are ``xyxy`` in the pixel coordinates of ``orig_target_sizes``.
        """
        scores = raw.logits.sigmoid()  # (B, num_queries, num_classes)
        num_classes = scores.shape[-1]
        # Flat top-k across the (query, class) grid, as in RT-DETR focal-loss decoding.
        scores, index = scores.flatten(1).topk(self.num_top_queries, dim=-1)
        labels = index % num_classes
        query_index = index // num_classes

        boxes = box_convert(raw.boxes, in_fmt="cxcywh", out_fmt="xyxy")
        boxes = boxes.gather(1, query_index.unsqueeze(-1).expand(-1, -1, 4))
        # Scale normalized boxes to pixel coordinates: (w, h, w, h) per image.
        boxes = boxes * orig_target_sizes.repeat(1, 2).unsqueeze(1)

        labels = self.internal_class_to_class[labels]
        return labels, boxes, scores

    def postprocess(
        self,
        raw: ObjectDetectionOutput,
        metadata: Sequence[dict[str, Any]],
        threshold: float,
    ) -> list[dict[str, Tensor]]:
        device = self.internal_class_to_class.device
        # Postprocessor expects (W, H) per image.
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
        metadata: dict[str, Any],
        *,
        threshold: float,
        nms_iou_threshold: float,
        global_local_iou_threshold: float,
        tile_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        device = self.internal_class_to_class.device
        tile_h, tile_w = tile_size
        orig_h = int(metadata["orig_h"])
        orig_w = int(metadata["orig_w"])
        tiles_coordinates = metadata["tiles_coordinates"].to(device)

        # Decoder expects (W, H). The first entry is the global image; all remaining
        # entries are fixed-size tiles.
        orig_target_sizes = torch.tensor(
            [[orig_w, orig_h], *[[tile_w, tile_h] for _ in range(len(tiles_coordinates))]],
            dtype=torch.int64,
            device=device,
        )
        labels, boxes, scores = self.decode(raw, orig_target_sizes)

        tiles_coordinates = (
            tiles_coordinates.repeat(1, 2).unsqueeze(1).expand(-1, boxes.shape[1], -1)
        )
        boxes[1:] += tiles_coordinates

        boxes_global = boxes[0].view(-1, 4)
        boxes_tiles = boxes[1:].view(-1, 4)
        labels_global = labels[0].flatten()
        labels_tiles = labels[1:].flatten()
        scores_global = scores[0].flatten()
        scores_tiles = scores[1:].flatten()

        keep_global = scores_global > threshold
        keep_tiles = scores_tiles > threshold

        labels, boxes, scores = tiling_utils.combine_object_detection_tiles(
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

        return {
            "labels": labels,
            "bboxes": boxes,
            "scores": scores,
        }
