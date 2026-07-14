#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Numpy/PIL re-implementation of the object detection pre- and post-processing.

This mirrors ``lightly_train._pre_post_processing.object_detection`` but depends
only on numpy and Pillow so that exported models can be run without the full
LightlyTrain stack (no torch/torchvision).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

# An input image can be given as a filesystem path or an already-loaded PIL image.
ImageInput = Union[str, "os.PathLike[str]", PILImage]


class ObjectDetectionMetadata(TypedDict):
    orig_h: int
    orig_w: int


class ObjectDetectionPreprocessor:
    """Per-image and batch preprocessing for object detection inference.

    Mirrors the torch ``ObjectDetectionPreprocessor`` order of operations:
    validate channels, scale to ``[0, 1]``, resize to ``image_size`` and finally
    normalize on the batched tensor.
    """

    def __init__(
        self,
        *,
        image_size: Tuple[int, int],
        image_normalize: Optional[Dict[str, Tuple[float, ...]]],
        expected_input_channels: int,
    ) -> None:
        self.image_size = image_size
        self.image_normalize = image_normalize
        self.expected_input_channels = expected_input_channels

    def preprocess_image(
        self, image: ImageInput
    ) -> Tuple[np.ndarray, ObjectDetectionMetadata]:
        """Load and preprocess a single image into a ``(C, H, W)`` float32 array."""
        pil_image = self._as_pil_image(image)
        orig_w, orig_h = pil_image.size  # PIL reports (width, height).

        pil_image = self._validate_channels(pil_image)

        # Resize with bilinear resampling to match the torch antialiased bilinear
        # resize. PIL expects (width, height).
        target_h, target_w = self.image_size
        pil_image = pil_image.resize((target_w, target_h), Image.Resampling.BILINEAR)

        # (H, W, C) uint8 -> (C, H, W) float32 in [0, 1]. Bilinear resize is linear,
        # so scaling after resize is numerically equivalent to the torch order
        # (scale then resize) up to rounding.
        x = np.asarray(pil_image, dtype=np.float32) / 255.0
        if x.ndim == 2:
            x = x[:, :, None]
        x = np.transpose(x, (2, 0, 1))
        return x, {"orig_h": orig_h, "orig_w": orig_w}

    def _validate_channels(self, image: PILImage) -> PILImage:
        # Expand grayscale to the expected channel count so images can be stacked.
        channels = len(image.getbands())
        expected_c = self.expected_input_channels
        if channels == 1 and expected_c > 1:
            # RGB covers the common expected_c == 3 case; the array-level channel
            # count is checked below for any other value.
            image = image.convert("RGB")
            channels = len(image.getbands())
        if channels != expected_c:
            raise ValueError(
                f"Image has {channels} channels but model expects {expected_c}."
            )
        return image

    def preprocess_batch(self, batch: np.ndarray) -> np.ndarray:
        """Normalize a stacked ``(B, C, H, W)`` batch in-place-safe fashion."""
        if self.image_normalize is not None:
            mean = np.asarray(self.image_normalize["mean"], dtype=np.float32)
            std = np.asarray(self.image_normalize["std"], dtype=np.float32)
            batch = (batch - mean[None, :, None, None]) / std[None, :, None, None]
        return batch

    @staticmethod
    def _as_pil_image(image: ImageInput) -> PILImage:
        if isinstance(image, PILImage):
            return image
        return Image.open(os.fspath(image))


class ObjectDetectionPostprocessor:
    """Decodes raw object detection outputs into per-image predictions.

    Performs the RT-DETR focal-loss decoding (sigmoid scores + flat top-k over
    queries and classes), box decoding/rescaling and the internal-to-user class-id
    mapping, in plain numpy.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        num_top_queries: int,
        internal_class_to_class: np.ndarray,
    ) -> None:
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.internal_class_to_class = internal_class_to_class

    def decode(
        self, logits: np.ndarray, boxes: np.ndarray, orig_target_sizes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-image ``(labels, boxes, scores)`` with class ids remapped.

        Args:
            logits:
                Raw logits of shape ``(B, num_queries, num_classes)``.
            boxes:
                Normalized ``cxcywh`` boxes of shape ``(B, num_queries, 4)``.
            orig_target_sizes:
                Array of shape ``(B, 2)`` giving the ``(width, height)`` each
                image's boxes should be rescaled to.

        Returns:
            ``(labels, boxes, scores)`` of shapes ``(B, K)``, ``(B, K, 4)``,
            ``(B, K)``. Boxes are ``xyxy`` in pixel coordinates.
        """
        scores = _sigmoid(logits)  # (B, num_queries, num_classes)
        batch_size, num_queries, num_classes = scores.shape

        # Flat top-k across the (query, class) grid, as in RT-DETR focal-loss
        # decoding. Requesting more than the grid size would be out of bounds.
        k = min(self.num_top_queries, num_queries * num_classes)
        flat = scores.reshape(batch_size, -1)
        index = _topk(flat, k)  # (B, k), descending by score.
        top_scores = np.take_along_axis(flat, index, axis=1)

        labels = index % num_classes
        query_index = index // num_classes

        boxes_xyxy = _cxcywh_to_xyxy(boxes)
        gather_index = np.broadcast_to(query_index[:, :, None], (batch_size, k, 4))
        top_boxes = np.take_along_axis(boxes_xyxy, gather_index, axis=1)

        # Scale normalized boxes to pixel coordinates: (w, h, w, h) per image.
        scale = np.concatenate([orig_target_sizes, orig_target_sizes], axis=1)
        top_boxes = top_boxes * scale[:, None, :].astype(np.float32)

        labels = self.internal_class_to_class[labels]
        return labels, top_boxes, top_scores

    def postprocess(
        self,
        logits: np.ndarray,
        boxes: np.ndarray,
        metadata: Sequence[ObjectDetectionMetadata],
        threshold: float,
    ) -> List[Dict[str, np.ndarray]]:
        # Postprocessor expects (W, H) per image.
        orig_target_sizes = np.asarray(
            [[m["orig_w"], m["orig_h"]] for m in metadata], dtype=np.int64
        )
        labels_batch, boxes_batch, scores_batch = self.decode(
            logits, boxes, orig_target_sizes
        )

        out: List[Dict[str, np.ndarray]] = []
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-``k`` scores per row, sorted descending."""
    # argpartition finds the top-k unordered, then argsort orders just those k.
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1)
