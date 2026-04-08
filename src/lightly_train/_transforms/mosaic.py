#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
import random
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from lightly_train.types import NDArrayBBoxes, NDArrayClasses, NDArrayImage

NDArrayXYXYBBoxes = NDArray[np.float32]


class _MosaicCacheItem(TypedDict):
    img: NDArrayImage
    boxes: NDArrayXYXYBBoxes
    labels: NDArrayClasses


class MosaicTransform:
    """Mosaic augmentation that combines 4 images into a 2x2 grid with post-mosaic
    affine transform. Uses an internal cache to avoid loading extra images.

    Replicates the mosaic behavior from LT-DETR.

    Input/output format:
        - image: numpy RGB array (H, W, 3), uint8
        - bboxes: numpy array (N, 4) in YOLO normalized format (cx, cy, w, h)
        - class_labels: numpy array (N,) of int class labels
    """

    def __init__(
        self,
        output_size: int,
        max_size: int | None,
        rotation_range: float,
        translation_range: tuple[float, float],
        scaling_range: tuple[float, float],
        fill_value: int,
        max_cached_images: int,
        random_pop: bool,
    ) -> None:
        self.output_size = output_size
        self.max_size = max_size
        # NOTE: Torchvision clips transformed boxes to the canvas but keeps
        # degenerate boxes after clipping. We preserve that behavior here.
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scaling_range = scaling_range
        self.fill_value = fill_value
        self._mosaic_cache: list[_MosaicCacheItem] = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    def __call__(
        self,
        image: NDArrayImage,
        bboxes: NDArrayBBoxes,
        class_labels: NDArrayClasses,
    ) -> tuple[NDArrayImage, NDArrayBBoxes, NDArrayClasses]:
        h, w = image.shape[:2]

        # Convert YOLO normalized (cx, cy, bw, bh) -> xyxy absolute.
        boxes_xyxy = _yolo_to_xyxy(bboxes, w, h)
        resized_img, resized_boxes = _resize_image_and_boxes(
            image=image,
            boxes=boxes_xyxy,
            output_size=self.output_size,
            max_size=self.max_size,
        )
        resized_labels = class_labels.copy()

        # Cache management (matching LT-DETR load_samples_from_cache).
        self._mosaic_cache.append(
            {
                "img": resized_img,
                "boxes": resized_boxes,
                "labels": resized_labels,
            }
        )
        if len(self._mosaic_cache) > self.max_cached_images:
            if self.random_pop:
                # Evict random index excluding the last (just appended).
                evict_idx = random.randint(0, len(self._mosaic_cache) - 2)
            else:
                evict_idx = 0
            self._mosaic_cache.pop(evict_idx)

        # NOTE: LT-DETR always creates a mosaic even when the cache has fewer
        # than 4 unique images, sampling with replacement. Early calls will
        # produce mosaics of mostly duplicate images.
        # Sample 3 from cache (may include duplicates).
        sampled_indices = random.choices(range(len(self._mosaic_cache)), k=3)
        samples: list[_MosaicCacheItem] = [
            {"img": resized_img, "boxes": resized_boxes, "labels": resized_labels}
        ]
        for idx in sampled_indices:
            samples.append(self._mosaic_cache[idx])

        # Mosaic assembly (matching LT-DETR create_mosaic_from_cache).
        max_h = max(s["img"].shape[0] for s in samples)
        max_w = max(s["img"].shape[1] for s in samples)

        canvas = np.zeros((2 * max_h, 2 * max_w, 3), dtype=np.uint8)

        offsets = [
            (0, 0),
            (max_w, 0),
            (0, max_h),
            (max_w, max_h),
        ]

        total_boxes = sum(sample["boxes"].shape[0] for sample in samples)
        mosaic_boxes = np.empty((total_boxes, 4), dtype=np.float32)
        mosaic_labels = np.empty((total_boxes,), dtype=resized_labels.dtype)
        box_start = 0

        for sample, (dx, dy) in zip(samples, offsets):
            sample_h, sample_w = sample["img"].shape[:2]
            canvas[dy : dy + sample_h, dx : dx + sample_w] = sample["img"]
            boxes = sample["boxes"]
            num_boxes = boxes.shape[0]
            if num_boxes == 0:
                continue

            box_end = box_start + num_boxes
            mosaic_boxes[box_start:box_end] = boxes
            mosaic_boxes[box_start:box_end, [0, 2]] += dx
            mosaic_boxes[box_start:box_end, [1, 3]] += dy
            mosaic_labels[box_start:box_end] = sample["labels"]
            box_start = box_end

        angle, translate, scale = _sample_affine_params(
            canvas_height=2 * max_h,
            canvas_width=2 * max_w,
            rotation_range=self.rotation_range,
            translation_range=self.translation_range,
            scaling_range=self.scaling_range,
        )
        out_img_np = _apply_affine_to_image(
            image=canvas,
            angle=angle,
            translate=translate,
            scale=scale,
            fill_value=self.fill_value,
        )
        out_boxes_xyxy = _apply_affine_to_boxes(
            boxes=mosaic_boxes,
            canvas_size=(2 * max_h, 2 * max_w),
            angle=angle,
            translate=translate,
            scale=scale,
        )

        # Convert PIL -> numpy, xyxy absolute -> YOLO normalized.
        out_h, out_w = out_img_np.shape[:2]
        out_bboxes = _xyxy_to_yolo(out_boxes_xyxy, out_w, out_h)
        out_labels = mosaic_labels

        return out_img_np, out_bboxes, out_labels


def _sample_affine_params(
    canvas_height: int,
    canvas_width: int,
    rotation_range: float,
    translation_range: tuple[float, float],
    scaling_range: tuple[float, float],
) -> tuple[float, tuple[int, int], float]:
    angle = random.uniform(-rotation_range, rotation_range)
    max_dx = translation_range[0] * canvas_width
    max_dy = translation_range[1] * canvas_height
    tx = int(round(random.uniform(-max_dx, max_dx)))
    ty = int(round(random.uniform(-max_dy, max_dy)))
    scale = random.uniform(scaling_range[0], scaling_range[1])
    return angle, (tx, ty), scale


def _resize_image_and_boxes(
    image: NDArrayImage,
    boxes: NDArrayXYXYBBoxes,
    output_size: int,
    max_size: int | None,
) -> tuple[NDArrayImage, NDArrayXYXYBBoxes]:
    height, width = image.shape[:2]
    new_height, new_width = _compute_resized_shape(
        height=height,
        width=width,
        output_size=output_size,
        max_size=max_size,
    )
    resized_image = np.array(
        Image.fromarray(image).resize(
            (new_width, new_height), resample=Image.Resampling.BILINEAR
        )
    )
    if boxes.size == 0:
        return resized_image, boxes.astype(np.float32, copy=True)

    scale_factors = np.array(
        [
            new_width / width,
            new_height / height,
            new_width / width,
            new_height / height,
        ],
        dtype=np.float32,
    )
    resized_boxes = boxes * scale_factors
    return resized_image, resized_boxes


def _compute_resized_shape(
    height: int, width: int, output_size: int, max_size: int | None
) -> tuple[int, int]:
    short_edge = min(height, width)
    long_edge = max(height, width)
    scale = output_size / short_edge
    if max_size is not None and scale * long_edge > max_size:
        scale = max_size / long_edge
    # Match torchvision Resize(size=int, max_size=...) shape computation, which
    # truncates the scaled dimensions instead of rounding them.
    return int(height * scale), int(width * scale)


def _apply_affine_to_image(
    image: NDArrayImage,
    angle: float,
    translate: tuple[int, int],
    scale: float,
    fill_value: int,
) -> NDArrayImage:
    height, width = image.shape[:2]
    inverse_affine_matrix = _get_affine_matrix(
        center=(width * 0.5, height * 0.5),
        angle=angle,
        translate=translate,
        scale=scale,
        inverted=True,
    )
    transformed_image = Image.fromarray(image).transform(
        size=(width, height),
        method=Image.Transform.AFFINE,
        data=inverse_affine_matrix,
        resample=Image.Resampling.NEAREST,
        fillcolor=(fill_value, fill_value, fill_value),
    )
    return np.array(transformed_image)


def _apply_affine_to_boxes(
    boxes: NDArrayXYXYBBoxes,
    canvas_size: tuple[int, int],
    angle: float,
    translate: tuple[int, int],
    scale: float,
) -> NDArrayXYXYBBoxes:
    if boxes.size == 0:
        return boxes.astype(np.float32, copy=True)

    canvas_height, canvas_width = canvas_size
    affine_matrix = (
        np.array(
            _get_affine_matrix(
                center=(canvas_width * 0.5, canvas_height * 0.5),
                angle=angle,
                translate=translate,
                scale=scale,
                inverted=False,
            ),
            dtype=np.float32,
        )
        .reshape(2, 3)
        .T
    )

    corners = boxes[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].reshape(-1, 2)
    points = np.concatenate(
        [corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1
    )
    transformed_points = points @ affine_matrix
    transformed_points = transformed_points.reshape(-1, 4, 2)

    out_bbox_mins = np.min(transformed_points, axis=1)
    out_bbox_maxs = np.max(transformed_points, axis=1)
    transformed_boxes = np.concatenate([out_bbox_mins, out_bbox_maxs], axis=1)
    transformed_boxes[:, [0, 2]] = np.clip(
        transformed_boxes[:, [0, 2]], a_min=0.0, a_max=canvas_width
    )
    transformed_boxes[:, [1, 3]] = np.clip(
        transformed_boxes[:, [1, 3]], a_min=0.0, a_max=canvas_height
    )
    return transformed_boxes.astype(np.float32, copy=False)


def _get_affine_matrix(
    center: tuple[float, float],
    angle: float,
    translate: tuple[int, int],
    scale: float,
    inverted: bool,
) -> list[float]:
    """Return a Pillow-compatible affine matrix matching torchvision geometry."""
    rot = math.radians(angle)
    sx = 0.0
    sy = 0.0

    cx, cy = center
    tx, ty = translate

    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [value / scale for value in matrix]
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [value * scale for value in matrix]
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix


def _yolo_to_xyxy(bboxes: NDArrayBBoxes, w: int, h: int) -> NDArrayXYXYBBoxes:
    """Convert YOLO normalized (cx, cy, bw, bh) to xyxy absolute coordinates."""
    if len(bboxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    bboxes_t = bboxes.astype(np.float32, copy=False)
    cx, cy, bw, bh = bboxes_t[:, 0], bboxes_t[:, 1], bboxes_t[:, 2], bboxes_t[:, 3]
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32, copy=False)


def _xyxy_to_yolo(boxes: NDArrayXYXYBBoxes, w: int, h: int) -> NDArrayBBoxes:
    """Convert xyxy absolute coordinates to YOLO normalized (cx, cy, bw, bh)."""
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return np.stack([cx, cy, bw, bh], axis=1).astype(np.float64, copy=False)
