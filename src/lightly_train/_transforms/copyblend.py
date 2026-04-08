#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

if TYPE_CHECKING:
    _Int64Array = np.ndarray[Any, np.dtype[np.int64]]
    _Float64Array = np.ndarray[Any, np.dtype[np.float64]]
else:
    _Int64Array = np.ndarray
    _Float64Array = np.ndarray


class _ObjectPool(TypedDict):
    source_indices: _Int64Array
    source_x1: _Int64Array
    source_y1: _Int64Array
    source_x2: _Int64Array
    source_y2: _Int64Array
    widths: _Int64Array
    heights: _Int64Array
    labels: _Int64Array


class _Placement(TypedDict):
    source_index: int
    source_x1: int
    source_y1: int
    source_x2: int
    source_y2: int
    dest_x1: int
    dest_y1: int
    dest_x2: int
    dest_y2: int
    bbox: _Float64Array


class CopyBlend:
    def __init__(
        self,
        *,
        area_threshold: int,
        num_objects: int,
        expand_ratios: tuple[float, float],
        beta_range: tuple[float, float] = (0.45, 0.55),
    ) -> None:
        self.area_threshold = area_threshold
        self.num_objects = num_objects
        self.expand_ratios = expand_ratios
        self.beta_range = beta_range

    def __call__(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # CopyBlend currently assumes bounding boxes are normalized YOLO coordinates
        # in (cx, cy, w, h) format.

        images = np.stack([item["image"] for item in batch], axis=0)
        image_height, image_width = images.shape[1:3]

        object_pool = self._build_object_pool(
            batch=batch,
            image_width=image_width,
            image_height=image_height,
        )
        if len(object_pool["labels"]) == 0:
            return batch

        max_objects = min(self.num_objects, len(object_pool["labels"]))

        beta = np.float32(np.random.uniform(*self.beta_range))
        one_minus_beta = np.float32(1.0) - beta
        blended_images = images.copy()
        copied_any_object = False

        updated_batch: list[dict[str, Any]] = []
        for image_index, item in enumerate(batch):
            selected_indices = np.random.choice(
                len(object_pool["labels"]),
                size=max_objects,
                replace=False,
            )

            copied_bboxes: list[_Float64Array] = []
            copied_labels: list[np.int64] = []

            for object_index in selected_indices:
                placement = self._sample_placement(
                    object_index=int(object_index),
                    object_pool=object_pool,
                    image_width=image_width,
                    image_height=image_height,
                )
                if placement is None:
                    continue

                source_index = placement["source_index"]
                src_y1, src_y2 = placement["source_y1"], placement["source_y2"]
                src_x1, src_x2 = placement["source_x1"], placement["source_x2"]
                dst_y1, dst_y2 = placement["dest_y1"], placement["dest_y2"]
                dst_x1, dst_x2 = placement["dest_x1"], placement["dest_x2"]

                source_patch = images[source_index, src_y1:src_y2, src_x1:src_x2]
                destination_patch = blended_images[
                    image_index, dst_y1:dst_y2, dst_x1:dst_x2
                ]
                blended_images[image_index, dst_y1:dst_y2, dst_x1:dst_x2] = (
                    destination_patch * beta + source_patch * one_minus_beta
                )

                copied_bboxes.append(placement["bbox"])
                copied_labels.append(object_pool["labels"][int(object_index)])
                copied_any_object = True

            if copied_bboxes:
                blended_bboxes = np.concatenate(
                    (item["bboxes"], np.stack(copied_bboxes, axis=0)),
                    axis=0,
                )
                blended_class_labels = np.concatenate(
                    (
                        item["class_labels"],
                        np.asarray(copied_labels, dtype=np.int64),
                    ),
                    axis=0,
                )
            else:
                blended_bboxes = item["bboxes"]
                blended_class_labels = item["class_labels"]

            updated_batch.append(
                {
                    "image": blended_images[image_index],
                    "bboxes": blended_bboxes,
                    "class_labels": blended_class_labels,
                }
            )

        if not copied_any_object:
            return batch

        return updated_batch

    def _build_object_pool(
        self,
        *,
        batch: list[dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> _ObjectPool:
        # Each tuple: (source_indices, x1, y1, x2, y2, widths, heights, labels)
        per_image: list[tuple[_Int64Array, ...]] = []

        for image_index, item in enumerate(batch):
            bboxes = item["bboxes"]
            if len(bboxes) == 0:
                continue

            x1 = ((bboxes[:, 0] - bboxes[:, 2] / 2.0) * image_width).astype(np.int64)
            y1 = ((bboxes[:, 1] - bboxes[:, 3] / 2.0) * image_height).astype(np.int64)
            x2 = ((bboxes[:, 0] + bboxes[:, 2] / 2.0) * image_width).astype(np.int64)
            y2 = ((bboxes[:, 1] + bboxes[:, 3] / 2.0) * image_height).astype(np.int64)

            x1 = np.clip(x1, 0, image_width)
            y1 = np.clip(y1, 0, image_height)
            x2 = np.clip(x2, 0, image_width)
            y2 = np.clip(y2, 0, image_height)

            widths_px = x2 - x1
            heights_px = y2 - y1
            areas = widths_px * heights_px
            keep = (widths_px > 0) & (heights_px > 0) & (areas >= self.area_threshold)
            if not np.any(keep):
                continue

            num_kept = int(np.count_nonzero(keep))
            per_image.append(
                (
                    np.full(num_kept, image_index, dtype=np.int64),
                    x1[keep],
                    y1[keep],
                    x2[keep],
                    y2[keep],
                    widths_px[keep],
                    heights_px[keep],
                    item["class_labels"][keep],
                )
            )

        if not per_image:
            empty_int = np.array([], dtype=np.int64)
            return {
                "source_indices": empty_int,
                "source_x1": empty_int,
                "source_y1": empty_int,
                "source_x2": empty_int,
                "source_y2": empty_int,
                "widths": empty_int,
                "heights": empty_int,
                "labels": empty_int,
            }

        si, sx1, sy1, sx2, sy2, w, h, lbl = (
            np.concatenate(col) for col in zip(*per_image)
        )
        return {
            "source_indices": si,
            "source_x1": sx1,
            "source_y1": sy1,
            "source_x2": sx2,
            "source_y2": sy2,
            "widths": w,
            "heights": h,
            "labels": lbl,
        }

    def _sample_placement(
        self,
        *,
        object_index: int,
        object_pool: _ObjectPool,
        image_width: int,
        image_height: int,
    ) -> _Placement | None:
        source_index = int(object_pool["source_indices"][object_index])
        source_x1 = int(object_pool["source_x1"][object_index])
        source_y1 = int(object_pool["source_y1"][object_index])
        source_x2 = int(object_pool["source_x2"][object_index])
        source_y2 = int(object_pool["source_y2"][object_index])
        object_width = int(object_pool["widths"][object_index])
        object_height = int(object_pool["heights"][object_index])

        if object_width <= 0 or object_height <= 0:
            return None

        if object_width < image_width:
            dest_x1 = int(np.random.randint(0, image_width - object_width + 1))
        else:
            dest_x1 = 0

        if object_height < image_height:
            dest_y1 = int(np.random.randint(0, image_height - object_height + 1))
        else:
            dest_y1 = 0

        dest_x2 = dest_x1 + object_width
        dest_y2 = dest_y1 + object_height

        bbox = np.array(
            [
                (dest_x1 + object_width / 2.0) / image_width,
                (dest_y1 + object_height / 2.0) / image_height,
                object_width / image_width,
                object_height / image_height,
            ],
            dtype=np.float64,
        )

        alpha = np.random.uniform(*self.expand_ratios)
        expand_width = int(object_width * alpha)
        expand_height = int(object_height * alpha)

        left_expand = min(expand_width, source_x1, dest_x1)
        top_expand = min(expand_height, source_y1, dest_y1)
        right_expand = min(
            expand_width,
            image_width - source_x2,
            image_width - dest_x2,
        )
        bottom_expand = min(
            expand_height,
            image_height - source_y2,
            image_height - dest_y2,
        )

        return {
            "source_index": source_index,
            "source_x1": source_x1 - left_expand,
            "source_y1": source_y1 - top_expand,
            "source_x2": source_x2 + right_expand,
            "source_y2": source_y2 + bottom_expand,
            "dest_x1": dest_x1 - left_expand,
            "dest_y1": dest_y1 - top_expand,
            "dest_x2": dest_x2 + right_expand,
            "dest_y2": dest_y2 + bottom_expand,
            "bbox": bbox,
        }
