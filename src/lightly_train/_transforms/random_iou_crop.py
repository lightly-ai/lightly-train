#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
from albumentations.augmentations.crops.transforms import RandomCrop
from numpy.typing import NDArray


class RandomIoUCrop(RandomCrop):  # type: ignore[misc]
    """Random IoU crop transformation, similar to torchvision's RandomIoUCrop.

    Args:
        min_scale: Minimum scale for the crop.
        max_scale: Maximum scale for the crop.
        min_aspect_ratio: Minimum aspect ratio for the crop.
        max_aspect_ratio: Maximum aspect ratio for the crop.
        sampler_options: List of minimal IoU (Jaccard) overlap between all the boxes and a cropped image.
        trials: Number of attempts to find a crop for a given value of minimal IoU.
    """

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Sequence[float] | None = None,
        trials: int = 40,
    ):
        # Hardcode required args for RandomCrop
        super().__init__(
            height=1,
            width=1,
            pad_if_needed=False,
            pad_position="center",
            border_mode=0,
            fill=0.0,
            fill_mask=0.0,
            p=1.0,
        )
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.options = (
            list(sampler_options)
            if sampler_options is not None
            else [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        )
        self.trials = trials

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        image_shape = data["image"].shape[:2]
        h, w = image_shape
        bboxes = data["bboxes"]

        while True:
            min_iou = random.choice(self.options)

            # Don't crop.
            if min_iou >= 1.0:
                return {
                    "crop_coords": (0, 0, w, h),
                    "pad_params": None,
                }

            for _ in range(self.trials):
                r = self.min_scale + (self.max_scale - self.min_scale) * np.random.rand(2)
                new_w = int(w * r[0])
                new_h = int(h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # Check for 0 area crops.
                r = np.random.rand(2)
                left = int((w - new_w) * r[0])
                top = int((h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # Check for any valid boxes with centers within the crop area.
                cx = 0.5 * (bboxes[..., 0] + bboxes[..., 2])
                cy = 0.5 * (bboxes[..., 1] + bboxes[..., 3])
                is_within_crop_area = (
                    (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                )
                if not is_within_crop_area.any():
                    continue

                # Check that at least one box has the required IoU with the crop.
                bboxes = bboxes[is_within_crop_area]
                ious = [
                    self._bboxes_iou(
                        bboxes, np.array([[left, top, right, bottom]], dtype=bboxes.dtype)
                    )
                ]
                
                if ious.max() < min_iou:
                    continue

                return {
                    "crop_coords": (left, top, right, bottom),
                    "pad_params": None,
                }
            # Fallback
            print("Fallback crop")
            return {
                "crop_coords": (0, 0, w, h),
                "pad_params": None,
            }

    def _bboxes_iou(
        self, box1: NDArray[np.float32], box2: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        # Calculate intersection areas
        inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
        inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
        inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
        inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        # Calculate union areas
        area_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = area_box1[:, None] + area_box2[None, :] - inter_area

        # Compute IoU
        iou = inter_area / union_area
        assert isinstance(iou, np.ndarray)
        assert iou.dtype == np.float32
        return iou
