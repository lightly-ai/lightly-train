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


class RandomIoUCrop(RandomCrop):  # type: ignore[misc]
    """Random IoU crop transformation, similar to torchvision's RandomIoUCrop.

    Args:
        min_scale: Minimum scale for the crop.
        max_scale: Maximum scale for the crop.
        min_aspect_ratio: Minimum aspect ratio for the crop.
        max_aspect_ratio: Maximum aspect ratio for the crop.
        sampler_options: List of minimal IoU (Jaccard) overlap between all the boxes and
            a cropped image.
        trials: Number of attempts to find a crop for a given value of minimal IoU.
    """

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Sequence[float] | None = None,
        crop_trials: int = 40,
        iou_trials: int = 1000,
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
        self.crop_trials = crop_trials
        self.iou_trials = iou_trials

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        orig_image_shape = data["image"].shape[:2]
        orig_h, orig_w = orig_image_shape
        orig_bboxes = np.array(data["bboxes"][:, :4])

        for _ in range(self.iou_trials):
            # 1. Sample a minimum IoU value.
            min_jaccard_overlap = random.choice(self.options)
            if min_jaccard_overlap >= 1.0:
                return {"crop_coords": (0, 0, orig_h, orig_w), "pad_params": None}

            for _ in range(self.crop_trials):
                # Sample scales in range [min_scale, max_scale]
                r = np.random.uniform(self.min_scale, self.max_scale, size=2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h

                # If the aspect ratio is not in the desired range, skip this trial.
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # Randomly place the crop.
                r = np.random.uniform(0, 1, size=2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h

                # If zero area crop, skip this trial.
                if left == right or top == bottom:
                    continue

                # Convert bboxes from [0, 1] to absolute image coordinates.
                bboxes_absolute = orig_bboxes * np.array(
                    [orig_w, orig_h, orig_w, orig_h]
                )

                # Get bboxes whose center is in the crop.
                cx = (bboxes_absolute[:, 0] + bboxes_absolute[:, 2]) / 2
                cy = (bboxes_absolute[:, 1] + bboxes_absolute[:, 3]) / 2
                is_within_crop_area = (
                    (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                )

                # If no bbox is in the crop, skip this trial.
                if not is_within_crop_area.any():
                    continue

                # Check if at least one bbox has the required IoU with the crop.
                bboxes_within = bboxes_absolute[is_within_crop_area]
                ious = [
                    self._iou(bbox, (left, top, right, bottom))
                    for bbox in bboxes_within
                ]
                if max(ious) < min_jaccard_overlap:
                    continue

                return {"crop_coords": (top, left, new_h, new_w), "pad_params": None}

    def _iou(self, box_a: Sequence[float], box_b: Sequence[float]) -> float:
        """Compute intersection over union of two boxes.

        Args:
            box_a: (left, top, right, bottom) of box A.
            box_b: (left, top, right, bottom) of box B.

        Returns:
            IoU value.
        """
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
