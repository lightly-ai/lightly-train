#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
from albumentations.augmentations.crops.transforms import RandomCrop
from lightning_utilities.core.imports import RequirementCache
from numpy.typing import NDArray

from lightly_train.types import NDArrayBBoxes, NDArrayImage

ALBUMENTATIONS_GEQ_1_4_21 = RequirementCache("albumentations>=1.4.21")
ALBUMENTATIONS_GEQ_1_4_15 = RequirementCache("albumentations>=1.4.15")
ALBUMENTATIONS_GEQ_1_4_11 = RequirementCache("albumentations>=1.4.11")

if not ALBUMENTATIONS_GEQ_1_4_21:
    import albumentations.augmentations.crops.functional as F


class RandomIoUCropBase(RandomCrop):  # type: ignore[misc]
    """Random IoU crop transformation, similar to torchvision's RandomIoUCrop.

    Args:
        min_scale: Minimum scale for the crop.
        max_scale: Maximum scale for the crop.
        min_aspect_ratio: Minimum aspect ratio for the crop.
        max_aspect_ratio: Maximum aspect ratio for the crop.
        sampler_options: List of minimal IoU (Jaccard) overlap between all the boxes and
            a cropped image.
        crop_trials: Number of trials for generating a crop.
        iou_trials: Number of trials for generating a crop with a valid IoU.
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
        p: float = 1.0,
    ):
        # Hardcode required args for RandomCrop, height and width will be set dynamically.
        if ALBUMENTATIONS_GEQ_1_4_21:
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
        else:
            super().__init__(
                height=1,
                width=1,
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
        self.p = p


def _get_crop_coords(
    orig_shape: tuple[int, int],
    min_scale: float,
    max_scale: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    sampler_options: Sequence[float],
    crop_trials: int,
    iou_trials: int,
    orig_bboxes: NDArray[np.float32],
) -> tuple[int, int, int, int]:
    """Find crop coordinates according to IoU constraints.

    Args:
        orig_shape: (height, width) of the original image.
        min_scale: Minimum scale for the crop.
        max_scale: Maximum scale for the crop.
        min_aspect_ratio: Minimum aspect ratio for the crop.
        max_aspect_ratio: Maximum aspect ratio for the crop.
        sampler_options: List of minimal IoU (Jaccard) overlap between all the boxes and
            a cropped image.
        crop_trials: Number of trials for generating a crop.
        iou_trials: Number of trials for generating a crop with a valid IoU.
        orig_bboxes: Bounding boxes as ndarray of shape (N, 4), normalized [0, 1].

    Returns:
        Crop coordinates as (x_min, y_min, x_max, y_max).
    """
    orig_h, orig_w = orig_shape
    for _ in range(iou_trials):
        min_jaccard_overlap = random.choice(sampler_options)
        if min_jaccard_overlap >= 1.0:
            return (0, 0, orig_w, orig_h)

        for _ in range(crop_trials):
            r = np.random.uniform(min_scale, max_scale, size=2)
            new_w = int(orig_w * r[0])
            new_h = int(orig_h * r[1])
            aspect_ratio = new_w / new_h if new_h > 0 else 0.0

            if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                continue

            r = np.random.uniform(0, 1, size=2)
            left = int((orig_w - new_w) * r[0])
            top = int((orig_h - new_h) * r[1])
            right = left + new_w
            bottom = top + new_h

            if left == right or top == bottom:
                continue

            bboxes_absolute = orig_bboxes * np.array([orig_w, orig_h, orig_w, orig_h])

            cx = (bboxes_absolute[:, 0] + bboxes_absolute[:, 2]) / 2
            cy = (bboxes_absolute[:, 1] + bboxes_absolute[:, 3]) / 2
            is_within_crop_area = (
                (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
            )

            if not is_within_crop_area.any():
                continue

            bboxes_within = bboxes_absolute[is_within_crop_area]
            ious = [_iou(bbox, (left, top, right, bottom)) for bbox in bboxes_within]
            if max(ious) < min_jaccard_overlap:
                continue

            return (left, top, right, bottom)

    return (0, 0, orig_w, orig_h)


class RandomIoUCropV3(RandomIoUCropBase):
    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        orig_image_shape = data["image"].shape[:2]
        orig_bboxes = np.array(data["bboxes"][:, :4])
        if np.random.rand() < self.p:
            crop_coords = _get_crop_coords(
                orig_shape=orig_image_shape,
                min_scale=self.min_scale,
                max_scale=self.max_scale,
                min_aspect_ratio=self.min_aspect_ratio,
                max_aspect_ratio=self.max_aspect_ratio,
                sampler_options=self.options,
                crop_trials=self.crop_trials,
                iou_trials=self.iou_trials,
                orig_bboxes=orig_bboxes,
            )
            return {"crop_coords": crop_coords, "pad_params": None}
        else:
            return {
                "crop_coords": (0, 0, orig_image_shape[1], orig_image_shape[0]),
                "pad_params": None,
            }


class RandomIoUCropV2(RandomIoUCropBase):
    def update_params(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        super().update_params(params, **kwargs)
        # Allow both 'shape' or ('rows', 'cols') in params.
        if "shape" in params:
            orig_image_shape = params["shape"][:2]
        else:
            orig_image_shape = (params["rows"], params["cols"])

        if np.random.rand() < self.p:
            # If no bboxes are provided, create empty 2D array.
            if "bboxes" not in kwargs or len(kwargs["bboxes"]) == 0:
                kwargs["bboxes"] = np.zeros((0, 4), dtype=np.float32)
            orig_bboxes = np.array(kwargs["bboxes"])[:, :4]

            crop_coords = _get_crop_coords(
                orig_shape=orig_image_shape,
                min_scale=self.min_scale,
                max_scale=self.max_scale,
                min_aspect_ratio=self.min_aspect_ratio,
                max_aspect_ratio=self.max_aspect_ratio,
                sampler_options=self.options,
                crop_trials=self.crop_trials,
                iou_trials=self.iou_trials,
                orig_bboxes=orig_bboxes,
            )
            params = params.copy()
            params.update(
                {"crop_coords": crop_coords, "orig_img_shape": orig_image_shape}
            )

        else:
            params = params.copy()
            params.update(
                {
                    "crop_coords": (
                        0,
                        0,
                        orig_image_shape[1],
                        orig_image_shape[0],
                    ),
                    "orig_img_shape": orig_image_shape,
                }
            )
        return params

    def apply(self, img: NDArrayImage, **params: Any) -> NDArrayImage:
        crop_coords = params["crop_coords"]
        x_min, y_min, x_max, y_max = crop_coords
        cropped = F.crop(
            img,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )
        return cropped  # type: ignore[no-any-return]

    def apply_to_bboxes(self, bboxes: NDArrayBBoxes, **params: Any) -> NDArrayBBoxes:
        crop_coords = params["crop_coords"]
        x_min, y_min, x_max, y_max = crop_coords

        cropped = F.crop_bboxes_by_coords(
            bboxes,
            crop_coords=(x_min, y_min, x_max, y_max),
            image_shape=params["orig_img_shape"],
        )
        return cropped  # type: ignore[no-any-return]

    def apply_to_keypoints(
        self, keypoints: NDArray[np.float32], **params: Any
    ) -> NDArray[np.float32]:
        raise NotImplementedError("Keypoints are not supported by RandomIoUCropV2.")


class RandomIoUCropV1(RandomIoUCropBase):
    def update_params(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        super().update_params(params, **kwargs)
        # Allow both 'shape' or ('rows', 'cols') in params.
        if "shape" in params:
            orig_image_shape = params["shape"][:2]
        else:
            orig_image_shape = (params["rows"], params["cols"])

        if np.random.rand() < self.p:
            # If no bboxes are provided, create empty 2D array.
            if "bboxes" not in kwargs or len(kwargs["bboxes"]) == 0:
                kwargs["bboxes"] = np.zeros((0, 4), dtype=np.float32)
            orig_bboxes = np.array(kwargs["bboxes"])[:, :4]

            crop_coords = _get_crop_coords(
                orig_shape=orig_image_shape,
                min_scale=self.min_scale,
                max_scale=self.max_scale,
                min_aspect_ratio=self.min_aspect_ratio,
                max_aspect_ratio=self.max_aspect_ratio,
                sampler_options=self.options,
                crop_trials=self.crop_trials,
                iou_trials=self.iou_trials,
                orig_bboxes=orig_bboxes,
            )
            params = params.copy()
            params.update(
                {"crop_coords": crop_coords, "orig_img_shape": orig_image_shape}
            )
        else:
            params = params.copy()
            params.update(
                {
                    "crop_coords": (
                        0,
                        0,
                        orig_image_shape[1],
                        orig_image_shape[0],
                    ),
                    "orig_img_shape": orig_image_shape,
                }
            )
        return params

    def apply(self, img: NDArrayImage, **params: Any) -> NDArrayImage:
        crop_coords = params["crop_coords"]
        x_min, y_min, x_max, y_max = crop_coords
        cropped = F.crop(
            img,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )
        return cropped  # type: ignore[no-any-return]

    def apply_to_bbox(
        self, bbox: tuple[float, float, float, float], **params: Any
    ) -> tuple[float, float, float, float]:
        crop_coords = params["crop_coords"]
        x_min, y_min, x_max, y_max = crop_coords

        if not ALBUMENTATIONS_GEQ_1_4_11:
            tr_bbox = F.crop_bbox_by_coords(
                bbox,
                crop_coords=(x_min, y_min, x_max, y_max),
                crop_height=y_max - y_min,
                crop_width=x_max - x_min,
                rows=params["orig_img_shape"][0],
                cols=params["orig_img_shape"][1],
            )
            return tr_bbox  # type: ignore[no-any-return]
        else:
            tr_bbox = F.crop_bbox_by_coords(
                bbox,
                crop_coords=(x_min, y_min, x_max, y_max),
                rows=params["orig_img_shape"][0],
                cols=params["orig_img_shape"][1],
            )
            return tr_bbox  # type: ignore[no-any-return]

    def apply_to_keypoint(
        self, keypoint: NDArray[np.float32], **params: Any
    ) -> NDArray[np.float32]:
        raise NotImplementedError("Keypoints are not supported by RandomIoUCropV1.")


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
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


if ALBUMENTATIONS_GEQ_1_4_21:
    RandomIoUCrop = RandomIoUCropV3  # type: ignore[misc]
elif ALBUMENTATIONS_GEQ_1_4_15:
    RandomIoUCrop = RandomIoUCropV2  # type: ignore[misc]
else:
    RandomIoUCrop = RandomIoUCropV1  # type: ignore[misc]
