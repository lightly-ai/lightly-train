#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import cv2
from albumentations import PadIfNeeded
from albumentations.augmentations.geometric import functional as fgeometric


class RandomZoomOut(PadIfNeeded):  # type: ignore[misc]
    """Approximate TorchVision's RandomZoomOut using Albumentations.

    Args:
        fill: Pixel fill value for padding. Can be an int, float, sequence, None, or dict.
        side_range: Range for the size of the output image as a multiple of the input size.
            Should be a sequence of two floats (min, max), e.g. (1.0, 4.0).
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        fill: float,
        side_range: tuple[float, float],
        p: float,
    ):
        super().__init__(
            min_height=1,  # Will be ignored and set dynamically based on input image size.
            min_width=1,  # Will be ignored and set dynamically based on input image size.
            position="random",
            border_mode=cv2.BORDER_CONSTANT,
            fill=fill,
            p=p,
        )
        self.fill = fill
        self.side_range = side_range
        self.p = p

        if self.side_range[0] < 1.0 or self.side_range[1] < self.side_range[0]:
            raise ValueError(
                f"side_range must be a sequence of two floats (min >= 1.0, max >= min), got {side_range}."
            )

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        scale = self.py_random.uniform(self.side_range[0], self.side_range[1])
        h, w = data["image"].shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        # This is copy-pasted from albumentations PadIfNeeded, with the exception of
        # setting min_height and min_width dynamically based on the input image size.
        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = (
            fgeometric.get_padding_params(
                image_shape=params["shape"][:2],
                min_height=new_h,
                min_width=new_w,
                pad_height_divisor=self.pad_height_divisor,
                pad_width_divisor=self.pad_width_divisor,
            )
        )

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = (
            fgeometric.adjust_padding_by_position(
                h_top=h_pad_top,
                h_bottom=h_pad_bottom,
                w_left=w_pad_left,
                w_right=w_pad_right,
                position=self.position,
                py_random=self.py_random,
            )
        )

        return {
            "pad_top": h_pad_top,
            "pad_bottom": h_pad_bottom,
            "pad_left": w_pad_left,
            "pad_right": w_pad_right,
        }
