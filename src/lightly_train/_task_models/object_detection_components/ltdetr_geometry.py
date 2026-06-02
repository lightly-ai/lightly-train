#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Geometry helpers for LT-DETR object detection models."""

from __future__ import annotations


def ltdetr_image_size_divisor(patch_size: int) -> int:
    """Return the LT-DETR image-size divisor for a patch size.

    LT-DETR uses a feature map that is 2x smaller than the ViT patch grid, so
    image sizes must be divisible by 2 * patch_size.
    """

    return 2 * patch_size
