#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Image preprocessing for the Depth Anything depth-estimation models.

Split into a per-image stage (``process_image``) and a shared batch stage
(``process_batch``), so the per-image stage can be baked into an export. Output order
matches input order.

The per-image stage returns a float RGB tensor in [0, 255] of shape ``(3, H, W)``,
resized per one of ``RESIZE_METHODS``:
  - ``upper_bound_resize``: longest side to ``process_res``, aspect preserved, both sides
    rounded to a multiple of ``patch_size``.
  - ``lower_bound_resize``: shortest side to ``process_res``, aspect preserved, both sides
    rounded to a multiple of ``patch_size``.
  - ``square_resize``: resize to exactly ``(process_res, process_res)``.

Resizing uses torchvision bilinear interpolation (the codebase-standard preprocessing
resize). Normalization happens later in ``process_batch``.

Batch stage (``process_batch``): center-crop to the smallest size in the batch, stack,
ImageNet-normalize.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal, get_args

import torch
from torch import Tensor
from torchvision.transforms.v2 import functional as transforms_functional
from torchvision.transforms.v2.functional import InterpolationMode

logger = logging.getLogger(__name__)

NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
ResizeMethod = Literal[
    "upper_bound_resize",
    "lower_bound_resize",
    "square_resize",
]
RESIZE_METHODS = get_args(ResizeMethod)


def process_image(
    img: Tensor,
    *,
    process_res: int,
    patch_size: int,
    process_res_method: ResizeMethod = "square_resize",
) -> Tensor:
    """Preprocesses one image for Depth Anything depth inference.

    Resizes according to ``process_res_method``, returning a float32 RGB tensor in
    [0, 255]. Normalization happens later in ``process_batch``.

    Args:
        img: Image of shape ``(C, H, W)``. Uint8 is read as [0, 255], float as [0, 1];
            other dtypes raise.
        process_res: Target resolution for the resize.
        patch_size: The aspect-preserving methods round both output dimensions to the
            nearest multiple of this value.
        process_res_method: One of ``RESIZE_METHODS``:
            ``"upper_bound_resize"`` scales the longest side to ``process_res``,
            ``"lower_bound_resize"`` scales the shortest side to ``process_res`` (both
            preserve the aspect ratio and round each side to a multiple of
            ``patch_size``), and ``"square_resize"`` resizes to exactly
            ``(process_res, process_res)``.

    Returns:
        A float32 RGB tensor in [0, 255] of shape ``(3, H, W)``.
    """
    if process_res_method not in RESIZE_METHODS:
        raise ValueError(
            f"Unsupported process_res_method '{process_res_method}'. Supported "
            f"methods are: {RESIZE_METHODS}."
        )
    image = _to_float_rgb(img)
    if process_res_method == "square_resize":
        return _resize(image, new_h=process_res, new_w=process_res)
    image = _resize_bound(image, target_size=process_res, method=process_res_method)
    image = _resize_to_patch_multiple(image, patch=patch_size)
    return image


def process_batch(images: Sequence[Tensor]) -> Tensor:
    """Stacks per-image tensors into a normalized, model-ready batch.

    Args:
        images: Float RGB tensors in [0, 255] of shape ``(3, H, W)`` from
            ``process_image``. Differing sizes are center-cropped to the smallest in the
            batch.

    Returns:
        A normalized batch of shape ``(N, 3, H, W)``.
    """
    if not images:
        raise ValueError("The input image list is empty.")
    unified = _unify_sizes(list(images))
    return _normalize_image(torch.stack(unified))


def _unify_sizes(images: list[Tensor]) -> list[Tensor]:
    """Center-crops all images to the smallest height and width in the batch."""
    sizes = {tuple(img.shape[-2:]) for img in images}
    if len(sizes) <= 1:
        return images
    min_h = min(h for h, _ in sizes)
    min_w = min(w for _, w in sizes)
    logger.warning(
        f"Images in batch have different sizes {sorted(sizes)}; "
        f"center-cropping all to smallest ({min_h},{min_w})"
    )
    return [
        transforms_functional.center_crop(img, output_size=[min_h, min_w])
        for img in images
    ]


def _to_float_rgb(img: Tensor) -> Tensor:
    """Converts a ``(C, H, W)`` image to a float32 RGB tensor in [0, 255].

    Uint8 values are kept exactly; float tensors are assumed to be in [0, 1] and scaled
    to [0, 255]; other dtypes raise. Grayscale is expanded to 3 channels, alpha dropped.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected image shape (C, H, W), got {tuple(img.shape)}.")
    channels = img.shape[0]
    if channels == 1:
        img = img.expand(3, -1, -1)
    elif channels == 4:
        img = img[:3]
    elif channels != 3:
        raise ValueError(f"Expected an image with 1, 3, or 4 channels, got {channels}.")
    if img.dtype == torch.uint8:
        return img.to(torch.float32)
    if not img.dtype.is_floating_point:
        raise ValueError(
            f"Unsupported image dtype {img.dtype}. Supported dtypes are uint8 "
            "and floating-point types."
        )
    return img.to(torch.float32) * 255.0


def _normalize_image(img: Tensor) -> Tensor:
    # Multiply by 1/255 instead of dividing by 255 to match
    # `to_dtype(..., scale=True)` bit-exactly.
    img = transforms_functional.normalize(
        img.mul(1.0 / 255.0),
        mean=list(NORMALIZE_MEAN),
        std=list(NORMALIZE_STD),
    )
    return img


def _resize_bound(img: Tensor, *, target_size: int, method: str) -> Tensor:
    """Resizes so the longest ("upper_bound_*") or shortest ("lower_bound_*")
    side matches ``target_size``, preserving the aspect ratio."""
    h, w = img.shape[-2:]
    bound = max(h, w) if method.startswith("upper_bound") else min(h, w)
    if bound == target_size:
        return img
    scale = target_size / float(bound)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return _resize(img, new_h=new_h, new_w=new_w)


def _resize_to_patch_multiple(img: Tensor, *, patch: int) -> Tensor:
    """Rounds each dimension to the nearest multiple of ``patch`` via a small resize."""

    def nearest_multiple(x: int) -> int:
        down = (x // patch) * patch
        up = down + patch
        return up if abs(up - x) <= abs(x - down) else down

    h, w = img.shape[-2:]
    new_h = max(1, nearest_multiple(h))
    new_w = max(1, nearest_multiple(w))
    if new_w == w and new_h == h:
        return img
    return _resize(img, new_h=new_h, new_w=new_w)


def _resize(img: Tensor, *, new_h: int, new_w: int) -> Tensor:
    """Resizes a ``(C, H, W)`` image to ``(new_h, new_w)`` with torchvision bilinear."""
    resized: Tensor = transforms_functional.resize(
        img,
        size=[new_h, new_w],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return resized
