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

Split into a per-image stage (one function per model family) and a shared batch stage,
so the per-image stage can be baked into an export. Output order matches input order.

Per-image stage, each returning a float RGB tensor in [0, 255] of shape ``(3, H, W)``:
  - ``process_image_dav3``: boundary resize, then round each side to a multiple of
    ``PATCH_SIZE``. Float32, rounding to the integer grid after every resize.
  - ``process_image_dav2``: a single cubic resize so the shorter side reaches the
    target, both sides a multiple of ``PATCH_SIZE``. Float64, no intermediate rounding.

Batch stage (``process_batch``): center-crop to the smallest size in the batch, stack,
ImageNet-normalize.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torchvision.transforms.v2 import functional as transforms_functional

logger = logging.getLogger(__name__)

NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 14
RESIZE_METHODS = (
    "upper_bound_resize",
    "lower_bound_resize",
)


def process_image_dav3(
    img: Tensor,
    *,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> Tensor:
    """Preprocesses one image for Depth Anything 3.

    Resizes so one side matches ``process_res`` (preserving aspect ratio), then rounds
    both sides to a multiple of ``PATCH_SIZE``. Runs in float32, rounding to the integer
    grid after each resize. Normalization happens later in ``process_batch``.

    Args:
        img: Image of shape ``(C, H, W)``. Uint8 is read as [0, 255], float as [0, 1];
            other dtypes raise.
        process_res: Target size for the boundary resize.
        process_res_method: One of ``RESIZE_METHODS``.

    Returns:
        A float32 RGB tensor in [0, 255] of shape ``(3, H, W)``.
    """
    if process_res_method not in RESIZE_METHODS:
        raise ValueError(
            f"Unsupported process_res_method '{process_res_method}'. Supported "
            f"methods are: {RESIZE_METHODS}."
        )
    image = _to_float_rgb(img)
    image = _resize_bound(image, target_size=process_res, method=process_res_method)
    image = _resize_to_patch_multiple(image, patch=PATCH_SIZE)
    return image


def process_image_dav2(
    img: Tensor,
    *,
    process_res: int = 518,
) -> Tensor:
    """Preprocesses one image for Depth Anything V2.

    Lower-bound resize so the shorter side reaches ``process_res``, both sides rounded
    to a multiple of ``PATCH_SIZE``, as a single cubic resize with no intermediate
    rounding. Stays in [0, 255] float; normalization happens later in ``process_batch``.

    Args:
        img: Image of shape ``(C, H, W)``. Uint8 is read as [0, 255], float as [0, 1];
            other dtypes raise.
        process_res: Target size for the shorter side.

    Returns:
        A float64 RGB tensor in [0, 255] of shape ``(3, H', W')``, H'/W' multiples of
        ``PATCH_SIZE``.
    """
    # Float64 mirrors the official float64 image and roughly halves the resize's
    # accumulation error vs float32, at negligible cost; DAv3 stays in float32.
    image = _to_float_rgb(img).to(torch.float64)
    h, w = image.shape[-2:]
    new_h, new_w = _dav2_target_size(
        h=h, w=w, process_res=process_res, patch=PATCH_SIZE
    )
    if (new_h, new_w) == (h, w):
        return image
    return _resize(image, new_h=new_h, new_w=new_w, method="cubic", round_to_grid=False)


def process_batch(images: Sequence[Tensor]) -> Tensor:
    """Stacks per-image tensors into a normalized, model-ready batch.

    Args:
        images: Float RGB tensors in [0, 255] of shape ``(3, H, W)`` from
            ``process_image_dav3``/``process_image_dav2``. Differing sizes are
            center-cropped to the smallest in the batch.

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
    return _resize_to_grid(img, new_h=new_h, new_w=new_w)


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
    return _resize_to_grid(img, new_h=new_h, new_w=new_w)


def _resize_to_grid(img: Tensor, *, new_h: int, new_w: int) -> Tensor:
    """Resizes to ``(new_h, new_w)`` and rounds back to the integer grid (DAv3 path).

    Kernel is joint over both axes: cubic if either axis enlarges, otherwise area.
    """
    h, w = img.shape[-2:]
    method = "cubic" if new_h > h or new_w > w else "area"
    return _resize(img, new_h=new_h, new_w=new_w, method=method, round_to_grid=True)


def _dav2_target_size(
    *, h: int, w: int, process_res: int, patch: int
) -> tuple[int, int]:
    """Computes the DAv2 lower-bound resize target size.

    Scales so the shorter side reaches ``process_res``, then rounds each side to a
    multiple of ``patch`` (ceiling up if rounding down would drop below ``process_res``).

    Returns:
        The ``(new_h, new_w)`` target size, both multiples of ``patch``.
    """
    scale = max(process_res / h, process_res / w)

    def _constrain(x: float) -> int:
        y = int(round(x / patch) * patch)
        if y < process_res:
            y = int(math.ceil(x / patch) * patch)
        return y

    return _constrain(scale * h), _constrain(scale * w)


def _resize(
    img: Tensor, *, new_h: int, new_w: int, method: str, round_to_grid: bool
) -> Tensor:
    """Resizes a ``(C, H, W)`` float image to ``(new_h, new_w)``, matching cv2 without it.

    ``"area"`` matches cv2 ``INTER_AREA`` and ``"cubic"`` matches ``INTER_CUBIC``. Both
    are separable: per-axis ``(dst, src)`` weight matrices (in ``img``'s dtype) applied
    over height then width.

    Args:
        img: Float image of shape ``(C, H, W)``.
        new_h: Target height.
        new_w: Target width.
        method: ``"area"`` or ``"cubic"``.
        round_to_grid: If True, round to the integer grid and clamp to [0, 255] (DAv3,
            for cv2 bit-exactness); DAv2 passes False to keep the unrounded float.

    Returns:
        The resized image of shape ``(C, new_h, new_w)``.
    """
    if method == "area":
        weights = _area_weights
    elif method == "cubic":
        weights = _cubic_weights
    else:
        raise ValueError(
            f"Unsupported resize method '{method}'. Supported methods are: "
            f"('area', 'cubic')."
        )
    h, w = img.shape[-2:]
    row_weights = weights(src=h, dst=new_h, device=img.device, dtype=img.dtype)
    col_weights = weights(src=w, dst=new_w, device=img.device, dtype=img.dtype)
    x = torch.einsum("ph,chw->cpw", row_weights, img)
    x = torch.einsum("qw,cpw->cpq", col_weights, x)
    if round_to_grid:
        x = x.round().clamp(min=0.0, max=255.0)
    return x


def _area_weights(
    *, src: int, dst: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Builds the ``(dst, src)`` cv2 ``INTER_AREA`` weight matrix for one axis.

    Output cell ``i`` spans ``[i*scale, (i+1)*scale)`` (``scale = src/dst``); pixel
    ``j``'s weight is its overlap with ``[j, j+1)`` divided by ``scale``. ``scale == 1``
    is the identity, so an unchanged axis passes through exactly.
    """
    scale = src / dst
    out_idx = torch.arange(dst, dtype=dtype, device=device)[:, None]
    src_idx = torch.arange(src, dtype=dtype, device=device)[None, :]
    lo = torch.maximum(out_idx * scale, src_idx)
    hi = torch.minimum((out_idx + 1) * scale, src_idx + 1)
    return (hi - lo).clamp(min=0) / scale


def _cubic_weights(
    *, src: int, dst: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tensor:
    """Builds the ``(dst, src)`` cv2 ``INTER_CUBIC`` weight matrix for one axis.

    Keys cubic kernel with ``a = -0.75`` (cv2's constant) and half-pixel centers: output
    ``i`` samples ``(i+0.5)*scale - 0.5`` from its four nearest taps, clamping
    out-of-range taps to the border (cv2 replicate). The four weights sum to one.

    Args:
        src: Source axis length.
        dst: Destination axis length.
        device: Device for the weights.
        dtype: Weight dtype. DAv2 uses float64 (less accumulation error vs the cv2
            reference); DAv3 uses float32.
    """
    a = -0.75
    scale = src / dst
    out_idx = torch.arange(dst, dtype=dtype, device=device)
    center = (out_idx + 0.5) * scale - 0.5
    base = torch.floor(center)
    frac = center - base

    def kernel(t: Tensor) -> Tensor:
        t = t.abs()
        inner = (a + 2) * t**3 - (a + 3) * t**2 + 1
        outer = a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a
        return torch.where(
            t <= 1, inner, torch.where(t < 2, outer, torch.zeros_like(t))
        )

    weights = torch.zeros(dst, src, dtype=dtype, device=device)
    rows = torch.arange(dst, device=device)
    for offset in (-1, 0, 1, 2):
        taps = (base + offset).long().clamp(min=0, max=src - 1)
        weights[rows, taps] += kernel(frac - offset)
    return weights
