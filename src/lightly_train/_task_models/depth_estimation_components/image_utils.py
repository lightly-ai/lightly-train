#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Image preprocessing for Depth Anything 3.

Exposes the preprocessing pipeline as two stages: a per-image stage
(``_process_image``) and a batch stage (``_process_batch``). The order of outputs
matches the input order. The square center-crop step is omitted for "*crop" methods.

Pipeline:
  1) ``_process_image`` (per image):
     a) Convert to a float32 RGB tensor in [0, 255]
     b) Boundary resize (upper/lower bound, preserving aspect ratio)
     c) Enforce divisibility by PATCH_SIZE:
        - "*resize" methods: each dimension is rounded to nearest multiple
          (may up/downscale a few px)
        - "*crop"   methods: each dimension is floored to nearest multiple via
          center crop
  2) ``_process_batch`` (per batch):
     a) Center-crop all images to the smallest size in the batch
     b) Stack into (N, 3, H, W)
     c) Apply ImageNet normalization
"""

from __future__ import annotations

import logging
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
    "upper_bound_crop",
    "lower_bound_resize",
    "lower_bound_crop",
)


def _process_image(
    img: Tensor,
    *,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> Tensor:
    """Processes a single image into a resized, patch-divisible tensor.

    Args:
        img: Image with shape ``(C, H, W)``. Integer tensors are interpreted
            in their dtype's value range (e.g. uint8 in [0, 255]) and float
            tensors in [0, 1].
        process_res: Target size for the boundary resize.
        process_res_method: One of ``RESIZE_METHODS``.

    Returns:
        A float32 RGB tensor in [0, 255] with shape ``(3, H, W)``. Normalization
        is applied later in ``_process_batch``.
    """
    if process_res_method not in RESIZE_METHODS:
        raise ValueError(
            f"Unsupported process_res_method '{process_res_method}'. Supported "
            f"methods are: {RESIZE_METHODS}."
        )
    image = _to_rgb_tensor(img)
    image = _resize_bound(image, target_size=process_res, method=process_res_method)
    if process_res_method.endswith("resize"):
        image = _make_divisible_by_resize(image, patch=PATCH_SIZE)
    else:
        image = _make_divisible_by_crop(image, patch=PATCH_SIZE)
    return image


def _process_batch(images: Sequence[Tensor]) -> Tensor:
    """Processes images from ``_process_image`` into a model-ready batch.

    Args:
        images: Float32 RGB tensors in [0, 255] with shape ``(3, H, W)``, as
            returned by ``_process_image``. Images with different sizes are
            center-cropped to the smallest size in the batch.

    Returns:
        A normalized batch tensor of shape ``(N, 3, H, W)``.
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


def _to_rgb_tensor(img: Tensor) -> Tensor:
    """Converts a ``(C, H, W)`` image tensor to a float32 RGB tensor in [0, 255].

    The pipeline operates in 0-255 float space so the resize arithmetic is
    bit-identical to the official uint8-based cv2 pipeline; uint8 values are kept
    exactly, other integer dtypes are rescaled from their value range, and float
    tensors are assumed to be in [0, 1]. Single-channel images are repeated to
    three channels and an alpha channel is dropped.
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
    out: Tensor = transforms_functional.to_dtype(img, dtype=torch.float32, scale=True)
    return out.mul_(255.0)


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
    # As in cv2, the kernel choice is joint over both axes: a resize that enlarges
    # either axis uses cubic, otherwise area.
    interpolation = "cubic" if new_h > h or new_w > w else "area"
    return _resize_to(img, new_h=new_h, new_w=new_w, method=interpolation)


def _resize_to(img: Tensor, *, new_h: int, new_w: int, method: str) -> Tensor:
    """Resizes a ``(C, H, W)`` float image to ``(new_h, new_w)``.

    Reproduces the cv2 resampling the official DA3 reference depends on without an
    OpenCV dependency: ``"area"`` matches ``INTER_AREA`` and ``"cubic"`` matches
    ``INTER_CUBIC``. Both are separable, so per-axis ``(dst, src)`` weight matrices
    are built and applied along the height then the width. The result is rounded
    back to the integer grid (in float, without a dtype change), reproducing cv2's
    round-back-to-uint8 so values match the official pipeline bit-exactly; skipping
    this quantization shifts the model input enough to visibly change the predicted
    depth.

    Args:
        img: Float image tensor with shape ``(C, H, W)``.
        new_h: Target height.
        new_w: Target width.
        method: Interpolation method, either ``"area"`` or ``"cubic"``.

    Returns:
        The resized image with shape ``(C, new_h, new_w)``.
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
    row_weights = weights(src=h, dst=new_h, device=img.device)
    col_weights = weights(src=w, dst=new_w, device=img.device)
    x = torch.einsum("ph,chw->cpw", row_weights, img)
    x = torch.einsum("qw,cpw->cpq", col_weights, x)
    return x.round().clamp(min=0.0, max=255.0)


def _area_weights(*, src: int, dst: int, device: torch.device) -> Tensor:
    """Builds the ``(dst, src)`` cv2 ``INTER_AREA`` overlap-weight matrix for one axis.

    Output cell ``i`` spans ``[i * scale, (i + 1) * scale)`` in source coordinates,
    where ``scale = src / dst``; the weight for source pixel ``j`` is the length of
    the overlap with ``[j, j + 1)`` divided by ``scale``. A ``scale`` of 1 yields the
    identity, so an unchanged axis is passed through exactly.
    """
    scale = src / dst
    out_idx = torch.arange(dst, dtype=torch.float32, device=device)[:, None]
    src_idx = torch.arange(src, dtype=torch.float32, device=device)[None, :]
    lo = torch.maximum(out_idx * scale, src_idx)
    hi = torch.minimum((out_idx + 1) * scale, src_idx + 1)
    return (hi - lo).clamp(min=0) / scale


def _cubic_weights(*, src: int, dst: int, device: torch.device) -> Tensor:
    """Builds the ``(dst, src)`` cv2 ``INTER_CUBIC`` weight matrix for one axis.

    Uses the Keys cubic convolution kernel with ``a = -0.75`` (cv2's constant) and
    half-pixel sample centers. Output ``i`` samples source coordinate
    ``(i + 0.5) * scale - 0.5`` from its four nearest taps; taps falling outside the
    image clamp to the border (cv2's replicate behavior), accumulating their weight.
    The four kernel weights sum to one, so no renormalization is needed.
    """
    a = -0.75
    scale = src / dst
    out_idx = torch.arange(dst, dtype=torch.float32, device=device)
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

    weights = torch.zeros(dst, src, dtype=torch.float32, device=device)
    rows = torch.arange(dst, device=device)
    for offset in (-1, 0, 1, 2):
        taps = (base + offset).long().clamp(min=0, max=src - 1)
        weights[rows, taps] += kernel(frac - offset)
    return weights


def _make_divisible_by_crop(img: Tensor, *, patch: int) -> Tensor:
    """Floors each dimension to the nearest multiple of ``patch`` via center crop.

    Example: 504x377 -> 504x364.
    """
    h, w = img.shape[-2:]
    new_h = (h // patch) * patch
    new_w = (w // patch) * patch
    if new_w == w and new_h == h:
        return img
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    # The annotation asserts the type at the torchvision boundary: torchvision
    # ships no py.typed marker, so mypy sees its functions as returning Any.
    cropped: Tensor = transforms_functional.crop(
        img, top=top, left=left, height=new_h, width=new_w
    )
    return cropped


def _make_divisible_by_resize(img: Tensor, *, patch: int) -> Tensor:
    """Rounds each dimension to the nearest multiple of ``patch`` via small resize."""

    def nearest_multiple(x: int) -> int:
        down = (x // patch) * patch
        up = down + patch
        return up if abs(up - x) <= abs(x - down) else down

    h, w = img.shape[-2:]
    new_h = max(1, nearest_multiple(h))
    new_w = max(1, nearest_multiple(w))
    if new_w == w and new_h == h:
        return img
    # As in cv2, the kernel choice is joint over both axes: a resize that enlarges
    # either axis uses cubic, otherwise area.
    interpolation = "cubic" if new_h > h or new_w > w else "area"
    return _resize_to(img, new_h=new_h, new_w=new_w, method=interpolation)
