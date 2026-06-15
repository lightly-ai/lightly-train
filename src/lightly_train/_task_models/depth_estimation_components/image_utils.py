#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Image preprocessing for the Depth Anything depth-estimation models.

The pipeline is split into a per-image stage and a shared batch stage so the
per-image stage can be baked into an export while the batch stage stays dynamic.
Output order always matches input order.

Per-image stage (one function per model family, both producing a float RGB
tensor in [0, 255] of shape ``(3, H, W)``):

  - ``process_image_dav3`` (Depth Anything 3):
      a) Convert to a float32 RGB tensor in [0, 255].
      b) Boundary resize (upper/lower bound, preserving aspect ratio).
      c) Round each dimension to the nearest multiple of ``PATCH_SIZE`` via a
         small resize.
    Every resize rounds back to the integer grid, reproducing the official
    uint8-based cv2 pipeline bit-exactly.

  - ``process_image_dav2`` (Depth Anything V2):
      A single cubic resize so the shorter side reaches the target, with both
      sides rounded to a multiple of ``PATCH_SIZE``. Runs in float64 with no
      intermediate rounding, matching the official float ``cv2.INTER_CUBIC``
      pipeline.

Batch stage (shared, ``process_batch``):
  a) Center-crop all images to the smallest size in the batch.
  b) Stack into ``(N, 3, H, W)``.
  c) Apply ImageNet normalization.
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
    """Processes a single image with the Depth Anything 3 preprocessing.

    Resizes the image so one side matches ``process_res`` (preserving the aspect
    ratio), then rounds both sides to a multiple of ``PATCH_SIZE``. Both resizes
    run in 0-255 float space and round back to the integer grid, matching the
    official uint8-based cv2 pipeline bit-exactly. ImageNet normalization is
    applied later in ``process_batch``.

    Args:
        img: Image with shape ``(C, H, W)``. Uint8 tensors are interpreted in
            [0, 255] and float tensors in [0, 1]; other dtypes raise.
        process_res: Target size for the boundary resize.
        process_res_method: One of ``RESIZE_METHODS``.

    Returns:
        A float32 RGB tensor in [0, 255] with shape ``(3, H, W)``.
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
    input_size: int = 518,
    patch: int = PATCH_SIZE,
) -> Tensor:
    """Processes a single image with the Depth Anything V2 preprocessing.

    Reproduces the official MiDaS-style transform: a lower-bound resize so the shorter
    side reaches ``input_size``, with both sides rounded to a multiple of ``patch``,
    applied as a single cubic resampling on the float image with no intermediate
    rounding (matching ``cv2.INTER_CUBIC`` on the float [0, 1] reference image). The
    result stays in [0, 255] float space; ImageNet normalization is applied later in
    ``process_batch``.

    Args:
        img: Image with shape ``(C, H, W)``. Uint8 tensors are interpreted in [0, 255]
            and float tensors in [0, 1]; other dtypes raise.
        input_size: Target size for the shorter side of the lower-bound resize.
        patch: The patch size; both output dimensions are rounded to a multiple of it.

    Returns:
        A float64 RGB tensor in [0, 255] with shape ``(3, H', W')`` where H' and W' are
        multiples of ``patch``.
    """
    image = _to_float_rgb(img).to(torch.float64)
    h, w = image.shape[-2:]
    new_h, new_w = _dav2_target_size(h=h, w=w, input_size=input_size, patch=patch)
    if (new_h, new_w) == (h, w):
        return image
    return _resize(image, new_h=new_h, new_w=new_w, method="cubic", round_to_grid=False)


def process_batch(images: Sequence[Tensor]) -> Tensor:
    """Processes per-image tensors into a model-ready batch.

    Args:
        images: Float RGB tensors in [0, 255] with shape ``(3, H, W)``, as returned by
            ``process_image_dav3`` or ``process_image_dav2``. Images with different
            sizes are center-cropped to the smallest size in the batch.

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


def _to_float_rgb(img: Tensor) -> Tensor:
    """Converts a ``(C, H, W)`` image tensor to a float32 RGB tensor in [0, 255].

    The pipeline operates in 0-255 float space so the resize arithmetic is
    bit-identical to the official uint8-based cv2 pipeline; uint8 values are kept
    exactly and float tensors are assumed to be in [0, 1]. Other dtypes raise.
    Single-channel images are repeated to three channels and an alpha channel is
    dropped.
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
    # As in cv2, the kernel choice is joint over both axes: a resize that enlarges
    # either axis uses cubic, otherwise area.
    method = "cubic" if new_h > h or new_w > w else "area"
    return _resize(img, new_h=new_h, new_w=new_w, method=method, round_to_grid=True)


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
    # As in cv2, the kernel choice is joint over both axes: a resize that enlarges
    # either axis uses cubic, otherwise area.
    method = "cubic" if new_h > h or new_w > w else "area"
    return _resize(img, new_h=new_h, new_w=new_w, method=method, round_to_grid=True)


def _dav2_target_size(
    *, h: int, w: int, input_size: int, patch: int
) -> tuple[int, int]:
    """Computes the Depth Anything V2 (MiDaS) target size for a lower-bound resize.

    Scales so the shorter side reaches ``input_size`` (both sides scaled by the same
    factor), then constrains each side to a multiple of ``patch``: round to the nearest
    multiple, but ceil to a multiple if that would fall below ``input_size``. Mirrors
    the official ``Resize.get_size`` with ``resize_method="lower_bound"`` and
    ``ensure_multiple_of=patch``.

    Returns:
        The ``(new_h, new_w)`` target size, both multiples of ``patch``.
    """
    scale = max(input_size / h, input_size / w)

    def _constrain(x: float) -> int:
        y = int(round(x / patch) * patch)
        if y < input_size:
            y = int(math.ceil(x / patch) * patch)
        return y

    return _constrain(scale * h), _constrain(scale * w)


def _resize(
    img: Tensor, *, new_h: int, new_w: int, method: str, round_to_grid: bool
) -> Tensor:
    """Resizes a ``(C, H, W)`` float image to ``(new_h, new_w)``.

    Reproduces the cv2 resampling the official Depth Anything reference depends on
    without an OpenCV dependency: ``"area"`` matches ``INTER_AREA`` and ``"cubic"``
    matches ``INTER_CUBIC``. Both are separable, so per-axis ``(dst, src)`` weight
    matrices are built (inheriting ``img``'s dtype to match cv2's arithmetic, float64
    for the DAv2 pipeline and float32 for DAv3) and applied along the height then the
    width.

    Args:
        img: Float image tensor with shape ``(C, H, W)``.
        new_h: Target height.
        new_w: Target width.
        method: Interpolation method, either ``"area"`` or ``"cubic"``.
        round_to_grid: If ``True``, round the result back to the integer grid (in
            float, without a dtype change) and clamp to [0, 255], reproducing cv2's
            round-back-to-uint8 so DAv3 values match the official pipeline bit-exactly;
            skipping this quantization shifts the model input enough to visibly change
            the predicted depth. The DAv2 pipeline keeps the unrounded float result and
            passes ``False``.

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
    """Builds the ``(dst, src)`` cv2 ``INTER_AREA`` overlap-weight matrix for one axis.

    Output cell ``i`` spans ``[i * scale, (i + 1) * scale)`` in source coordinates,
    where ``scale = src / dst``; the weight for source pixel ``j`` is the length of
    the overlap with ``[j, j + 1)`` divided by ``scale``. A ``scale`` of 1 yields the
    identity, so an unchanged axis is passed through exactly.
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

    Uses the Keys cubic convolution kernel with ``a = -0.75`` (cv2's constant) and
    half-pixel sample centers. Output ``i`` samples source coordinate
    ``(i + 0.5) * scale - 0.5`` from its four nearest taps; taps falling outside the
    image clamp to the border (cv2's replicate behavior), accumulating their weight.
    The four kernel weights sum to one, so no renormalization is needed.

    Args:
        src: Source axis length.
        dst: Destination axis length.
        device: Device on which the weights are created.
        dtype: Floating-point dtype of the weights. The Depth Anything V2 pipeline uses
            float64 to match cv2's double-precision cubic on float images.
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
