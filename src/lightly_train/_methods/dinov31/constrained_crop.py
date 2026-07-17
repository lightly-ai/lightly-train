#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Geometry-coupled crop helpers for the PaKA dense relational loss.

- [0]: 2025, PaKA: https://arxiv.org/abs/2509.05606

These are data-augmentation utilities (not SSL math): they build extra views
whose crop geometry depends on another view's crop, which the standard
stateless ``ViewTransform`` cannot express.

- :func:`render_clean_global` renders an augmentation-free ("clean") copy of an
  already-sampled global crop, re-using its recorded geometry. The clean global
  feeds the PaKA cross-view loss as an un-augmented teacher (paper Fig 2a /
  Sec. 4 "Augmentation-free Teacher") while DINO/iBOT keep the augmented globals.
- :func:`sample_high_overlap_box` samples a small student local crop fully inside
  a parent global crop (paper Sec. 4 "Global-Local Intersection Maximization",
  overlap ``m >= 0.9``). Kept here so it is unit-testable in isolation.
- :func:`render_augmented_local` renders such a local crop with the standard
  photometric augmentations.
"""

from __future__ import annotations

import math
import random

import cv2
import torch
from albumentations import (
    Compose,
    Crop,
    HorizontalFlip,
    Resize,
    VerticalFlip,
)
from albumentations.pytorch.transforms import ToTensorV2

from lightly_train._transforms.normalize import NormalizeDtypeAware as Normalize
from lightly_train._transforms.transform import (
    ColorJitterArgs,
    GaussianBlurArgs,
    NormalizeArgs,
    SolarizeArgs,
)
from lightly_train._transforms.view_transform import build_photometric_ops
from lightly_train.types import (
    ImageSizeTuple,
    TransformInput,
    TransformOutputSingleView,
)

# DINO-style local-crop scale: the PaKA student local is a small sub-region of
# the teacher global crop (mirroring DINO's local-crop scale range).
_LOCAL_AREA_MIN = 0.05
_LOCAL_AREA_MAX = 0.40


def render_clean_global(
    input: TransformInput,
    geometry: torch.Tensor,
    size: ImageSizeTuple,
    normalize: NormalizeArgs,
) -> TransformOutputSingleView:
    """Renders a global crop with NO photometric augmentation (clean teacher).

    Crops ``input`` to the box stored in ``geometry`` (``x0,y0,x1,y1`` in
    original-image pixels), resizes to ``size``, re-applies the recorded h/v
    flips, and normalizes -- crop + flip + resize + normalize only, no color
    jitter / blur / solarize / grayscale. The returned view re-uses ``geometry``
    verbatim so its recorded crop/flip matches the source (augmented) global crop
    exactly; the PaKA ROI alignment can then treat the two as the same region.

    Args:
        input: The transform input dict (``{"image": np.ndarray}``).
        geometry: The 8-element geometry tensor of the source global crop
            ``[x0, y0, x1, y1, image_w, image_h, hflip, vflip]``.
        size: Output ``(height, width)`` for the rendered crop.
        normalize: Normalization mean/std.

    Returns:
        The rendered view dict with the same ``"geometry"`` as the source.
    """
    x0 = int(round(float(geometry[0])))
    y0 = int(round(float(geometry[1])))
    x1 = int(round(float(geometry[2])))
    y1 = int(round(float(geometry[3])))
    hflip = float(geometry[6]) > 0.5
    vflip = float(geometry[7]) > 0.5

    transforms = [
        Crop(x_min=x0, y_min=y0, x_max=x1, y_max=y1),
        Resize(height=size[0], width=size[1], interpolation=cv2.INTER_AREA),
    ]
    if hflip:
        transforms.append(HorizontalFlip(p=1.0))
    if vflip:
        transforms.append(VerticalFlip(p=1.0))
    transforms.append(Normalize(mean=normalize.mean, std=normalize.std))
    transforms.append(ToTensorV2())

    out: TransformOutputSingleView = Compose(transforms)(**input)
    out["geometry"] = geometry.clone()
    return out


def sample_high_overlap_box(
    parent_box: tuple[int, int, int, int],
    min_iou: float,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    """Samples a small local box fully inside ``parent_box``.

    The PaKA paper (Sec. 4) rejects student local crops whose overlap with the
    teacher global crop is below a ratio ``m`` (=0.9). A literal box IoU between
    a small local and a large global is geometrically tiny, so the paper's
    "IoU" is in fact the CONTAINMENT ratio -- the fraction of the local that
    lies inside the global. Sampling the local fully inside the parent therefore
    satisfies the ``>= min_iou`` containment constraint by construction
    (containment == 1.0).

    The local area is drawn in ``[_LOCAL_AREA_MIN, _LOCAL_AREA_MAX]`` of the
    parent area, with a random aspect ratio and a random position inside the
    parent. All coordinates are original-image pixels.

    Args:
        parent_box: ``(x0, y0, x1, y1)`` of the parent (global) crop.
        min_iou: Minimum required CONTAINMENT of the local within the parent
            (fraction of the local inside the parent), in (0, 1]. Always
            satisfied here because the local is sampled fully inside the parent.
        rng: A seeded ``random.Random`` (pass a per-call instance for
            worker-safety).

    Returns:
        ``(x0, y0, x1, y1)`` of the sampled local box, fully inside the parent.
    """
    px0, py0, px1, py1 = parent_box
    pw = px1 - px0
    ph = py1 - py0
    if pw <= 1 or ph <= 1:
        return parent_box

    parent_area = float(pw * ph)
    target_area = rng.uniform(_LOCAL_AREA_MIN, _LOCAL_AREA_MAX) * parent_area
    aspect = rng.uniform(3.0 / 4.0, 4.0 / 3.0)
    w = round(math.sqrt(target_area * aspect))
    h = round(math.sqrt(target_area / aspect))
    # Clip to the parent (only ever shrinks, so still <= _LOCAL_AREA_MAX).
    w = min(max(1, w), pw)
    h = min(max(1, h), ph)
    x0 = px0 + rng.randint(0, pw - w)
    y0 = py0 + rng.randint(0, ph - h)
    return x0, y0, x0 + w, y0 + h


def render_augmented_local(
    input: TransformInput,
    box: tuple[int, int, int, int],
    size: ImageSizeTuple,
    hflip: bool,
    vflip: bool,
    color_jitter: ColorJitterArgs | None,
    random_gray_scale: float | None,
    gaussian_blur: GaussianBlurArgs | None,
    solarize: SolarizeArgs | None,
    normalize: NormalizeArgs,
) -> TransformOutputSingleView:
    """Renders an augmented student local crop from a pre-sampled box.

    Crops ``input`` to ``box`` (``x0,y0,x1,y1`` original-image pixels), resizes
    to ``size``, applies the given (Python-decided) flips, then the standard
    DINO photometric augmentations, then normalizes. Used for the PaKA
    high-overlap student local crops: the crop box is sampled deterministically
    (so we know it for geometry + the containment guarantee) while the teacher
    stays clean. The recorded geometry uses ``box`` and the applied flips so the
    PaKA ROI alignment is exact.
    """
    x0, y0, x1, y1 = box
    image_h, image_w = input["image"].shape[:2]
    transforms = [
        Crop(x_min=x0, y_min=y0, x_max=x1, y_max=y1),
        Resize(height=size[0], width=size[1], interpolation=cv2.INTER_AREA),
    ]
    if hflip:
        transforms.append(HorizontalFlip(p=1.0))
    if vflip:
        transforms.append(VerticalFlip(p=1.0))
    transforms += build_photometric_ops(
        color_jitter=color_jitter,
        random_gray_scale=random_gray_scale,
        gaussian_blur=gaussian_blur,
        solarize=solarize,
    )
    transforms += [
        Normalize(mean=normalize.mean, std=normalize.std),
        ToTensorV2(),
    ]
    out: TransformOutputSingleView = Compose(transforms)(**input)
    out["geometry"] = torch.tensor(
        [
            float(x0),
            float(y0),
            float(x1),
            float(y1),
            float(image_w),
            float(image_h),
            1.0 if hflip else 0.0,
            1.0 if vflip else 0.0,
        ],
        dtype=torch.float32,
    )
    return out
