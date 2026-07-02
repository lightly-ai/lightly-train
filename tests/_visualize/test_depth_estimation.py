#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import matplotlib
import torch
from PIL import ImageChops
from torch import Tensor

from lightly_train._visualize import depth_estimation
from lightly_train.types import DepthEstimationBatch


def _far_pixel() -> tuple[int, int, int]:
    # Non-positive depth is rendered with the low end of the active colormap.
    rgb = matplotlib.colormaps[depth_estimation._DEPTH_COLORMAP](0.0)[:3]
    return tuple(int(channel * 255) for channel in rgb)


def _make_batch(*, depth: Tensor, sky: Tensor) -> DepthEstimationBatch:
    batch_size, _, height, width = depth.shape
    return DepthEstimationBatch(
        image_path=[f"img_{i}.png" for i in range(batch_size)],
        image=torch.zeros(batch_size, 3, height, width),
        depth=depth,
        sky=sky,
    )


def test_plot_depth_labels__sky_does_not_affect_non_sky_colors() -> None:
    # The depth in sky regions is garbage. Two label images that differ only in the
    # sky-region depth must render identically in the non-sky region: if sky leaked into
    # the min-max normalization, its extreme value would recolor every non-sky pixel.
    depth = torch.ones(1, 1, 32, 32)
    depth[:, :, :16, :] = 50.0  # Garbage in the top-half sky region.
    sky = torch.zeros(1, 1, 32, 32)
    sky[:, :, :16, :] = 1.0

    depth_other = depth.clone()
    depth_other[:, :, :16, :] = 999.0  # Different garbage in the same sky region.

    image = depth_estimation.plot_depth_labels(
        batch=_make_batch(depth=depth, sky=sky),
        max_images=1,
        image_normalize=None,
    )
    image_other = depth_estimation.plot_depth_labels(
        batch=_make_batch(depth=depth_other, sky=sky),
        max_images=1,
        image_normalize=None,
    )

    assert ImageChops.difference(image, image_other).getbbox() is None


def test_plot_depth_labels__sky_filled_as_distant_not_black() -> None:
    # Sky pixels are excluded from the normalization and then filled with the 99th
    # percentile of the non-sky depth, so they render as distant scenery (not as the
    # black far value) and take the same color as the farthest valid pixel.
    depth = torch.ones(1, 1, 32, 32)
    depth[:, :, :16, :] = 50.0  # Garbage in the top-half sky region.
    # A vertical gradient in the valid (bottom-half) region so its pixels span the
    # colormap; the largest valid depth is the 99th-percentile fill value used for sky.
    depth[:, :, 16:, :] = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 16, 1)
    sky = torch.zeros(1, 1, 32, 32)
    sky[:, :, :16, :] = 1.0

    depth_panel = depth_estimation._depth_to_pil(depth=depth[0], sky=sky[0] >= 0.5)

    # A sky pixel is not black; it matches the color of the farthest valid pixel
    # (bottom row, the largest non-sky depth ~= the 99th percentile).
    farthest_valid = depth_panel.getpixel((0, 31))
    assert depth_panel.getpixel((0, 0)) != _far_pixel()
    assert depth_panel.getpixel((0, 0)) == farthest_valid


def test_plot_depth_labels__nonpositive_depth_rendered_as_far_value() -> None:
    # Genuinely invalid (non-positive) depth pixels still collapse to the far value.
    depth = torch.full((1, 1, 32, 32), 5.0)
    depth[:, :, :16, :] = 0.0  # Invalid top half.
    sky = torch.zeros(1, 1, 32, 32)

    depth_panel = depth_estimation._depth_to_pil(depth=depth[0], sky=sky[0] >= 0.5)

    assert depth_panel.getpixel((0, 0)) == _far_pixel()
