#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from PIL import ImageChops
from torch import Tensor

from lightly_train._visualize import depth_estimation
from lightly_train.types import DepthEstimationBatch

# The magma colormap renders depth==0 (the colormap's far value) as a near-black pixel;
# this is what an invalid/sky pixel collapses to.
_FAR_PIXEL: tuple[int, int, int] = (0, 0, 3)


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


def test_plot_depth_labels__sky_rendered_as_far_value() -> None:
    # Sky pixels are excluded from the colorization and rendered as the colormap's far
    # value, while the valid non-sky region is colorized normally.
    depth = torch.ones(1, 1, 32, 32)
    depth[:, :, :16, :] = 50.0  # Garbage in the top-half sky region.
    # A vertical gradient in the valid (bottom-half) region so its pixels span the
    # colormap and are not all the far value.
    depth[:, :, 16:, :] = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 16, 1)
    sky = torch.zeros(1, 1, 32, 32)
    sky[:, :, :16, :] = 1.0

    image = depth_estimation.plot_depth_labels(
        batch=_make_batch(depth=depth, sky=sky),
        max_images=1,
        image_normalize=None,
    )

    # The label panel is the right half of the RGB|depth concatenation; probe the depth
    # half directly via the helper to avoid the concat offset.
    depth_panel = depth_estimation._depth_to_pil(depth=depth[0], invalid=sky[0] >= 0.5)
    # A pixel in the sky region collapses to the far value.
    assert depth_panel.getpixel((0, 0)) == _FAR_PIXEL
    # The farthest valid pixel (largest depth) is colorized to a non-far color.
    assert depth_panel.getpixel((0, 31)) != _FAR_PIXEL
    # Sanity: the label panel matches the right half of the rendered grid.
    assert image.size[0] == 2 * depth_panel.size[0]
