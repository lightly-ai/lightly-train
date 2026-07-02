#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import torch
from PIL import Image
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms import functional as torchvision_functional

from lightly_train._visualize import utils
from lightly_train.types import DepthEstimationBatch

# Colormap used to render depth maps during training. DAv3 predicts larger values for
# farther pixels, so nearby pixels sit at the low end of the colormap. A multi-hue map
# makes small near-depth changes easier to spot than the old dark-to-bright ``magma``.
_DEPTH_COLORMAP = "Spectral_r"


@dataclass
class DepthEstimationTaskStepVisualization:
    batch: DepthEstimationBatch
    image_normalize: dict[str, tuple[float, ...]] | None
    max_images: int
    pred_depth: Tensor | None = None

    def create_label_image(self) -> PILImage | None:
        return plot_depth_labels(
            batch=self.batch,
            image_normalize=self.image_normalize,
            max_images=self.max_images,
        )

    def create_prediction_image(self) -> PILImage | None:
        if self.pred_depth is None:
            return None
        return plot_depth_predictions(
            batch=self.batch,
            pred_depth=self.pred_depth,
            image_normalize=self.image_normalize,
            max_images=self.max_images,
        )


def plot_depth_labels(
    batch: DepthEstimationBatch,
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
) -> PILImage:
    """Render a grid pairing each input image with its ground truth depth map.

    Args:
        batch: Depth estimation batch with images and depth maps.
        max_images: Maximum number of images to include in the grid.
        image_normalize: Optional dict with "mean" and "std" tuples used to
            denormalize images before rendering. If None, images pass through
            unchanged.

    Returns:
        A single PIL image with up to max_images image/depth pairs arranged in a grid.
    """
    images = _as_tensor(batch["image"])
    depth = _as_tensor(batch["depth"])
    # The ground-truth depth in sky regions is garbage (the teacher has no valid depth
    # there), so exclude sky pixels from the colorization and fill them as distant.
    sky = _as_tensor(batch["sky"])
    n = min(max_images, images.shape[0])

    pil_images: list[PILImage] = []
    for i in range(n):
        rgb = _image_to_pil(image=images[i], image_normalize=image_normalize)
        depth_img = _depth_to_pil(depth=depth[i], sky=sky[i] >= 0.5)
        pil_images.append(_concat_horizontal(left=rgb, right=depth_img))

    return utils._render_grid(pil_images)


def plot_depth_predictions(
    batch: DepthEstimationBatch,
    pred_depth: Tensor,
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
) -> PILImage:
    """Render a grid pairing each input image with the predicted depth map.

    Args:
        batch: Depth estimation batch with images.
        pred_depth: Predicted depth of shape (batch_size, 1, H, W).
        max_images: Maximum number of images to include in the grid.
        image_normalize: Optional dict with "mean" and "std" tuples used to
            denormalize images before rendering. If None, images pass through
            unchanged.

    Returns:
        A single PIL image with up to max_images image/prediction pairs in a grid.
    """
    images = _as_tensor(batch["image"])
    pred = pred_depth.detach().to(device="cpu", dtype=torch.float32)
    n = min(max_images, images.shape[0])

    pil_images: list[PILImage] = []
    for i in range(n):
        rgb = _image_to_pil(image=images[i], image_normalize=image_normalize)
        depth_img = _depth_to_pil(depth=pred[i])
        pil_images.append(_concat_horizontal(left=rgb, right=depth_img))

    return utils._render_grid(pil_images)


def _as_tensor(value: Tensor | list[Tensor]) -> Tensor:
    """Returns a stacked, CPU float tensor for both batched and per-sample inputs."""
    if isinstance(value, list):
        value = torch.stack(value)
    return value.detach().to(device="cpu", dtype=torch.float32)


def _image_to_pil(
    image: Tensor, image_normalize: dict[str, tuple[float, ...]] | None
) -> PILImage:
    image = image.clone().to(dtype=torch.float32)
    if image_normalize is not None:
        image = utils._denormalize_image(
            image=image,
            mean=image_normalize["mean"],
            std=image_normalize["std"],
        )
    pil_image: PILImage = torchvision_functional.to_pil_image(image)
    return pil_image


def _depth_to_pil(depth: Tensor, sky: Tensor | None = None) -> PILImage:
    """Colorizes a (1, H, W) depth map with a colormap, ignoring invalid pixels.

    Depth is min-max normalized over the valid (positive) pixels of the sample so the
    full colormap range is used regardless of the absolute depth scale.

    Args:
        depth: Depth map of shape ``(1, H, W)``.
        sky: Optional boolean sky mask of shape ``(1, H, W)``. The ground-truth depth in
            sky regions is garbage, so sky pixels are excluded from the normalization
            and then filled with the 99th percentile of the non-sky depth, rendering
            them as distant scenery (matching the ``predict`` postprocessing) instead of
            as the colormap's far value.
    """
    depth = depth.squeeze(0)
    valid = depth > 0
    if sky is not None:
        valid = valid & ~sky.squeeze(0).bool()
    if bool(valid.any()):
        finite = depth[valid]
        d_min = finite.min()
        d_max = finite.max()
        depth = torch.where(valid, depth, torch.quantile(finite, 0.99))
        normalized = (depth - d_min) / (d_max - d_min + 1e-6)
    else:
        normalized = torch.zeros_like(depth)
    normalized = torch.clamp(normalized, 0.0, 1.0)
    # Non-positive (genuinely invalid) pixels are rendered as the colormap's far value.
    normalized = torch.where(depth > 0, normalized, torch.zeros_like(normalized))

    colormap = matplotlib.colormaps[_DEPTH_COLORMAP]
    colored = colormap(normalized.numpy())[..., :3]  # Drop alpha, keep RGB.
    colored_uint8 = (colored * 255).astype("uint8")
    return Image.fromarray(colored_uint8)


def _concat_horizontal(left: PILImage, right: PILImage) -> PILImage:
    """Pastes two equally sized images side by side into one."""
    height = max(left.size[1], right.size[1])
    width = left.size[0] + right.size[0]
    canvas = Image.new("RGB", (width, height))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.size[0], 0))
    return canvas
