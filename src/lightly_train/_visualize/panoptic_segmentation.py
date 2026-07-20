#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms import functional as torchvision_functional

from lightly_train._visualize import utils
from lightly_train.types import MaskPanopticSegmentationBatch


@dataclass
class PanopticSegmentationTaskStepVisualization:
    batch: MaskPanopticSegmentationBatch
    class_names: dict[int, str]
    image_normalize: dict[str, tuple[float, ...]] | None
    max_images: int
    alpha: float
    pred_masks: Sequence[Tensor] | None = None

    def create_label_image(self) -> PILImage | None:
        return plot_panoptic_segmentation_labels(
            batch=self.batch,
            class_names=self.class_names,
            image_normalize=self.image_normalize,
            max_images=self.max_images,
            alpha=self.alpha,
        )

    def create_prediction_image(self) -> PILImage | None:
        if self.pred_masks is None:
            return None
        return plot_panoptic_segmentation_predictions(
            batch=self.batch,
            pred_masks=self.pred_masks,
            class_names=self.class_names,
            image_normalize=self.image_normalize,
            max_images=self.max_images,
            alpha=self.alpha,
        )


def plot_panoptic_segmentation_labels(
    batch: MaskPanopticSegmentationBatch,
    class_names: dict[int, str],
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
    alpha: float,
) -> PILImage:
    """Render a grid of images annotated with ground truth panoptic segmentation masks.

    The class label channel of each mask is colorized using ``class_names``,
    and contours are drawn along segment boundaries so that distinct instances
    of the same class remain visually separated.

    Args:
        batch: Panoptic segmentation batch with images and (H, W, 2) masks where
            channel 0 holds internal class labels and channel 1 holds segment
            ids (segment id -1 indicates unassigned pixels).
        class_names: A dict mapping internal class IDs to class names. May
            also contain the internal ignore class id mapped to "ignored" when
            masks include ignored pixels.
        max_images: Maximum number of images to include in the grid.
        image_normalize: Optional dict with "mean" and "std" tuples used to
            denormalize images before rendering. If None, images pass through
            unchanged.
        alpha: Blending factor for the mask overlay in [0, 1]. 0 shows only the
            image, 1 shows only the mask.

    Returns:
        A single PIL image containing up to max_images annotated images arranged
        in a grid.
    """
    images = batch["image"]
    masks = batch["masks"]
    gt_images = (
        [img.float().cpu() for img in images]
        if isinstance(images, list)
        else images.float().cpu()
    )
    gt_masks = [m.cpu() for m in masks] if isinstance(masks, list) else masks.cpu()
    n = min(max_images, len(gt_images))

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = gt_images[i].clone()
        if image_normalize is not None:
            image_tensor = utils._denormalize_image(
                image=image_tensor,
                mean=image_normalize["mean"],
                std=image_normalize["std"],
            )
        img = torchvision_functional.to_pil_image(image_tensor).convert("RGB")
        # (H, W, 2): channel 0 is the class label, channel 1 is the segment id.
        label_mask = gt_masks[i][..., 0]
        segment_mask = gt_masks[i][..., 1]

        overlay = utils._build_panoptic_mask_overlay(
            label_mask=label_mask,
            segment_mask=segment_mask,
            size=img.size,
            class_names=class_names,
        )
        blended = Image.blend(img, overlay, alpha=alpha)
        # Use segment id boundaries so different instances of the same class
        # remain visually separated.
        blended = utils._draw_mask_contours(image=blended, mask=segment_mask)

        labels, colors = utils._legend_entries_for_mask(
            mask=label_mask, class_names=class_names
        )
        blended = utils._draw_class_legend(image=blended, labels=labels, colors=colors)

        pil_images.append(blended)

    return utils._render_grid(pil_images)


def plot_panoptic_segmentation_predictions(
    batch: MaskPanopticSegmentationBatch,
    pred_masks: Sequence[Tensor],
    class_names: dict[int, str],
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
    alpha: float,
) -> PILImage:
    """Render a grid of images annotated with predicted panoptic segmentation masks.

    Args:
        batch: Panoptic segmentation batch with images.
        pred_masks: Per-image predicted (H, W, 2) masks where channel 0 holds
            internal class labels and channel 1 holds segment ids. Each mask
            must have the same spatial size as its corresponding image in
            ``batch["image"]``.
        class_names: A dict mapping internal class IDs to class names.
        max_images: Maximum number of images to include in the grid.
        image_normalize: Optional dict with "mean" and "std" tuples used to
            denormalize images before rendering. If None, images pass through
            unchanged.
        alpha: Blending factor for the mask overlay in [0, 1]. 0 shows only the
            image, 1 shows only the mask.

    Returns:
        A single PIL image containing up to max_images annotated images arranged
        in a grid.
    """
    images = batch["image"]
    gt_images = (
        [img.float().cpu() for img in images]
        if isinstance(images, list)
        else images.float().cpu()
    )
    n = min(max_images, len(gt_images), len(pred_masks))

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = gt_images[i].clone()
        if image_normalize is not None:
            image_tensor = utils._denormalize_image(
                image=image_tensor,
                mean=image_normalize["mean"],
                std=image_normalize["std"],
            )
        img = torchvision_functional.to_pil_image(image_tensor).convert("RGB")
        pred_mask = pred_masks[i].cpu()
        # (H, W, 2): channel 0 is the class label, channel 1 is the segment id.
        label_mask = pred_mask[..., 0]
        segment_mask = pred_mask[..., 1]

        overlay = utils._build_panoptic_mask_overlay(
            label_mask=label_mask,
            segment_mask=segment_mask,
            size=img.size,
            class_names=class_names,
        )
        blended = Image.blend(img, overlay, alpha=alpha)
        # Use segment id boundaries so different instances of the same class
        # remain visually separated.
        blended = utils._draw_mask_contours(image=blended, mask=segment_mask)

        labels, colors = utils._legend_entries_for_mask(
            mask=label_mask, class_names=class_names
        )
        blended = utils._draw_class_legend(image=blended, labels=labels, colors=colors)

        pil_images.append(blended)

    return utils._render_grid(pil_images)
