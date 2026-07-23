#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms import functional as torchvision_functional

from lightly_train._visualize import utils
from lightly_train.types import ObjectDetectionBatch

logger = logging.getLogger(__name__)


@dataclass
class ObjectDetectionTaskStepVisualization:
    batch: ObjectDetectionBatch
    class_names: dict[int, str]
    image_normalize: dict[str, tuple[float, ...]] | None
    max_images: int
    score_threshold: float
    results: list[dict[str, Tensor]] | None = None

    def create_label_image(self) -> PILImage | None:
        return plot_object_detection_labels(
            batch=self.batch,
            class_names=self.class_names,
            image_normalize=self.image_normalize,
            max_images=self.max_images,
        )

    def create_prediction_image(self) -> PILImage | None:
        if self.results is None:
            return None
        return plot_object_detection_predictions(
            batch=self.batch,
            results=self.results,
            class_names=self.class_names,
            image_normalize=self.image_normalize,
            score_threshold=self.score_threshold,
            max_images=self.max_images,
        )


def plot_object_detection_labels(
    batch: ObjectDetectionBatch,
    class_names: dict[int, str],
    max_images: int,
    image_normalize: dict[str, tuple[float, ...]] | None,
) -> PILImage:
    """Render a grid of images annotated with ground truth bounding boxes.

    Args:
        batch: Object detection batch with images, bboxes (cxcywh normalized), and
            classes.
        class_names: Mapping from class ID to class name.
        image_normalize: Optional dict with "mean" and "std" tuples used to
            denormalize images before rendering. If None, images pass through
            unchanged.
        max_images: Maximum number of images to include in the grid.

    Returns:
        A single PIL image containing up to max_images annotated images arranged
        in a grid.
    """
    gt_images = batch["image"].cpu()
    gt_bboxes = [b.cpu() for b in batch["bboxes"]]
    gt_classes = [c.cpu() for c in batch["classes"]]
    n = min(max_images, gt_images.shape[0])

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = gt_images[i].clone().to(dtype=torch.float32)
        if image_normalize is not None:
            image_tensor = utils._denormalize_image(
                image=image_tensor,
                mean=image_normalize["mean"],
                std=image_normalize["std"],
            )

        _, img_height, img_width = image_tensor.shape
        img = torchvision_functional.to_pil_image(image_tensor)

        boxes = gt_bboxes[i]
        class_ids = gt_classes[i]
        boxes_xyxy = utils._cxcywh_to_xyxy(boxes=boxes, w=img_width, h=img_height)
        utils._draw_labeled_boxes(
            image=img,
            bboxes_xyxy=boxes_xyxy,
            labels=class_ids,
            scores=None,
            class_names=class_names,
        )
        pil_images.append(img)

    return utils._render_grid(pil_images)


def plot_object_detection_predictions(
    batch: ObjectDetectionBatch,
    results: list[dict[str, Tensor]],
    class_names: dict[int, str],
    max_images: int,
    score_threshold: float,
    image_normalize: dict[str, tuple[float, ...]] | None,
) -> PILImage:
    """Render a grid of images annotated with predicted bounding boxes.

    Predictions are filtered by score_threshold selecting the highest-confidence detections.

    Args:
        batch: Object detection batch with images and original_size.
        results: Postprocessor outputs, each a dict with 'boxes' (xyxy in original
            image coordinates), 'labels', and 'scores'.
        class_names: Mapping from class ID to class name.
        score_threshold: Minimum score for a predicted box to be shown.
        image_normalize: Optional dict with "mean" and "std" tuples used to
            denormalize images before rendering. If None, images pass through
            unchanged.

    Returns:
        A single PIL image containing up to max_images annotated images arranged
        in a grid.
    """
    results = [
        {k: v.cpu() if isinstance(v, Tensor) else v for k, v in r.items()}
        for r in results
    ]
    gt_images = batch["image"].cpu()
    orig_target_sizes = batch["original_size"]
    n = min(max_images, gt_images.shape[0])

    _, img_height, img_width = gt_images.shape[1:]

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = gt_images[i].clone().to(dtype=torch.float32)
        if image_normalize is not None:
            image_tensor = utils._denormalize_image(
                image=image_tensor,
                mean=image_normalize["mean"],
                std=image_normalize["std"],
            )

        img = torchvision_functional.to_pil_image(image_tensor)

        result = results[i]
        boxes = result["boxes"]
        class_ids = result["labels"]
        scores = result["scores"]

        if len(boxes) > 0:
            # Scale predicted boxes from original image coordinates to tensor dimensions.
            orig_width, orig_height = orig_target_sizes[i]
            boxes = boxes.clone()
            boxes[:, 0] = boxes[:, 0] * img_width / orig_width
            boxes[:, 1] = boxes[:, 1] * img_height / orig_height
            boxes[:, 2] = boxes[:, 2] * img_width / orig_width
            boxes[:, 3] = boxes[:, 3] * img_height / orig_height

            valid = utils._valid_box_mask(boxes)
            if not bool(valid.all()):
                logger.warning(
                    "Skipped %d degenerate predicted box(es) (x1>x2 or y1>y2) in "
                    "val-step visualization. This usually indicates unstable box "
                    "regression.",
                    int((~valid).sum()),
                )
                boxes = boxes[valid]
                class_ids = class_ids[valid]
                scores = scores[valid]

            mask = scores >= score_threshold
            boxes = boxes[mask]
            class_ids = class_ids[mask]
            scores = scores[mask]

            if len(scores) > 0:
                utils._draw_labeled_boxes(
                    image=img,
                    bboxes_xyxy=boxes,
                    labels=class_ids,
                    scores=scores,
                    class_names=class_names,
                )
        pil_images.append(img)

    return utils._render_grid(pil_images)
