#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Literal

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.transforms import functional as torchvision_functional

from lightly_train._visualize.utils import (
    _denormalize_image,
    _draw_class_legend,
    _render_grid,
)
from lightly_train.types import ImageClassificationBatch


def plot_image_classification_labels(
    batch: ImageClassificationBatch,
    class_names: dict[int, str],
    max_images: int,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> PILImage:
    """Render a grid of images annotated with ground truth class labels.

    Args:
        batch: Image classification batch with images and class IDs.
        class_names: Mapping from class ID to class name.
        max_images: Maximum number of images to include in the grid.
        mean: Per-channel mean used for image normalization (for denormalization).
        std: Per-channel std used for image normalization (for denormalization).

    Returns:
        A single PIL image containing up to max_images annotated images arranged
        in a grid.
    """
    images = batch["image"].cpu()
    gt_classes = [c.cpu() for c in batch["classes"]]
    n = min(max_images, images.shape[0])

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = images[i].clone().to(dtype=torch.float32)
        if mean is not None and std is not None:
            image_tensor = _denormalize_image(image=image_tensor, mean=mean, std=std)

        img = torchvision_functional.to_pil_image(image_tensor)

        labels = [
            class_names.get(int(cid), f"Class {int(cid)}") for cid in gt_classes[i]
        ]
        img = _draw_class_legend(image=img, labels=labels)

        pil_images.append(img)

    return _render_grid(pil_images)


def plot_image_classification_predictions(
    batch: ImageClassificationBatch,
    logits: Tensor,
    class_names: dict[int, str],
    max_images: int,
    top_k: int,
    classification_task: Literal["multiclass", "multilabel"] = "multiclass",
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> PILImage:
    """Render a grid of images annotated with top-k predicted class labels and scores.

    Labels are drawn vertically in the top-left corner, sorted from highest to lowest
    confidence.

    Args:
        batch: Image classification batch with images.
        logits: Model output logits of shape (batch_size, num_classes).
        class_names: Mapping from class ID to class name.
        max_images: Maximum number of images to include in the grid.
        top_k: Number of top predictions to display per image.
        classification_task: Whether the task is "multiclass" (softmax scores) or
            "multilabel" (sigmoid scores).
        mean: Per-channel mean used for image normalization (for denormalization).
        std: Per-channel std used for image normalization (for denormalization).

    Returns:
        A single PIL image containing up to max_images annotated images arranged
        in a grid.
    """
    images = batch["image"].cpu()
    gt_classes = [c.cpu() for c in batch["classes"]]
    logits = logits.detach().to(device="cpu", dtype=torch.float32)
    n = min(max_images, images.shape[0])

    if classification_task == "multilabel":
        probs = torch.sigmoid(logits[:n])
    else:
        probs = torch.softmax(logits[:n], dim=-1)
    num_classes = probs.shape[1]
    max_gt_labels = max((len(gt_classes[i]) for i in range(n)), default=0)
    topk_k = min(num_classes, max(top_k, max_gt_labels))
    top_scores, top_class_ids = torch.topk(probs, k=topk_k, dim=-1)

    pil_images: list[PILImage] = []
    for i in range(n):
        image_tensor = images[i].clone().to(dtype=torch.float32)
        if mean is not None and std is not None:
            image_tensor = _denormalize_image(image=image_tensor, mean=mean, std=std)

        img = torchvision_functional.to_pil_image(image_tensor)

        # Ensure we show at least as many predictions as there are ground truth
        # labels (in case of multi-label classification), but never exceed num_classes.
        effective_k = min(num_classes, max(top_k, len(gt_classes[i])))

        labels: list[str] = []
        for rank in range(effective_k):
            class_id = int(top_class_ids[i, rank].item())
            score = float(top_scores[i, rank].item())
            class_name = class_names.get(class_id, f"Class {class_id}")
            labels.append(f"{class_name}: {score:.2f}")
        img = _draw_class_legend(image=img, labels=labels)

        pil_images.append(img)

    return _render_grid(pil_images)
