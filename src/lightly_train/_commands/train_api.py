#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Training workflows for the LightlyTrain API service.

These fit a head on top of a frozen pretrained model, in memory and in process:
no Fabric, no dataloaders, no augmentations, no checkpoint files. Every run is
bounded by a step count and a wall clock deadline.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from torch.nn import Linear
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from lightly_train._task_models import task_model_helpers
from lightly_train._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)
from lightly_train._task_models.ltdetr_object_detection.train_model import (
    LTDETRObjectDetectionTrainArgs,
)
from lightly_train._task_models.object_detection_components.dfine_criterion import (
    DFINECriterion,
)
from lightly_train._task_models.object_detection_components.matcher import (
    HungarianMatcher,
)

logger = logging.getLogger(__name__)

# State dict of the trained head only, not of the full model.
HeadWeights = Dict[str, Tensor]


@dataclass(frozen=True)
class TrainMetrics:
    train_loss: float
    # Only defined for classification.
    train_accuracy: float | None = None


@dataclass(frozen=True)
class FittedHead:
    weights: HeadWeights
    metrics: TrainMetrics


def normalize_features(features: Tensor) -> Tensor:
    """L2 normalization applied before both training and prediction."""
    return F.normalize(features, dim=-1)


def fit_classification_head(
    *,
    features: Tensor,
    labels: Tensor,
    num_classes: int,
    steps: int,
    lr: float,
    weight_decay: float,
    max_seconds: float,
) -> FittedHead:
    """Fits a randomly initialized linear head on all features. Full batch."""
    features = normalize_features(features)

    head = Linear(features.shape[-1], num_classes)
    head.weight.data.normal_(mean=0.0, std=0.01)
    head.bias.data.zero_()

    optimizer = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps)

    loss = torch.zeros(())
    for _ in _steps_within_deadline(steps=steps, max_seconds=max_seconds):
        optimizer.zero_grad()
        logits = head(features)
        loss = F.cross_entropy(logits, labels)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        logits = head(features)
        accuracy = (logits.argmax(dim=-1) == labels).float().mean()

    return FittedHead(
        weights={"weight": head.weight.detach(), "bias": head.bias.detach()},
        metrics=TrainMetrics(
            train_loss=float(loss.detach()), train_accuracy=float(accuracy)
        ),
    )


@lru_cache(maxsize=1)
def _load_checkpoint(model_name: str) -> dict[str, Any] | None:
    """Loads and caches the pretrained checkpoint. Downloaded at most once.

    Returns None for a model name that names an architecture without pretrained
    detection weights.
    """
    try:
        path = task_model_helpers.download_checkpoint(checkpoint=model_name)
    except ValueError:
        if not LTDETRObjectDetection.is_supported_model(model_name):
            raise
        logger.warning(
            f"No pretrained weights for '{model_name}'. The detector is randomly "
            "initialized and its predictions are meaningless."
        )
        return None
    checkpoint: dict[str, Any] = torch.load(
        path, weights_only=False, map_location="cpu"
    )
    return checkpoint


def load_detector(
    *,
    model_name: str,
    classes: Mapping[int, str],
    image_size: tuple[int, int],
    device: torch.device,
) -> LTDETRObjectDetection:
    """Returns a pretrained detector with a randomly initialized class head.

    The class heads are sized for ``classes``, which differs from the checkpoint's
    class count, so the decoder's load hooks keep the module's own initialization
    for them. Every other weight comes from the checkpoint.
    """
    checkpoint = _load_checkpoint(model_name)
    if checkpoint is None:
        model = LTDETRObjectDetection(
            model_name=model_name,
            classes=dict(classes),
            image_size=image_size,
            load_weights=False,
        )
        return model.to(device)

    init_args = dict(checkpoint["model_init_args"])
    init_args["classes"] = dict(classes)
    init_args["image_size"] = image_size
    init_args["load_weights"] = False

    model = LTDETRObjectDetection(**init_args)
    model.load_train_state_dict(state_dict=checkpoint["train_model"])
    return model.to(device)


def preprocess_detection_image(
    *, model: LTDETRObjectDetection, image: PILImage, device: torch.device
) -> Tensor:
    """Returns the resized (C, H, W) tensor the detector expects, unnormalized.

    Normalization is deferred to :meth:`ObjectDetectionPreprocessor.preprocess_batch`
    so the result can be cached and restacked cheaply.
    """
    tensor, _ = model.preprocessor.preprocess_image(
        image, device=device, dtype=torch.float32
    )
    return tensor


def boxes_to_target(
    *, boxes: Sequence[Sequence[float]], labels: Sequence[int], size: tuple[int, int]
) -> dict[str, Tensor]:
    """Converts absolute xyxy boxes to the normalized cxcywh targets of the criterion.

    Preprocessing resizes without preserving the aspect ratio, so normalizing by the
    original size is enough.
    """
    height, width = size
    if not boxes:
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
    xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    x1, y1, x2, y2 = (xyxy / scale).unbind(dim=-1)
    cxcywh = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)
    return {
        "boxes": cxcywh,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def fit_detection_head(
    *,
    model: LTDETRObjectDetection,
    images: Tensor,
    targets: Sequence[Mapping[str, Tensor]],
    steps: int,
    lr: float,
    weight_decay: float,
    max_seconds: float,
) -> FittedHead:
    """Fits the class head of a frozen detector on all samples. Full batch."""
    model.freeze_all()
    parameters = list(model.class_head_parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)

    criterion = _build_criterion(model=model)
    device = images.device
    batch = model.preprocessor.preprocess_batch(images)
    batch_targets = [
        {key: value.to(device) for key, value in target.items()} for target in targets
    ]

    optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps)

    loss = torch.zeros((), device=device)
    with _detection_train_mode(model):
        for _ in _steps_within_deadline(steps=steps, max_seconds=max_seconds):
            optimizer.zero_grad()
            outputs = model._forward_train(x=batch, targets=batch_targets)
            loss_dict = criterion(outputs=outputs, targets=batch_targets, world_size=1)
            loss = torch.stack(list(loss_dict.values())).sum()
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            scheduler.step()

    weights = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return FittedHead(
        weights=weights, metrics=TrainMetrics(train_loss=float(loss.detach()))
    )


def _build_criterion(*, model: LTDETRObjectDetection) -> DFINECriterion:
    # decoder_name defaults to "auto"; the effective loss fields only resolve once it
    # names a concrete decoder.
    args = LTDETRObjectDetectionTrainArgs(decoder_name="dfine")
    matcher = HungarianMatcher(  # type: ignore[no-untyped-call]
        weight_dict=args.matcher_weight_dict,
        use_focal_loss=args.matcher_use_focal_loss,
        alpha=args.matcher_alpha,
        gamma=args.matcher_gamma,
    )
    criterion: DFINECriterion = DFINECriterion(  # type: ignore[no-untyped-call]
        matcher=matcher,
        weight_dict=args.effective_loss_weight_dict,
        losses=args.effective_losses,
        alpha=args.loss_alpha,
        gamma=args.loss_gamma,
        num_classes=len(model.classes),
        reg_max=model.decoder.reg_max,
    )
    return criterion.to(next(model.parameters()).device)


def _steps_within_deadline(*, steps: int, max_seconds: float) -> Iterator[int]:
    """Yields step indices until the step count or the wall clock budget is spent."""
    deadline = time.monotonic() + max_seconds
    for step in range(steps):
        yield step
        if time.monotonic() > deadline:
            return


@contextmanager
def _detection_train_mode(model: LTDETRObjectDetection) -> Iterator[None]:
    """Enables training-mode behavior without disturbing the frozen weights.

    The decoder only emits the auxiliary and denoising outputs the criterion needs
    while it is in training mode. The backbone and the encoder are kept in eval mode
    so their batch norm running statistics stay at the pretrained values.
    """
    was_training = model.training
    model.train()
    model.backbone.eval()
    model.encoder.eval()
    try:
        yield
    finally:
        model.train(was_training)
