#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import time

import pytest
import torch
from torch import Tensor

from lightly_train._commands import train_api
from lightly_train._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)

IMAGE_SIZE = (224, 224)


def _separable(
    num_classes: int, per_class: int = 8, dim: int = 16
) -> tuple[Tensor, Tensor]:
    centers = torch.eye(num_classes, dim) * 10.0
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    features = centers[labels] + 0.01 * torch.randn(len(labels), dim)
    return features, labels


def _detector(num_classes: int = 2) -> LTDETRObjectDetection:
    classes = {index: f"class_{index}" for index in range(num_classes)}
    return LTDETRObjectDetection(
        model_name="dinov2/_vittest14-ltdetrv2",
        classes=classes,
        image_size=IMAGE_SIZE,
        load_weights=False,
    )


def _targets(num_classes: int = 2) -> list[dict[str, Tensor]]:
    return [
        train_api.boxes_to_target(
            boxes=[[10.0, 10.0, 50.0, 60.0]], labels=[0], size=IMAGE_SIZE
        ),
        train_api.boxes_to_target(
            boxes=[[5.0, 5.0, 30.0, 30.0], [60.0, 60.0, 100.0, 110.0]],
            labels=[num_classes - 1, 0],
            size=IMAGE_SIZE,
        ),
    ]


def test_fit_classification_head() -> None:
    features, labels = _separable(num_classes=3)

    fitted = train_api.fit_classification_head(
        features=features,
        labels=labels,
        num_classes=3,
        steps=1000,
        lr=1e-1,
        weight_decay=0.0,
        max_seconds=5.0,
    )

    assert fitted.weights["weight"].shape == (3, features.shape[-1])
    assert fitted.weights["bias"].shape == (3,)
    assert fitted.metrics.train_accuracy == 1.0
    logits = torch.nn.functional.linear(
        train_api.normalize_features(features),
        fitted.weights["weight"],
        fitted.weights["bias"],
    )
    assert torch.equal(logits.argmax(dim=-1), labels)


def test_fit_classification_head__time_budget() -> None:
    features, labels = _separable(num_classes=2)

    start = time.monotonic()
    train_api.fit_classification_head(
        features=features,
        labels=labels,
        num_classes=2,
        steps=1_000_000,
        lr=1e-1,
        weight_decay=0.0,
        max_seconds=0.5,
    )

    assert time.monotonic() - start < 5.0


def test_boxes_to_target() -> None:
    target = train_api.boxes_to_target(
        boxes=[[0.0, 0.0, 100.0, 50.0]], labels=[1], size=(100, 200)
    )

    # cxcywh normalized by (width=200, height=100).
    assert torch.allclose(target["boxes"], torch.tensor([[0.25, 0.25, 0.5, 0.5]]))
    assert torch.equal(target["labels"], torch.tensor([1]))


def test_boxes_to_target__empty() -> None:
    target = train_api.boxes_to_target(boxes=[], labels=[], size=(100, 100))

    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_fit_detection_head() -> None:
    torch.manual_seed(0)
    model = _detector()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    buffers_before = {name: b.detach().clone() for name, b in model.named_buffers()}

    fitted = train_api.fit_detection_head(
        model=model,
        images=torch.rand(2, 3, *IMAGE_SIZE),
        targets=_targets(),
        steps=2,
        lr=1e-3,
        weight_decay=0.0,
        max_seconds=60.0,
    )

    assert torch.isfinite(torch.tensor(fitted.metrics.train_loss))
    assert fitted.metrics.train_accuracy is None
    # Only the class head is trained, everything else keeps its pretrained values.
    changed = {
        name for name, p in model.named_parameters() if not torch.equal(before[name], p)
    }
    assert changed
    assert changed <= set(fitted.weights)
    # Frozen batch norm statistics must not drift.
    assert all(
        torch.equal(buffers_before[name], b) for name, b in model.named_buffers()
    )


def test_fit_detection_head__trains_all_class_head_parameters() -> None:
    torch.manual_seed(0)
    model = _detector()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    fitted = train_api.fit_detection_head(
        model=model,
        images=torch.rand(2, 3, *IMAGE_SIZE),
        targets=_targets(),
        steps=3,
        lr=1e-2,
        weight_decay=0.0,
        max_seconds=60.0,
    )

    # The auxiliary and denoising losses must reach every class head parameter,
    # which only happens while the decoder is in training mode.
    assert set(fitted.weights) == {
        name for name, p in model.named_parameters() if not torch.equal(before[name], p)
    }


def test_fit_detection_head__restores_eval_mode() -> None:
    model = _detector()

    train_api.fit_detection_head(
        model=model,
        images=torch.rand(2, 3, *IMAGE_SIZE),
        targets=_targets(),
        steps=1,
        lr=1e-3,
        weight_decay=0.0,
        max_seconds=60.0,
    )

    assert not model.training


@pytest.mark.parametrize("num_classes", [1, 3])
def test_fit_detection_head__num_classes(num_classes: int) -> None:
    model = _detector(num_classes=num_classes)

    fitted = train_api.fit_detection_head(
        model=model,
        images=torch.rand(2, 3, *IMAGE_SIZE),
        targets=_targets(num_classes=num_classes),
        steps=1,
        lr=1e-3,
        weight_decay=0.0,
        max_seconds=60.0,
    )

    assert fitted.weights["decoder.enc_score_head.weight"].shape[0] == num_classes


def test_load_detector__architecture_only() -> None:
    train_api._load_checkpoint.cache_clear()

    model = train_api.load_detector(
        model_name="dinov2/_vittest14-ltdetrv2",
        classes={0: "cat", 1: "dog"},
        image_size=IMAGE_SIZE,
        device=torch.device("cpu"),
    )

    assert model.classes == {0: "cat", 1: "dog"}
    assert model.decoder.enc_score_head.out_features == 2
    assert model.image_size == IMAGE_SIZE


def test_load_detector__unknown_model() -> None:
    train_api._load_checkpoint.cache_clear()

    with pytest.raises(ValueError, match="Unknown model name"):
        train_api.load_detector(
            model_name="not-a-model",
            classes={0: "cat"},
            image_size=IMAGE_SIZE,
            device=torch.device("cpu"),
        )
