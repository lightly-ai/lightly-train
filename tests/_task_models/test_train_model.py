#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
import torch
from lightning_fabric import Fabric

from lightly_train._data.image_classification_dataset import (
    ImageClassificationMulticlassDataArgs,
)
from lightly_train._metrics.classification.task_metric import (
    MulticlassClassificationTaskMetricArgs,
)
from lightly_train._task_models.image_classification.train_model import (
    ImageClassificationTrain,
    ImageClassificationTrainArgs,
)
from lightly_train._task_models.image_classification.transforms import (
    ImageClassificationTrainTransformArgs,
    ImageClassificationValTransformArgs,
)


def _make_train_model(gradient_clip_val: float) -> ImageClassificationTrain:
    data_args = ImageClassificationMulticlassDataArgs(
        train="/tmp/train",
        val="/tmp/val",
        classes={0: "a", 1: "b"},
    )
    train_transform_args = ImageClassificationTrainTransformArgs()
    train_transform_args.resolve_auto(model_init_args={})
    val_transform_args = ImageClassificationValTransformArgs()
    val_transform_args.resolve_auto(model_init_args={})
    model_args = ImageClassificationTrainArgs(gradient_clip_val=gradient_clip_val)
    model_args.resolve_auto(
        total_steps=10,
        gradient_accumulation_steps=1,
        train_num_batches=4,
        model_name="dinov2/_vittest14",
        model_init_args={},
        data_args=data_args,
    )
    return ImageClassificationTrain(
        model_name="dinov2/_vittest14",
        model_args=model_args,
        data_args=data_args,
        train_transform_args=train_transform_args,
        val_transform_args=val_transform_args,
        load_weights=False,
        metric_args=MulticlassClassificationTaskMetricArgs(),
        gradient_accumulation_steps=1,
    )


@pytest.mark.parametrize("gradient_clip_val", [3.0, 0.0])
def test_clip_gradients__returns_total_norm(gradient_clip_val: float) -> None:
    """clip_gradients returns the pre-clipping total gradient norm.

    The returned norm must match the total L2 norm of all gradients, independent
    of whether clipping is enabled (max_norm=val) or disabled (max_norm=inf).
    """
    train_model = _make_train_model(gradient_clip_val=gradient_clip_val)
    fabric = Fabric(accelerator="cpu", devices=1)
    train_model, optimizer = fabric.setup(
        train_model, torch.optim.SGD(train_model.parameters(), lr=1e-3)
    )

    # Forward + backward to populate gradients.
    images = torch.randn(2, 3, 14, 14)
    targets = torch.tensor([0, 1])
    logits = train_model.forward(images)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    fabric.backward(loss)

    # Compute the expected total norm independently of clip_grad_norm_.
    grads = [
        p.grad.detach().flatten()
        for p in train_model.parameters()
        if p.grad is not None
    ]
    expected_norm = torch.linalg.vector_norm(torch.cat(grads))

    returned_norm = train_model.clip_gradients(fabric=fabric, optimizer=optimizer)

    assert returned_norm is not None
    assert torch.allclose(returned_norm, expected_norm, rtol=1e-5), (
        f"Expected {expected_norm}, got {returned_norm}"
    )


def test_clip_gradients__returns_none_when_no_gradients() -> None:
    """clip_gradients still returns a (zero) norm tensor when gradients are absent.

    Fabric's clip_gradients_norm delegates to torch.nn.utils.clip_grad_norm_, which
    returns 0.0 for parameters without gradients. This confirms the method always
    returns a tensor rather than None for task models.
    """
    train_model = _make_train_model(gradient_clip_val=3.0)
    fabric = Fabric(accelerator="cpu", devices=1)
    train_model, optimizer = fabric.setup(
        train_model, torch.optim.SGD(train_model.parameters(), lr=1e-3)
    )

    # No backward — gradients are None.
    returned_norm = train_model.clip_gradients(fabric=fabric, optimizer=optimizer)

    assert returned_norm is not None
    assert float(returned_norm) == pytest.approx(0.0)
