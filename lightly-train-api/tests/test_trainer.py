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

from lightly_train_api import encoder, trainer


def _separable(
    num_classes: int, per_class: int = 8, dim: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.eye(num_classes, dim) * 10.0
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    features = centers[labels] + 0.01 * torch.randn(len(labels), dim)
    return features, labels


def test_fit_linear_head() -> None:
    features, labels = _separable(num_classes=3)
    fitted = trainer.fit_linear_head(features=features, labels=labels, num_classes=3)
    assert fitted.weights.weight.shape == (3, features.shape[-1])
    assert fitted.metrics.train_accuracy == 1.0
    logits = torch.nn.functional.linear(
        encoder.normalize_features(features),
        fitted.weights.weight,
        fitted.weights.bias,
    )
    assert torch.equal(logits.argmax(dim=-1), labels)


def test_fit_linear_head__num_classes() -> None:
    features, labels = _separable(num_classes=4)
    fitted = trainer.fit_linear_head(features=features, labels=labels, num_classes=4)
    assert fitted.weights.weight.shape[0] == 4
    assert fitted.weights.bias.shape[0] == 4


def test_dump_weights() -> None:
    features, labels = _separable(num_classes=2)
    weights = trainer.fit_linear_head(
        features=features, labels=labels, num_classes=2
    ).weights
    loaded = trainer.load_weights(trainer.dump_weights(weights))
    assert torch.equal(loaded.weight, weights.weight)
    assert torch.equal(loaded.bias, weights.bias)


def test_fit_linear_head__time_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from lightly_train_api import settings

    monkeypatch.setenv("LIGHTLY_TRAIN_API_TRAIN_STEPS", "1000000")
    monkeypatch.setenv("LIGHTLY_TRAIN_API_TRAIN_MAX_SECONDS", "0.5")
    settings.get_settings.cache_clear()

    features, labels = _separable(num_classes=2)
    start = time.monotonic()
    trainer.fit_linear_head(features=features, labels=labels, num_classes=2)
    assert time.monotonic() - start < 5.0
