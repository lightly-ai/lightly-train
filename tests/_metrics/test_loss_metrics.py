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

from lightly_train._metrics.loss_metrics import LossMetrics


class TestLossMetrics:
    def test_compute__train_split(self) -> None:
        # RunningMean(window=1) keeps only the last value.
        metric = LossMetrics(split="train", loss_names=["loss"])
        metric.update({"loss": torch.tensor(1.0)}, weight=1)
        metric.update({"loss": torch.tensor(3.0)}, weight=1)
        result = metric.compute()
        assert result == {"train_loss": pytest.approx(3.0)}

    def test_compute__val_split(self) -> None:
        # MeanMetric accumulates a weighted mean across batches.
        metric = LossMetrics(split="val", loss_names=["loss"])
        metric.update({"loss": torch.tensor(1.0)}, weight=1)
        metric.update({"loss": torch.tensor(3.0)}, weight=1)
        result = metric.compute()
        assert result == {"val_loss": pytest.approx(2.0)}

    def test_compute__multiple_losses(self) -> None:
        # "loss" and other names coexist with different key formats.
        metric = LossMetrics(split="val", loss_names=["loss", "loss_vfl"])
        metric.update(
            {"loss": torch.tensor(1.0), "loss_vfl": torch.tensor(2.0)}, weight=1
        )
        assert metric.compute() == {
            "val_loss": pytest.approx(1.0),
            "val_loss/loss_vfl": pytest.approx(2.0),
        }

    def test_update__mismatched_keys(self) -> None:
        metric = LossMetrics(split="val", loss_names=["loss"])
        with pytest.raises(ValueError):
            metric.update({"wrong_key": torch.tensor(1.0)}, weight=1)

    def test_update__missing_keys(self) -> None:
        metric = LossMetrics(split="val", loss_names=["loss", "loss_vfl"])
        with pytest.raises(ValueError):
            metric.update({"loss": torch.tensor(1.0)}, weight=1)
