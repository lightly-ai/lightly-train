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
from lightning_utilities.core.imports import RequirementCache

if RequirementCache("torchmetrics<1.5"):
    # Skip test if torchmetrics version is too old. This can happen if SuperGradients
    # is installed which requires torchmetrics==0.8
    pytest.skip("Old torchmetrics version", allow_module_level=True)


from lightly_train._metrics.classification.task_metric import (
    ClassificationTaskMetric,
    MulticlassClassificationTaskMetricArgs,
    MultilabelClassificationTaskMetricArgs,
)


class TestClassificationTaskMetric:
    def test_update__multiclass(self) -> None:
        metric = ClassificationTaskMetric(
            task_metric_args=MulticlassClassificationTaskMetricArgs(),
            split="val",
            class_names=["cat", "dog", "bird"],
            loss_names=["loss"],
        )
        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([0, 1])
        metric.update_with_predictions(preds, target)
        metric.update_with_losses({"loss": torch.tensor(0.5)}, weight=1)
        result = metric.compute()
        assert result.metrics["val_loss"] == 0.5
        assert result.metrics.keys() == {
            "val_loss",
            "val_metric/top1_acc_micro",
            "val_metric/f1_macro",
            "val_metric/precision_macro",
            "val_metric/recall_macro",
        }

    def test_update__multilabel(self) -> None:
        metric = ClassificationTaskMetric(
            task_metric_args=MultilabelClassificationTaskMetricArgs(),
            split="val",
            class_names=["cat", "dog", "bird"],
            loss_names=["loss"],
        )
        preds = torch.tensor([[0.8, 0.5, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([[1, 1, 0], [0, 1, 1]])
        metric.update_with_predictions(preds, target)
        metric.update_with_losses({"loss": torch.tensor(0.5)}, weight=1)
        result = metric.compute()
        assert result.metrics["val_loss"] == 0.5
        assert result.metrics.keys() == {
            "val_loss",
            "val_metric/accuracy_micro",
            "val_metric/auroc_macro",
            "val_metric/avg_precision_macro",
            "val_metric/f1_macro",
            "val_metric/hamming_distance",
        }

    def test_update__reset(self) -> None:
        metric = ClassificationTaskMetric(
            task_metric_args=MulticlassClassificationTaskMetricArgs(),
            split="val",
            class_names=["cat", "dog", "bird"],
            loss_names=["loss"],
        )
        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([0, 1])
        metric.update_with_predictions(preds, target)
        metric.update_with_losses({"loss": torch.tensor(1.0)}, weight=1)
        result1 = metric.compute()
        metric.reset()
        metric.update_with_predictions(preds, target)
        metric.update_with_losses({"loss": torch.tensor(0.1)}, weight=1)
        result2 = metric.compute()
        assert result1 != result2

    def test_update__classwise(self) -> None:
        metric = ClassificationTaskMetric(
            task_metric_args=MulticlassClassificationTaskMetricArgs(classwise=True),
            split="val",
            class_names=["cat", "dog"],
            loss_names=["loss"],
        )
        preds = torch.tensor([[0.8, 0.1], [0.1, 0.7]])
        target = torch.tensor([0, 1])
        metric.update_with_predictions(preds, target)
        metric.update_with_losses({"loss": torch.tensor(0.5)}, weight=1)
        result = metric.compute()
        assert result.metrics["val_loss"] == 0.5
        assert result.metrics.keys() == {
            "val_loss",
            "val_metric/top1_acc_micro",
            "val_metric/f1_macro",
            "val_metric/precision_macro",
            "val_metric/recall_macro",
            "val_metric_classwise/top1_acc_cat",
            "val_metric_classwise/top1_acc_dog",
            "val_metric_classwise/f1_cat",
            "val_metric_classwise/f1_dog",
            "val_metric_classwise/precision_cat",
            "val_metric_classwise/precision_dog",
            "val_metric_classwise/recall_cat",
            "val_metric_classwise/recall_dog",
        }
