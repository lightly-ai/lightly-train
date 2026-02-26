#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import torch

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
            log_classwise=False,
            classwise_metric_args=None,
            loss_names=["loss"],
        )
        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([0, 1])
        metric.update(preds, target)
        metric.update_loss({"loss": torch.tensor(0.5)}, weight=1)
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
            log_classwise=False,
            classwise_metric_args=None,
            loss_names=["loss"],
        )
        preds = torch.tensor([[0.8, 0.5, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([[1, 1, 0], [0, 1, 1]])
        metric.update(preds, target)
        metric.update_loss({"loss": torch.tensor(0.5)}, weight=1)
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

    def test_reset(self) -> None:
        metric = ClassificationTaskMetric(
            task_metric_args=MulticlassClassificationTaskMetricArgs(),
            split="val",
            class_names=["cat", "dog", "bird"],
            log_classwise=False,
            classwise_metric_args=None,
            loss_names=["loss"],
        )
        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([0, 1])
        metric.update(preds, target)
        metric.update_loss({"loss": torch.tensor(1.0)}, weight=1)
        result1 = metric.compute()
        metric.reset()
        metric.update(preds, target)
        metric.update_loss({"loss": torch.tensor(0.1)}, weight=1)
        result2 = metric.compute()
        assert result1 != result2

    def test_update__classwise(self) -> None:
        metric = ClassificationTaskMetric(
            task_metric_args=MulticlassClassificationTaskMetricArgs(),
            split="val",
            class_names=["cat", "dog"],
            log_classwise=True,
            classwise_metric_args=None,
            loss_names=["loss"],
        )
        preds = torch.tensor([[0.8, 0.1], [0.1, 0.7]])
        target = torch.tensor([0, 1])
        metric.update(preds, target)
        metric.update_loss({"loss": torch.tensor(0.5)}, weight=1)
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
