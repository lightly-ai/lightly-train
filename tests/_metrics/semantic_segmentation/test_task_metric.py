#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import torch

from lightly_train._metrics.semantic_segmentation.task_metric import (
    SemanticSegmentationTaskMetric,
    SemanticSegmentationTaskMetricArgs,
)


class TestSemanticSegmentationTaskMetric:
    def test_update(self) -> None:
        metric = SemanticSegmentationTaskMetric(
            task_metric_args=SemanticSegmentationTaskMetricArgs(),
            split="val",
            class_names=["cat", "dog"],
            ignore_index=None,
            log_classwise=False,
            loss_names=["loss"],
        )
        preds = torch.zeros(2, 10, 10, dtype=torch.long)
        target = torch.zeros(2, 10, 10, dtype=torch.long)
        metric.update(preds, target)
        metric.update_loss({"loss": torch.tensor(0.5)}, weight=1)
        result = metric.compute()
        assert result.metrics["val_loss"] == 0.5
        assert result.metrics.keys() == {
            "val_loss",
            "val_metric/miou",
        }

    def test_update__ignore_index(self) -> None:
        metric = SemanticSegmentationTaskMetric(
            task_metric_args=SemanticSegmentationTaskMetricArgs(),
            split="val",
            class_names=["cat", "dog"],
            ignore_index=255,
            log_classwise=False,
            loss_names=["loss"],
        )
        preds = torch.zeros(2, 4, 4, dtype=torch.long)
        target = torch.zeros(2, 4, 4, dtype=torch.long)
        target[:, 0, 0] = 255  # ignored pixels
        metric.update(preds, target)
        metric.update_loss({"loss": torch.tensor(0.5)}, weight=1)
        result = metric.compute()
        assert result.metrics["val_loss"] == 0.5
        assert result.metrics.keys() == {"val_loss", "val_metric/miou"}

    def test_update__classwise(self) -> None:
        metric = SemanticSegmentationTaskMetric(
            task_metric_args=SemanticSegmentationTaskMetricArgs(),
            split="val",
            class_names=["cat", "dog"],
            ignore_index=None,
            log_classwise=True,
            loss_names=["loss"],
        )
        preds = torch.zeros(2, 10, 10, dtype=torch.long)
        target = torch.zeros(2, 10, 10, dtype=torch.long)
        metric.update(preds, target)
        metric.update_loss({"loss": torch.tensor(0.5)}, weight=1)
        result = metric.compute()
        assert result.metrics["val_loss"] == 0.5
        assert result.metrics.keys() == {
            "val_loss",
            "val_metric/miou",
            "val_metric_classwise/iou_cat",
            "val_metric_classwise/iou_dog",
        }
