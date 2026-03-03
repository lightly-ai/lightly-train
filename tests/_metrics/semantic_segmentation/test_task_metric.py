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


from lightly_train._metrics.semantic_segmentation.task_metric import (
    SemanticSegmentationTaskMetric,
    SemanticSegmentationTaskMetricArgs,
)


class TestSemanticSegmentationTaskMetric:
    def test_update(self) -> None:
        metric = SemanticSegmentationTaskMetric(
            task_metric_args=SemanticSegmentationTaskMetricArgs(classwise=False),
            split="val",
            class_names=["cat", "dog"],
            ignore_index=None,
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
            task_metric_args=SemanticSegmentationTaskMetricArgs(classwise=False),
            split="val",
            class_names=["cat", "dog"],
            ignore_index=255,
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
