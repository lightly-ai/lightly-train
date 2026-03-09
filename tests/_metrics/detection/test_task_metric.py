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


from lightly_train._metrics.detection.task_metric import (
    ObjectDetectionTaskMetric,
    ObjectDetectionTaskMetricArgs,
)


class TestObjectDetectionTaskMetric:
    def test_update__default(self) -> None:
        metric = ObjectDetectionTaskMetric(
            task_metric_args=ObjectDetectionTaskMetricArgs(),
            split="val",
            class_names=["cat", "dog"],
            box_format="xyxy",
            loss_names=["loss", "loss_vfl", "loss_bbox", "loss_giou"],
        )
        metric.update_with_predictions(
            preds=[
                {
                    "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            ],
            target=[
                {
                    "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                    "labels": torch.tensor([0]),
                }
            ],
        )
        metric.update_with_losses(
            {
                "loss": torch.tensor(0.5),
                "loss_vfl": torch.tensor(0.1),
                "loss_bbox": torch.tensor(0.2),
                "loss_giou": torch.tensor(0.2),
            },
            weight=1,
        )
        result = metric.compute()
        assert result.metrics["val_loss"] == 0.5
        assert result.metrics.keys() == {
            "val_loss",
            "val_loss/loss_vfl",
            "val_loss/loss_bbox",
            "val_loss/loss_giou",
            "val_metric/map",
            "val_metric/map_50",
            "val_metric/map_75",
            "val_metric/map_small",
            "val_metric/map_medium",
            "val_metric/map_large",
            "val_metric/mar_1",
            "val_metric/mar_10",
            "val_metric/mar_100",
            "val_metric/mar_small",
            "val_metric/mar_medium",
            "val_metric/mar_large",
        }

    def test_update__classwise(self) -> None:
        metric = ObjectDetectionTaskMetric(
            task_metric_args=ObjectDetectionTaskMetricArgs(classwise=True),
            split="val",
            class_names=["cat", "dog"],
            box_format="xyxy",
            loss_names=["loss", "loss_vfl", "loss_bbox", "loss_giou"],
        )
        metric.update_with_predictions(
            preds=[
                {
                    "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            ],
            target=[
                {
                    "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                    "labels": torch.tensor([0]),
                }
            ],
        )
        metric.update_with_losses(
            {
                "loss": torch.tensor(0.5),
                "loss_vfl": torch.tensor(0.1),
                "loss_bbox": torch.tensor(0.2),
                "loss_giou": torch.tensor(0.2),
            },
            weight=1,
        )
        result = metric.compute()
        assert result.metrics["val_loss"] == 0.5
        print(result.metrics.keys())
        assert result.metrics.keys() == {
            "val_loss",
            "val_loss/loss_vfl",
            "val_loss/loss_bbox",
            "val_loss/loss_giou",
            "val_metric/map",
            "val_metric/map_50",
            "val_metric/map_75",
            "val_metric/map_small",
            "val_metric/map_medium",
            "val_metric/map_large",
            "val_metric/mar_1",
            "val_metric/mar_10",
            "val_metric/mar_100",
            "val_metric/mar_small",
            "val_metric/mar_medium",
            "val_metric/mar_large",
            "val_metric_classwise/map_cat",
            # val_metric_classwise/map_dog is not returned because we only have one
            # sample and it belongs to class "cat"
        }
