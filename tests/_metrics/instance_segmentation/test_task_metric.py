#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import torch

from lightly_train._metrics.instance_segmentation.task_metric import (
    InstanceSegmentationTaskMetric,
    InstanceSegmentationTaskMetricArgs,
)


class TestInstanceSegmentationTaskMetric:
    def test_update(self) -> None:
        metric = InstanceSegmentationTaskMetric(
            task_metric_args=InstanceSegmentationTaskMetricArgs(),
            split="val",
            class_names=["cat", "dog"],
            log_classwise=False,
            loss_names=["loss"],
        )
        metric.update(
            preds=[
                {
                    "masks": torch.tensor([[[0, 0, 0], [0, 1, 1], [0, 1, 1]]]).bool(),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            ],
            target=[
                {
                    "masks": torch.tensor([[[0, 0, 0], [0, 1, 1], [0, 1, 1]]]).bool(),
                    "labels": torch.tensor([0]),
                }
            ],
        )
        metric.update_loss({"loss": torch.tensor(0.5)}, weight=1)
        result = metric.compute()
        assert result.metrics["val_loss"] == 0.5
        assert result.metrics.keys() == {
            "val_loss",
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
        metric = InstanceSegmentationTaskMetric(
            task_metric_args=InstanceSegmentationTaskMetricArgs(),
            split="val",
            class_names=["cat", "dog"],
            log_classwise=True,
            loss_names=["loss"],
        )
        metric.update(
            preds=[
                {
                    "masks": torch.tensor([[[0, 0, 0], [0, 1, 1], [0, 1, 1]]]).bool(),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            ],
            target=[
                {
                    "masks": torch.tensor([[[0, 0, 0], [0, 1, 1], [0, 1, 1]]]).bool(),
                    "labels": torch.tensor([0]),
                }
            ],
        )
        metric.update_loss({"loss": torch.tensor(0.5)}, weight=1)
        result = metric.compute()
        assert result.metrics["val_loss"] == 0.5
        print(result.metrics.keys())
        assert result.metrics.keys() == {
            "val_loss",
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
