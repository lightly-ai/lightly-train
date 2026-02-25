#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import torch

from lightly_train._metrics.panoptic_segmentation.task_metric import (
    PanopticSegmentationTaskMetric,
    PanopticSegmentationTaskMetricArgs,
)


class TestPanopticSegmentationTaskMetric:
    def test_update(self) -> None:
        metric = PanopticSegmentationTaskMetric(
            task_metric_args=PanopticSegmentationTaskMetricArgs(),
            split="val",
            things=[0, 1],
            stuffs=[2],
            thing_class_names=["cat", "dog"],
            stuff_class_names=["road"],
            log_classwise=False,
        )
        preds = torch.randint(0, 3, (1, 10, 10, 2), dtype=torch.int32)
        target = torch.randint(0, 3, (1, 10, 10, 2), dtype=torch.int32)
        metric.update(preds, target)
        result = metric.compute()
        assert result.metrics.keys() == {
            "val_loss",
            "val_metric/pq",
            "val_metric/sq",
            "val_metric/rq",
        }

    def test_update__classwise(self) -> None:
        metric = PanopticSegmentationTaskMetric(
            task_metric_args=PanopticSegmentationTaskMetricArgs(),
            split="val",
            things=[0, 1],
            stuffs=[2],
            thing_class_names=["cat", "dog"],
            stuff_class_names=["road"],
            log_classwise=True,
        )
        preds = torch.randint(0, 3, (1, 10, 10, 2), dtype=torch.int32)
        target = torch.randint(0, 3, (1, 10, 10, 2), dtype=torch.int32)
        metric.update(preds, target)
        result = metric.compute()
        assert result.metrics.keys() == {
            "val_loss",
            "val_metric/pq",
            "val_metric/sq",
            "val_metric/rq",
            "val_metric_classwise/pq_cat",
            "val_metric_classwise/sq_cat",
            "val_metric_classwise/rq_cat",
            "val_metric_classwise/pq_dog",
            "val_metric_classwise/sq_dog",
            "val_metric_classwise/rq_dog",
            "val_metric_classwise/pq_road",
            "val_metric_classwise/sq_road",
            "val_metric_classwise/rq_road",
        }
