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

from lightly_train._metrics.panoptic_segmentation.task_metric import (
    PanopticSegmentationTaskMetric,
    PanopticSegmentationTaskMetricArgs,
)


class TestPanopticSegmentationTaskMetric:
    def test_update_with_predictions__default(self) -> None:
        metric = PanopticSegmentationTaskMetric(
            task_metric_args=PanopticSegmentationTaskMetricArgs(),
            split="val",
            things=[0, 1],
            stuffs=[2],
            thing_class_names=["cat", "dog"],
            stuff_class_names=["road"],
            loss_names=["loss"],
        )
        preds = torch.randint(0, 3, (1, 10, 10, 2), dtype=torch.int32)
        target = torch.randint(0, 3, (1, 10, 10, 2), dtype=torch.int32)
        metric.update_with_predictions(preds, target)
        result = metric.compute_aggregated_values()
        assert result.metric_values.keys() == {
            "val_loss",
            "val_metric/pq",
            "val_metric/sq",
            "val_metric/rq",
        }

    def test_update_with_predictions__classwise(self) -> None:
        metric = PanopticSegmentationTaskMetric(
            task_metric_args=PanopticSegmentationTaskMetricArgs(classwise=True),
            split="val",
            things=[0, 1],
            stuffs=[2, 3],
            thing_class_names=["cat", "dog"],
            stuff_class_names=["road", "ignore"],
            loss_names=["loss"],
        )
        preds = torch.randint(0, 3, (1, 10, 10, 2), dtype=torch.int32)
        target = torch.randint(0, 3, (1, 10, 10, 2), dtype=torch.int32)
        metric.update_with_predictions(preds, target)
        result = metric.compute_aggregated_values()
        assert result.metric_values.keys() == {
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
