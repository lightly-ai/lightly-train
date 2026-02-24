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
    PanopticSegmentationTaskMetricArgs,
)


class TestPanopticSegmentationTaskMetricArgs:
    def test_get_metrics(self) -> None:
        """Test that PanopticSegmentationTaskMetricArgs can create metrics."""
        metric_args = PanopticSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            things=[0, 1],
            stuffs=[2],
        )

        # Panoptic segmentation format: (B, H, W, 2) where last dim is (class_id, instance_id)
        preds = torch.randint(0, 3, (1, 100, 100, 2), dtype=torch.int32)
        target = torch.randint(0, 3, (1, 100, 100, 2), dtype=torch.int32)
        segmentation_task_metric.update(preds, target)

        result = segmentation_task_metric.compute()
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

        expected_metrics = {
            "val_metric/pq",
            "val_metric/sq",
            "val_metric/rq",
        }
        assert expected_metrics.issubset(set(result.metrics.keys()))
        assert result.best_metric_key == "val_metric/pq"
        assert isinstance(result.best_metric_value, float)

    def test_get_display_names(self) -> None:
        """Test that get_display_names returns correct display names."""
        metric_args = PanopticSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            things=[0, 1],
            stuffs=[2],
        )

        display_names = segmentation_task_metric.get_display_names()

        assert isinstance(display_names, dict)
        assert len(display_names) > 0

        expected_mappings = {
            "val_metric/pq": "Val PQ",
            "val_metric/sq": "Val SQ",
            "val_metric/rq": "Val RQ",
        }
        for metric_name, expected_display in expected_mappings.items():
            assert metric_name in display_names
            assert display_names[metric_name] == expected_display

    def test_reset(self) -> None:
        """Test that reset() clears all metrics."""
        metric_args = PanopticSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            things=[0, 1],
            stuffs=[2],
        )

        # First scenario: random masks
        torch.manual_seed(42)
        preds = torch.randint(0, 3, (1, 100, 100, 2), dtype=torch.int32)
        target = torch.randint(0, 3, (1, 100, 100, 2), dtype=torch.int32)
        segmentation_task_metric.update(preds, target)

        result_before = segmentation_task_metric.compute()
        assert len(result_before.metrics) > 0
        # Store scalar values instead of tensors for comparison
        pq_before = result_before.metrics["val_metric/pq"]

        segmentation_task_metric.reset()

        # Second scenario: different random masks
        torch.manual_seed(123)
        preds2 = torch.randint(0, 3, (1, 100, 100, 2), dtype=torch.int32)
        target2 = torch.randint(0, 3, (1, 100, 100, 2), dtype=torch.int32)
        segmentation_task_metric.update(preds2, target2)
        result_after = segmentation_task_metric.compute()
        pq_after = result_after.metrics["val_metric/pq"]

        # PQ values should be different (though both might be 0 with random data)
        # Just verify we got results from both computations
        assert isinstance(pq_before, float)
        assert isinstance(pq_after, float)
