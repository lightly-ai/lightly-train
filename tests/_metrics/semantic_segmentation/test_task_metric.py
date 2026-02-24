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
    SemanticSegmentationTaskMetricArgs,
)


class TestSemanticSegmentationTaskMetricArgs:
    def test_get_metrics(self) -> None:
        """Test that SemanticSegmentationTaskMetricArgs can create metrics."""
        metric_args = SemanticSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            num_classes=3,
            ignore_index=255,
            log_classwise=False,
        )

        # Semantic segmentation format: (B, H, W) for preds and targets
        preds = torch.randint(0, 3, (2, 100, 100))
        target = torch.randint(0, 3, (2, 100, 100))
        segmentation_task_metric.update(preds, target)

        result = segmentation_task_metric.compute()
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

        expected_metrics = {
            "val_metric/miou",
        }
        assert expected_metrics.issubset(set(result.metrics.keys()))
        assert result.best_metric_key == "val_metric/miou"
        assert isinstance(result.best_metric_value, float)

    def test_get_metrics__classwise(self) -> None:
        """Test that classwise metrics are created correctly."""
        metric_args = SemanticSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            num_classes=3,
            ignore_index=255,
            log_classwise=True,
        )

        assert segmentation_task_metric.metrics_classwise is not None

        # Semantic segmentation format with multiple samples
        preds = torch.randint(0, 3, (2, 100, 100))
        target = torch.randint(0, 3, (2, 100, 100))
        segmentation_task_metric.update(preds, target)

        result = segmentation_task_metric.compute()
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

        regular_metrics = {k for k in result.metrics.keys() if "classwise" not in k}
        result_classwise = {k: v for k, v in result.metrics.items() if "classwise" in k}
        assert isinstance(result_classwise, dict)
        assert len(result_classwise) > 0

        assert len(regular_metrics) > 0
        expected_regular_metrics = {
            "val_metric/miou",
        }
        assert expected_regular_metrics.issubset(set(regular_metrics))

        # Check for classwise metrics (one per class)
        for class_idx in [0, 1, 2]:
            expected_classwise_metrics = {
                f"val_metric_classwise/iou_{class_idx}",
            }
            assert expected_classwise_metrics.issubset(set(result_classwise.keys()))

    def test_get_display_names(self) -> None:
        """Test that get_display_names returns correct display names."""
        metric_args = SemanticSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            num_classes=3,
            ignore_index=255,
            log_classwise=False,
        )

        display_names = segmentation_task_metric.get_display_names()

        assert isinstance(display_names, dict)
        assert len(display_names) > 0

        expected_mappings = {
            "val_metric/miou": "Val mIoU",
        }
        for metric_name, expected_display in expected_mappings.items():
            assert metric_name in display_names
            assert display_names[metric_name] == expected_display

    def test_get_display_names__classwise(self) -> None:
        """Test that get_display_names works with classwise metrics."""
        metric_args = SemanticSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            num_classes=3,
            ignore_index=255,
            log_classwise=True,
        )

        display_names = segmentation_task_metric.get_display_names()

        assert "val_metric/miou" in display_names
        assert "val_metric_classwise/iou_0" in display_names
        assert "val_metric_classwise/iou_1" in display_names
        assert "val_metric_classwise/iou_2" in display_names

        assert display_names["val_metric/miou"] == "Val mIoU"
        assert display_names["val_metric_classwise/iou_0"] == "Val IoU"

    def test_reset(self) -> None:
        """Test that reset() clears all metrics."""
        metric_args = SemanticSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            num_classes=3,
            ignore_index=255,
            log_classwise=True,
        )

        # First scenario: random predictions
        torch.manual_seed(42)
        preds = torch.randint(0, 3, (2, 100, 100))
        target = torch.randint(0, 3, (2, 100, 100))
        segmentation_task_metric.update(preds, target)

        result_before = segmentation_task_metric.compute()
        assert len(result_before.metrics) > 0

        segmentation_task_metric.reset()

        # Second scenario: different random predictions
        torch.manual_seed(123)
        preds2 = torch.randint(0, 3, (2, 100, 100))
        target2 = torch.randint(0, 3, (2, 100, 100))
        segmentation_task_metric.update(preds2, target2)
        result_after = segmentation_task_metric.compute()

        # Results should be different
        assert result_after.metrics != result_before.metrics
