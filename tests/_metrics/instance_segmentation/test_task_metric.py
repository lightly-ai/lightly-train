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
    InstanceSegmentationTaskMetricArgs,
)


class TestInstanceSegmentationTaskMetricArgs:
    def test_get_metrics(self) -> None:
        """Test that InstanceSegmentationTaskMetricArgs can create metrics."""
        metric_args = InstanceSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            split="val",
            class_names=["cat", "dog", "bird"],
            log_classwise=False,
        )

        # Instance segmentation format: list of dicts with "masks", "scores", "labels"
        preds = [
            {
                "masks": torch.randint(0, 2, (1, 100, 100), dtype=torch.bool),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "masks": torch.randint(0, 2, (1, 100, 100), dtype=torch.bool),
                "labels": torch.tensor([0]),
            }
        ]
        segmentation_task_metric.update(preds, targets)

        result = segmentation_task_metric.compute()
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

        expected_metrics = {
            "val_metric/map",
            "val_metric/map_50",
            "val_metric/map_75",
            "val_metric/map_small",
            "val_metric/map_medium",
            "val_metric/map_large",
        }
        assert expected_metrics.issubset(set(result.metrics.keys()))
        assert result.best_metric_key == "val_metric/map"
        assert isinstance(result.best_metric_value, float)

    def test_get_metrics__classwise(self) -> None:
        """Test that classwise metrics are created correctly."""
        metric_args = InstanceSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            split="val",
            class_names=["cat__type_a", "dog__breed__b", "bird"],
            log_classwise=True,
        )

        assert segmentation_task_metric.metrics_classwise is not None

        # Instance segmentation format with multiple predictions
        preds = [
            {
                "masks": torch.randint(0, 2, (2, 100, 100), dtype=torch.bool),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        targets = [
            {
                "masks": torch.randint(0, 2, (2, 100, 100), dtype=torch.bool),
                "labels": torch.tensor([0, 1]),
            }
        ]
        segmentation_task_metric.update(preds, targets)

        result = segmentation_task_metric.compute()
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

        regular_metrics = {k for k in result.metrics.keys() if "classwise" not in k}
        result_classwise = {k: v for k, v in result.metrics.items() if "classwise" in k}
        assert isinstance(result_classwise, dict)
        assert len(result_classwise) > 0

        assert len(regular_metrics) > 0
        expected_regular_metrics = {
            "val_metric/map",
            "val_metric/map_50",
            "val_metric/map_75",
            "val_metric/map_small",
            "val_metric/map_medium",
            "val_metric/map_large",
        }
        assert expected_regular_metrics.issubset(set(regular_metrics))

        # Check for classwise metrics (only classes actually in predictions)
        for class_name in ["cat__type_a", "dog__breed__b"]:
            expected_classwise_metrics = {
                f"val_metric_classwise/map_{class_name}",
            }
            assert expected_classwise_metrics.issubset(set(result_classwise.keys()))

    def test_get_display_names(self) -> None:
        """Test that get_display_names returns correct display names."""
        metric_args = InstanceSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            split="val",
            class_names=["cat", "dog", "bird"],
            log_classwise=False,
        )

        display_names = segmentation_task_metric.get_display_names()

        assert isinstance(display_names, dict)
        assert len(display_names) > 0

        expected_mappings = {
            "val_metric/map": "Val mAP@0.5:0.95",
            "val_metric/map_50": "Val mAP@0.5",
            "val_metric/map_75": "Val mAP@0.75",
            "val_metric/map_small": "Val mAP (small)",
            "val_metric/map_medium": "Val mAP (medium)",
            "val_metric/map_large": "Val mAP (large)",
        }
        for metric_name, expected_display in expected_mappings.items():
            assert metric_name in display_names
            assert display_names[metric_name] == expected_display

    def test_get_display_names__classwise(self) -> None:
        """Test that get_display_names works with classwise metrics."""
        metric_args = InstanceSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            split="val",
            class_names=["cat__type_a", "dog__breed__b", "bird"],
            log_classwise=True,
        )

        display_names = segmentation_task_metric.get_display_names()

        assert "val_metric/map" in display_names
        assert "val_metric_classwise/map" in display_names
        assert "val_metric_classwise/map_50" in display_names
        assert "val_metric_classwise/map_75" in display_names

        assert display_names["val_metric/map"] == "Val mAP@0.5:0.95"
        assert display_names["val_metric_classwise/map"] == "Val mAP@0.5:0.95"
        assert display_names["val_metric_classwise/map_50"] == "Val mAP@0.5"

    def test_reset(self) -> None:
        """Test that reset() clears all metrics."""
        metric_args = InstanceSegmentationTaskMetricArgs()
        segmentation_task_metric = metric_args.get_metrics(
            split="val",
            class_names=["cat", "dog", "bird"],
            log_classwise=True,
        )

        # First scenario: random masks
        torch.manual_seed(42)
        preds = [
            {
                "masks": torch.randint(0, 2, (1, 100, 100), dtype=torch.bool),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "masks": torch.randint(0, 2, (1, 100, 100), dtype=torch.bool),
                "labels": torch.tensor([0]),
            }
        ]
        segmentation_task_metric.update(preds, targets)

        result_before = segmentation_task_metric.compute()
        assert len(result_before.metrics) > 0

        segmentation_task_metric.reset()

        # Second scenario: different random masks
        torch.manual_seed(123)
        preds2 = [
            {
                "masks": torch.randint(0, 2, (1, 100, 100), dtype=torch.bool),
                "scores": torch.tensor([0.8]),
                "labels": torch.tensor([1]),
            }
        ]
        targets2 = [
            {
                "masks": torch.randint(0, 2, (1, 100, 100), dtype=torch.bool),
                "labels": torch.tensor([1]),
            }
        ]
        segmentation_task_metric.update(preds2, targets2)
        result_after = segmentation_task_metric.compute()

        # Results should be different
        assert result_after.metrics != result_before.metrics
