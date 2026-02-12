#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import torch

from lightly_train._metrics.detection.task_metric import (
    ObjectDetectionTaskMetricArgs,
)


class TestObjectDetectionTaskMetricArgs:
    def test_get_metrics(self) -> None:
        """Test that ObjectDetectionTaskMetricArgs can create metrics."""
        metric_args = ObjectDetectionTaskMetricArgs()
        detection_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat", "dog", "bird"],
            log_classwise=False,
            classwise_metric_args=None,
        )

        # Detection format: list of dicts with "boxes", "scores", "labels"
        preds = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            }
        ]
        detection_task_metric.update(preds, targets)

        result = detection_task_metric.compute()
        assert isinstance(result, dict)
        assert len(result) > 0

        expected_metrics = {
            "val_metric/map",
            "val_metric/map_50",
            "val_metric/map_75",
            "val_metric/map_small",
            "val_metric/map_medium",
            "val_metric/map_large",
        }
        assert expected_metrics.issubset(set(result.keys()))

    def test_get_metrics__classwise(self) -> None:
        """Test that classwise metrics are created correctly."""
        metric_args = ObjectDetectionTaskMetricArgs()
        detection_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat__type_a", "dog__breed__b", "bird"],
            log_classwise=True,
            classwise_metric_args=None,
        )

        assert detection_task_metric.metrics_classwise is not None

        # Detection format with multiple predictions
        preds = [
            {
                "boxes": torch.tensor(
                    [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]
                ),
                "scores": torch.tensor([0.9, 0.8]),
                "labels": torch.tensor([0, 1]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor(
                    [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]
                ),
                "labels": torch.tensor([0, 1]),
            }
        ]
        detection_task_metric.update(preds, targets)

        result = detection_task_metric.compute()
        assert isinstance(result, dict)
        assert len(result) > 0

        regular_metrics = {k for k in result.keys() if "classwise" not in k}
        result_classwise = {k: v for k, v in result.items() if "classwise" in k}
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

        for class_name in ["cat__type_a", "dog__breed__b", "bird"]:
            expected_classwise_metrics = {
                f"val_metric_classwise/map_{class_name}",
            }
            # Only check if the class was actually in the predictions
            # (bird was not used, so won't have metrics)
            if class_name != "bird":
                assert expected_classwise_metrics.issubset(set(result_classwise.keys()))

    def test_get_display_names(self) -> None:
        """Test that get_display_names returns correct display names."""
        metric_args = ObjectDetectionTaskMetricArgs()
        detection_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat", "dog", "bird"],
            log_classwise=False,
            classwise_metric_args=None,
        )

        display_names = detection_task_metric.get_display_names()

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
        metric_args = ObjectDetectionTaskMetricArgs()
        detection_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat__type_a", "dog__breed__b", "bird"],
            log_classwise=True,
            classwise_metric_args=None,
        )

        display_names = detection_task_metric.get_display_names()

        assert "val_metric/map" in display_names
        assert "val_metric_classwise/map" in display_names
        assert "val_metric_classwise/map_50" in display_names
        assert "val_metric_classwise/map_75" in display_names

        assert display_names["val_metric/map"] == "Val mAP@0.5:0.95"
        assert display_names["val_metric_classwise/map"] == "Val mAP@0.5:0.95"
        assert display_names["val_metric_classwise/map_50"] == "Val mAP@0.5"

    def test_reset(self) -> None:
        """Test that reset() clears all metrics."""
        metric_args = ObjectDetectionTaskMetricArgs()
        detection_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat", "dog", "bird"],
            log_classwise=True,
            classwise_metric_args=None,
        )

        # First scenario: perfect prediction
        preds = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([0]),
            }
        ]
        detection_task_metric.update(preds, targets)

        result_before = detection_task_metric.compute()
        assert len(result_before) > 0

        detection_task_metric.reset()

        # Second scenario: imperfect prediction (wrong location)
        preds2 = [
            {
                "boxes": torch.tensor([[50.0, 60.0, 70.0, 80.0]]),
                "scores": torch.tensor([0.8]),
                "labels": torch.tensor([1]),
            }
        ]
        targets2 = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "labels": torch.tensor([1]),
            }
        ]
        detection_task_metric.update(preds2, targets2)
        result_after = detection_task_metric.compute()

        # Results should be different (perfect vs. imperfect detection)
        assert result_after != result_before
