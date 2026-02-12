#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import torch

from lightly_train._metrics.classification.task_metric import (
    MulticlassClassificationTaskMetricArgs,
    MultilabelClassificationTaskMetricArgs,
)


class TestMulticlassClassificationTaskMetricArgs:
    def test_get_metrics(self) -> None:
        """Test that MulticlassClassificationTaskMetricArgs can create metrics."""
        metric_args = MulticlassClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat", "dog", "bird"],
            log_classwise=False,
            classwise_metric_args=None,
        )

        # Verify metrics can be updated with dummy data
        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([0, 1])
        classification_task_metric.metrics.update(preds, target)

        # Verify metrics can be computed
        result = classification_task_metric.metrics.compute()
        assert isinstance(result, dict)
        assert len(result) > 0

        # Verify expected metric names are present
        expected_metrics = {
            "val_metric/top1_acc_micro",
            "val_metric/f1_macro",
            "val_metric/precision_macro",
            "val_metric/recall_macro",
        }
        assert expected_metrics.issubset(set(result.keys()))

    def test_get_metrics__classwise(self) -> None:
        """Test that classwise metrics are created correctly."""
        metric_args = MulticlassClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat", "dog", "bird"],
            log_classwise=True,
            classwise_metric_args=None,
        )

        # Verify classwise metrics exist
        assert classification_task_metric.metrics_classwise is not None

        # Verify metrics can be updated with dummy data
        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]])
        target = torch.tensor([0, 1, 2])
        classification_task_metric.metrics.update(preds, target)
        classification_task_metric.metrics_classwise.update(preds, target)

        # Verify regular metrics can be computed
        result = classification_task_metric.metrics.compute()
        assert isinstance(result, dict)
        assert len(result) > 0

        # Verify classwise metrics can be computed
        result_classwise = classification_task_metric.metrics_classwise.compute()
        assert isinstance(result_classwise, dict)
        assert len(result_classwise) > 0

        # Verify expected classwise metric names are present for each class
        for class_name in ["cat", "dog", "bird"]:
            expected_classwise_metrics = {
                f"val_metric_classwise/top1_acc_micro_{class_name}",
                f"val_metric_classwise/f1_macro_{class_name}",
                f"val_metric_classwise/precision_macro_{class_name}",
                f"val_metric_classwise/recall_macro_{class_name}",
            }
            assert expected_classwise_metrics.issubset(set(result_classwise.keys()))

    def test_get_metrics__classwise_with_double_underscores_in_class_names(
        self,
    ) -> None:
        """Test that classwise metrics handle class names with double underscores."""
        metric_args = MulticlassClassificationTaskMetricArgs()
        # Use class names that contain double underscores
        class_names = ["cat__type_a", "dog__breed__b", "bird"]
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=class_names,
            log_classwise=True,
            classwise_metric_args=None,
        )

        # Verify classwise metrics exist
        assert classification_task_metric.metrics_classwise is not None

        # Verify metrics can be updated with dummy data
        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]])
        target = torch.tensor([0, 1, 2])
        classification_task_metric.metrics.update(preds, target)
        classification_task_metric.metrics_classwise.update(preds, target)

        # Verify classwise metrics can be computed
        result_classwise = classification_task_metric.metrics_classwise.compute()
        assert isinstance(result_classwise, dict)
        assert len(result_classwise) > 0

        # Verify that double underscores in class names are preserved
        for class_name in class_names:
            expected_classwise_metrics = {
                f"val_metric_classwise/top1_acc_micro_{class_name}",
                f"val_metric_classwise/f1_macro_{class_name}",
                f"val_metric_classwise/precision_macro_{class_name}",
                f"val_metric_classwise/recall_macro_{class_name}",
            }
            assert expected_classwise_metrics.issubset(set(result_classwise.keys()))


class TestMultilabelClassificationTaskMetricArgs:
    def test_get_metrics(self) -> None:
        """Test that MultilabelClassificationTaskMetricArgs can create metrics."""
        metric_args = MultilabelClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat", "dog", "bird"],
            log_classwise=False,
            classwise_metric_args=None,
        )

        # Verify metrics can be updated with dummy data
        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([[1, 0, 0], [0, 1, 1]])
        classification_task_metric.metrics.update(preds, target)

        # Verify metrics can be computed
        result = classification_task_metric.metrics.compute()
        assert isinstance(result, dict)
        assert len(result) > 0

        # Verify expected metric names are present
        expected_metrics = {
            "val_metric/accuracy_micro",
            "val_metric/f1_macro",
            "val_metric/auroc_macro",
            "val_metric/avg_precision_macro",
            "val_metric/hamming_distance",
        }
        assert expected_metrics.issubset(set(result.keys()))

    def test_get_metrics__classwise(self) -> None:
        """Test that classwise metrics are created correctly."""
        metric_args = MultilabelClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat", "dog", "bird"],
            log_classwise=True,
            classwise_metric_args=None,
        )

        # Verify classwise metrics exist
        assert classification_task_metric.metrics_classwise is not None

        # Verify metrics can be updated with dummy data
        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]])
        target = torch.tensor([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
        classification_task_metric.metrics.update(preds, target)
        classification_task_metric.metrics_classwise.update(preds, target)

        # Verify regular metrics can be computed
        result = classification_task_metric.metrics.compute()
        assert isinstance(result, dict)
        assert len(result) > 0

        # Verify classwise metrics can be computed
        result_classwise = classification_task_metric.metrics_classwise.compute()
        assert isinstance(result_classwise, dict)
        assert len(result_classwise) > 0

        # Verify expected classwise metric names are present for each class
        for class_name in ["cat", "dog", "bird"]:
            expected_classwise_metrics = {
                f"val_metric_classwise/accuracy_micro_{class_name}",
                f"val_metric_classwise/f1_macro_{class_name}",
                f"val_metric_classwise/auroc_macro_{class_name}",
                f"val_metric_classwise/avg_precision_macro_{class_name}",
            }
            assert expected_classwise_metrics.issubset(set(result_classwise.keys()))

    def test_get_metrics__classwise_with_double_underscores_in_class_names(
        self,
    ) -> None:
        """Test that classwise metrics handle class names with double underscores."""
        metric_args = MultilabelClassificationTaskMetricArgs()
        # Use class names that contain double underscores
        class_names = ["cat__type_a", "dog__breed__b", "bird"]
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=class_names,
            log_classwise=True,
            classwise_metric_args=None,
        )

        # Verify classwise metrics exist
        assert classification_task_metric.metrics_classwise is not None

        # Verify metrics can be updated with dummy data
        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]])
        target = torch.tensor([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
        classification_task_metric.metrics.update(preds, target)
        classification_task_metric.metrics_classwise.update(preds, target)

        # Verify classwise metrics can be computed
        result_classwise = classification_task_metric.metrics_classwise.compute()
        assert isinstance(result_classwise, dict)
        assert len(result_classwise) > 0

        # Verify that double underscores in class names are preserved
        for class_name in class_names:
            expected_classwise_metrics = {
                f"val_metric_classwise/accuracy_micro_{class_name}",
                f"val_metric_classwise/f1_macro_{class_name}",
                f"val_metric_classwise/auroc_macro_{class_name}",
                f"val_metric_classwise/avg_precision_macro_{class_name}",
            }
            assert expected_classwise_metrics.issubset(set(result_classwise.keys()))
