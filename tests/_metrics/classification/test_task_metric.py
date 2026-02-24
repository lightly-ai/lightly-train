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

        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([0, 1])
        classification_task_metric.update(preds, target)

        result = classification_task_metric.compute()
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

        expected_metrics = {
            "val_metric/top1_acc_micro",
            "val_metric/f1_macro",
            "val_metric/precision_macro",
            "val_metric/recall_macro",
        }
        assert expected_metrics.issubset(set(result.metrics.keys()))
        assert result.best_metric_key == "val_metric/top1_acc_micro"
        assert isinstance(result.best_metric_value, float)

    def test_get_metrics__classwise(self) -> None:
        """Test that classwise metrics are created correctly."""
        metric_args = MulticlassClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat__type_a", "dog__breed__b", "bird"],
            log_classwise=True,
            classwise_metric_args=None,
        )

        assert classification_task_metric.metrics_classwise is not None

        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]])
        target = torch.tensor([0, 1, 2])
        classification_task_metric.update(preds, target)

        result = classification_task_metric.compute()
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

        regular_metrics = {k for k in result.metrics.keys() if "classwise" not in k}
        result_classwise = {k: v for k, v in result.metrics.items() if "classwise" in k}
        assert isinstance(result_classwise, dict)
        assert len(result_classwise) > 0

        assert len(regular_metrics) > 0
        expected_regular_metrics = {
            "val_metric/top1_acc_micro",
            "val_metric/f1_macro",
            "val_metric/precision_macro",
            "val_metric/recall_macro",
        }
        assert expected_regular_metrics.issubset(set(regular_metrics))

        for class_name in ["cat__type_a", "dog__breed__b", "bird"]:
            expected_classwise_metrics = {
                f"val_metric_classwise/top1_acc_micro_{class_name}",
                f"val_metric_classwise/f1_macro_{class_name}",
                f"val_metric_classwise/precision_macro_{class_name}",
                f"val_metric_classwise/recall_macro_{class_name}",
            }
            assert expected_classwise_metrics.issubset(set(result_classwise.keys()))

    def test_get_display_names(self) -> None:
        """Test that get_display_names returns correct display names."""
        metric_args = MulticlassClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat", "dog", "bird"],
            log_classwise=False,
            classwise_metric_args=None,
        )

        display_names = classification_task_metric.get_display_names()

        assert isinstance(display_names, dict)
        assert len(display_names) > 0

        expected_mappings = {
            "val_metric/top1_acc_micro": "Val Top-1 Acc (Micro)",
            "val_metric/f1_macro": "Val F1 (Macro)",
            "val_metric/precision_macro": "Val Precision (Macro)",
            "val_metric/recall_macro": "Val Recall (Macro)",
        }
        for metric_name, expected_display in expected_mappings.items():
            assert metric_name in display_names
            assert display_names[metric_name] == expected_display

    def test_get_display_names__classwise(self) -> None:
        """Test that get_display_names works with classwise metrics."""
        metric_args = MulticlassClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat__type_a", "dog__breed__b", "bird"],
            log_classwise=True,
            classwise_metric_args=None,
        )

        display_names = classification_task_metric.get_display_names()

        assert "val_metric/top1_acc_micro" in display_names
        assert "val_metric_classwise/top1_acc_micro" in display_names
        assert "val_metric_classwise/f1_macro" in display_names
        assert "val_metric_classwise/precision_macro" in display_names
        assert "val_metric_classwise/recall_macro" in display_names

        assert (
            display_names["val_metric_classwise/top1_acc_micro"]
            == "Val Top-1 Acc (Micro)"
        )
        assert display_names["val_metric_classwise/f1_macro"] == "Val F1 (Macro)"

    def test_reset(self) -> None:
        """Test that reset() clears all metrics."""
        metric_args = MulticlassClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat", "dog", "bird"],
            log_classwise=True,
            classwise_metric_args=None,
        )

        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([0, 1])
        classification_task_metric.update(preds, target)

        result_before = classification_task_metric.compute()
        assert len(result_before.metrics) > 0

        classification_task_metric.reset()

        preds2 = torch.tensor([[0.1, 0.9, 0.0], [0.0, 0.1, 0.9]])
        target2 = torch.tensor([1, 2])
        classification_task_metric.update(preds2, target2)
        result_after = classification_task_metric.compute()

        assert result_after.metrics != result_before.metrics


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

        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([[1, 0, 0], [0, 1, 1]])
        classification_task_metric.update(preds, target)

        result = classification_task_metric.compute()
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

        expected_metrics = {
            "val_metric/accuracy_micro",
            "val_metric/f1_macro",
            "val_metric/auroc_macro",
            "val_metric/avg_precision_macro",
            "val_metric/hamming_distance",
        }
        assert expected_metrics.issubset(set(result.metrics.keys()))
        assert result.best_metric_key == "val_metric/f1_macro"
        assert isinstance(result.best_metric_value, float)

    def test_get_metrics__classwise(self) -> None:
        """Test that classwise metrics are created correctly."""
        metric_args = MultilabelClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat__type_a", "dog__breed__b", "bird"],
            log_classwise=True,
            classwise_metric_args=None,
        )

        assert classification_task_metric.metrics_classwise is not None

        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.1, 0.8]])
        target = torch.tensor([[1, 0, 0], [0, 1, 1], [1, 0, 1]])
        classification_task_metric.update(preds, target)

        result = classification_task_metric.compute()
        assert isinstance(result.metrics, dict)
        assert len(result.metrics) > 0

        regular_metrics = {k for k in result.metrics.keys() if "classwise" not in k}
        result_classwise = {k: v for k, v in result.metrics.items() if "classwise" in k}
        assert isinstance(result_classwise, dict)
        assert len(result_classwise) > 0

        assert len(regular_metrics) > 0
        expected_regular_metrics = {
            "val_metric/accuracy_micro",
            "val_metric/f1_macro",
            "val_metric/auroc_macro",
            "val_metric/avg_precision_macro",
            "val_metric/hamming_distance",
        }
        assert expected_regular_metrics.issubset(set(regular_metrics))

        for class_name in ["cat__type_a", "dog__breed__b", "bird"]:
            expected_classwise_metrics = {
                f"val_metric_classwise/accuracy_micro_{class_name}",
                f"val_metric_classwise/f1_macro_{class_name}",
                f"val_metric_classwise/auroc_macro_{class_name}",
                f"val_metric_classwise/avg_precision_macro_{class_name}",
            }
            assert expected_classwise_metrics.issubset(set(result_classwise.keys()))

    def test_get_display_names(self) -> None:
        """Test that get_display_names returns correct display names."""
        metric_args = MultilabelClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat", "dog", "bird"],
            log_classwise=False,
            classwise_metric_args=None,
        )

        display_names = classification_task_metric.get_display_names()

        assert isinstance(display_names, dict)
        assert len(display_names) > 0

        expected_mappings = {
            "val_metric/accuracy_micro": "Val Accuracy (Micro)",
            "val_metric/f1_macro": "Val F1 (Macro)",
            "val_metric/auroc_macro": "Val AUROC (Macro)",
            "val_metric/avg_precision_macro": "Val Avg Precision (Macro)",
            "val_metric/hamming_distance": "Val Hamming Distance",
        }
        for metric_name, expected_display in expected_mappings.items():
            assert metric_name in display_names
            assert display_names[metric_name] == expected_display

    def test_get_display_names__classwise(self) -> None:
        """Test that get_display_names works with classwise metrics."""
        metric_args = MultilabelClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat__type_a", "dog__breed__b", "bird"],
            log_classwise=True,
            classwise_metric_args=None,
        )

        display_names = classification_task_metric.get_display_names()

        assert "val_metric/accuracy_micro" in display_names
        assert "val_metric_classwise/accuracy_micro" in display_names
        assert "val_metric_classwise/f1_macro" in display_names
        assert "val_metric_classwise/auroc_macro" in display_names
        assert "val_metric_classwise/avg_precision_macro" in display_names

        assert (
            display_names["val_metric_classwise/accuracy_micro"]
            == "Val Accuracy (Micro)"
        )
        assert display_names["val_metric_classwise/f1_macro"] == "Val F1 (Macro)"

    def test_reset(self) -> None:
        """Test that reset() clears all metrics."""
        metric_args = MultilabelClassificationTaskMetricArgs()
        classification_task_metric = metric_args.get_metrics(
            prefix="val_metric/",
            class_names=["cat", "dog", "bird"],
            log_classwise=True,
            classwise_metric_args=None,
        )

        preds = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]])
        target = torch.tensor([[1, 0, 0], [0, 1, 1]])
        classification_task_metric.update(preds, target)

        result_before = classification_task_metric.compute()
        assert len(result_before.metrics) > 0

        classification_task_metric.reset()

        preds2 = torch.tensor([[0.1, 0.9, 0.0], [0.0, 0.1, 0.9]])
        target2 = torch.tensor([[0, 1, 0], [1, 0, 1]])
        classification_task_metric.update(preds2, target2)
        result_after = classification_task_metric.compute()

        assert result_after.metrics != result_before.metrics
