#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from torchmetrics import MeanMetric

from lightly_train._commands.train_task_helpers import compute_metrics, reset_metrics
from lightly_train._metrics.semantic_segmentation.task_metric import (
    SemanticSegmentationTaskMetric,
    SemanticSegmentationTaskMetricArgs,
)


def _create_task_metric(
    split: str = "val",
) -> SemanticSegmentationTaskMetric:
    """Create a SemanticSegmentationTaskMetric for testing."""
    return SemanticSegmentationTaskMetricArgs().get_metrics(
        split=split,
        num_classes=3,
        ignore_index=None,
        log_classwise=False,
    )


def test_compute_metrics__task_metric() -> None:
    """TaskMetric in log_dict should have compute() called and metrics unpacked."""
    task_metric = _create_task_metric()

    preds = torch.zeros(2, 10, 10, dtype=torch.long)
    targets = torch.zeros(2, 10, 10, dtype=torch.long)
    task_metric.update(preds, targets)

    result = compute_metrics({"val_metrics": task_metric})

    # The TaskMetric key itself should not appear in result.
    assert "val_metrics" not in result
    # The unpacked metric key should appear.
    assert "val_metric/miou" in result
    assert isinstance(result["val_metric/miou"], float)


def test_compute_metrics__task_metric_with_plain_float() -> None:
    """TaskMetric and plain float can coexist in log_dict."""
    task_metric = _create_task_metric()

    preds = torch.zeros(2, 10, 10, dtype=torch.long)
    targets = torch.zeros(2, 10, 10, dtype=torch.long)
    task_metric.update(preds, targets)

    result = compute_metrics(
        {
            "val_metrics": task_metric,
            "val_loss": 0.42,
        }
    )

    assert "val_metric/miou" in result
    assert result["val_loss"] == 0.42


def test_compute_metrics__task_metric_with_torchmetrics_metric() -> None:
    """TaskMetric and torchmetrics Metric can coexist in log_dict."""
    task_metric = _create_task_metric()
    mean_metric = MeanMetric()

    preds = torch.zeros(2, 10, 10, dtype=torch.long)
    targets = torch.zeros(2, 10, 10, dtype=torch.long)
    task_metric.update(preds, targets)
    mean_metric.update(torch.tensor(0.5))

    result = compute_metrics(
        {
            "val_metrics": task_metric,
            "val_loss": mean_metric,
        }
    )

    assert "val_metric/miou" in result
    assert "val_loss" in result
    assert abs(result["val_loss"] - 0.5) < 1e-6


def test_compute_metrics__multiple_task_metrics() -> None:
    """Multiple TaskMetric entries in log_dict are all unpacked."""
    val_metric = _create_task_metric(split="val")
    train_metric = _create_task_metric(split="train")

    preds = torch.zeros(2, 10, 10, dtype=torch.long)
    targets = torch.zeros(2, 10, 10, dtype=torch.long)
    val_metric.update(preds, targets)
    train_metric.update(preds, targets)

    result = compute_metrics(
        {
            "val_metrics": val_metric,
            "train_metrics": train_metric,
        }
    )

    assert "val_metric/miou" in result
    assert "train_metric/miou" in result


def test_reset_metrics__task_metric() -> None:
    """reset_metrics should call reset() on TaskMetric objects."""
    task_metric = _create_task_metric()

    preds = torch.zeros(2, 10, 10, dtype=torch.long)
    targets = torch.zeros(2, 10, 10, dtype=torch.long)
    task_metric.update(preds, targets)

    # Compute before reset so we have a reference value.
    result_before = task_metric.compute()
    assert "val_metric/miou" in result_before.metrics

    reset_metrics({"val_metrics": task_metric})

    # After reset, update with different data to confirm state was cleared.
    preds2 = torch.zeros(2, 10, 10, dtype=torch.long)
    targets2 = torch.ones(2, 10, 10, dtype=torch.long)  # All wrong predictions.
    task_metric.update(preds2, targets2)
    result_after = task_metric.compute()

    # miou should differ since the metric was reset and new data was fed.
    assert (
        result_after.metrics["val_metric/miou"]
        != result_before.metrics["val_metric/miou"]
    )


def test_reset_metrics__task_metric_and_torchmetrics_metric() -> None:
    """reset_metrics handles TaskMetric and torchmetrics Metric together."""
    task_metric = _create_task_metric()
    mean_metric = MeanMetric()

    preds = torch.zeros(2, 10, 10, dtype=torch.long)
    targets = torch.zeros(2, 10, 10, dtype=torch.long)
    task_metric.update(preds, targets)
    mean_metric.update(torch.tensor(1.0))

    log_dict = {"val_metrics": task_metric, "val_loss": mean_metric}
    reset_metrics(log_dict)

    # After reset, update with new values and verify only the new data is used.
    mean_metric.update(torch.tensor(0.25))
    assert abs(mean_metric.compute().item() - 0.25) < 1e-6
