#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import torch

from lightly_train._metrics.multihead_task_metric import (
    MultiheadTaskMetric,
    _rename_key_for_head,
)
from lightly_train._metrics.semantic_segmentation.task_metric import (
    SemanticSegmentationTaskMetric,
    SemanticSegmentationTaskMetricArgs,
)


def _create_head_metric(prefix: str = "val_metric/") -> SemanticSegmentationTaskMetric:
    """Helper to create a SemanticSegmentationTaskMetric for testing."""
    return SemanticSegmentationTaskMetricArgs().get_metrics(
        prefix=prefix,
        num_classes=3,
        ignore_index=255,
        log_classwise=False,
    )


def test_rename_key_for_head__val_metric() -> None:
    assert (
        _rename_key_for_head("val_metric/miou", "lr0_001")
        == "val_metric_head/miou_lr0_001"
    )


def test_rename_key_for_head__val_metric_classwise() -> None:
    assert (
        _rename_key_for_head("val_metric_classwise/iou_0", "lr0_001")
        == "val_metric_head_classwise/iou_0_lr0_001"
    )


def test_rename_key_for_head__train_metric() -> None:
    assert (
        _rename_key_for_head("train_metric/f1_macro", "lr0_001")
        == "train_metric_head/f1_macro_lr0_001"
    )


class TestMultiheadTaskMetric:
    def test_compute__selects_best_head(self) -> None:
        """Best head should be promoted to top-level prefix."""
        head_metrics = {
            "lr0_001": _create_head_metric(),
            "lr0_01": _create_head_metric(),
        }
        wrapper = MultiheadTaskMetric(
            head_metrics=head_metrics,  # type: ignore[arg-type]
            best_metric_mode="max",
        )

        # lr0_001 gets worse predictions, lr0_01 gets perfect predictions
        preds_bad = torch.zeros(2, 100, 100, dtype=torch.long)  # all class 0
        targets_varied = torch.randint(0, 3, (2, 100, 100))  # mixed classes

        preds_good = torch.zeros(2, 100, 100, dtype=torch.long)  # all class 0
        targets_same = torch.zeros(
            2, 100, 100, dtype=torch.long
        )  # all class 0 (perfect)

        wrapper.head_metrics["lr0_001"].update(preds_bad, targets_varied)  # type: ignore[operator]
        wrapper.head_metrics["lr0_01"].update(preds_good, targets_same)  # type: ignore[operator]

        result = wrapper.compute()

        # Per-head keys must exist
        assert "val_metric_head/miou_lr0_001" in result.metrics
        assert "val_metric_head/miou_lr0_01" in result.metrics

        # Top-level key must exist (best head promoted)
        assert "val_metric/miou" in result.metrics

        # Best head value promoted correctly
        assert result.best_metric_key == "val_metric/miou"
        assert result.best_metric_value == result.metrics["val_metric/miou"]

        # Top-level value equals the best head's per-head value
        assert (
            result.metrics["val_metric/miou"]
            == result.metrics["val_metric_head/miou_lr0_01"]
        )

        # Best head tracking
        assert result.best_head_name == "lr0_01"
        assert result.best_head_metrics == {"val_metric/miou": result.metrics["val_metric/miou"]}

    def test_compute__min_mode(self) -> None:
        """With mode='min', the head with lowest value should win."""
        head_metrics = {
            "lr0_001": _create_head_metric(),
            "lr0_01": _create_head_metric(),
        }
        wrapper = MultiheadTaskMetric(
            head_metrics=head_metrics,  # type: ignore[arg-type]
            best_metric_mode="min",
        )

        # lr0_001: perfect (high miou), lr0_01: bad (low miou)
        preds_perfect = torch.zeros(2, 100, 100, dtype=torch.long)
        targets_perfect = torch.zeros(2, 100, 100, dtype=torch.long)
        preds_bad = torch.zeros(2, 100, 100, dtype=torch.long)
        targets_bad = torch.randint(0, 3, (2, 100, 100))

        wrapper.head_metrics["lr0_001"].update(preds_perfect, targets_perfect)  # type: ignore[operator]
        wrapper.head_metrics["lr0_01"].update(preds_bad, targets_bad)  # type: ignore[operator]

        result = wrapper.compute()

        # With min mode, lr0_01 (lowest miou) should be promoted
        assert (
            result.best_metric_value <= result.metrics["val_metric_head/miou_lr0_001"]
        )
        assert (
            result.metrics["val_metric/miou"]
            == result.metrics["val_metric_head/miou_lr0_01"]
        )
        assert result.best_head_name == "lr0_01"
        assert result.best_head_metrics == {"val_metric/miou": result.metrics["val_metric/miou"]}

    def test_reset(self) -> None:
        """Reset should clear all head metrics."""
        head_metrics = {
            "lr0_001": _create_head_metric(),
        }
        wrapper = MultiheadTaskMetric(
            head_metrics=head_metrics,  # type: ignore[arg-type]
            best_metric_mode="max",
        )

        preds = torch.randint(0, 3, (2, 100, 100))
        targets = torch.randint(0, 3, (2, 100, 100))
        wrapper.head_metrics["lr0_001"].update(preds, targets)  # type: ignore[operator]
        result_before = wrapper.compute()

        wrapper.reset()

        torch.manual_seed(999)
        preds2 = torch.randint(0, 3, (2, 100, 100))
        targets2 = torch.randint(0, 3, (2, 100, 100))
        wrapper.head_metrics["lr0_001"].update(preds2, targets2)  # type: ignore[operator]
        result_after = wrapper.compute()

        # After reset, metrics should be independent from before
        assert result_after.metrics != result_before.metrics

    def test_get_display_names(self) -> None:
        """Display names should include per-head and top-level names."""
        head_metrics = {
            "lr0_001": _create_head_metric(),
            "lr0_01": _create_head_metric(),
        }
        wrapper = MultiheadTaskMetric(
            head_metrics=head_metrics,  # type: ignore[arg-type]
            best_metric_mode="max",
        )

        display_names = wrapper.get_display_names()

        assert isinstance(display_names, dict)
        # Per-head display names
        assert "val_metric_head/miou_lr0_001" in display_names
        assert "val_metric_head/miou_lr0_01" in display_names
        # Top-level display names (from best head promotion)
        assert "val_metric/miou" in display_names

    def test_head_order_alphabetical(self) -> None:
        """Head names like lr0_001, lr0_01 should sort in LR order alphabetically."""
        head_names = ["lr0_1", "lr0_001", "lr0_01", "lr0_0001", "lr0_03"]
        sorted_names = sorted(head_names)
        # Should be in increasing LR order
        assert sorted_names == ["lr0_0001", "lr0_001", "lr0_01", "lr0_03", "lr0_1"]
