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

from lightly_train._metrics.multihead_task_metric import (
    MultiheadTaskMetric,
    _rename_key_for_head,
)
from lightly_train._metrics.semantic_segmentation.task_metric import (
    SemanticSegmentationTaskMetric,
    SemanticSegmentationTaskMetricArgs,
)


def _make_head_metric(split: str = "val") -> SemanticSegmentationTaskMetric:
    return SemanticSegmentationTaskMetric(
        task_metric_args=SemanticSegmentationTaskMetricArgs(classwise=False),
        split=split,
        class_names=["cat", "dog", "bird"],
        ignore_index=None,
        loss_names=["loss"],
    )


class TestMultiheadTaskMetric:
    def test_compute(self) -> None:
        """Best head is promoted to top-level prefix."""
        head_metrics = {
            "lr0_001": _make_head_metric(),
            "lr0_01": _make_head_metric(),
        }
        wrapper = MultiheadTaskMetric(
            head_metrics=head_metrics,  # type: ignore[arg-type]
        )
        wrapper.head_metrics["lr0_001"].update(  # type: ignore[operator]
            torch.zeros(2, 10, 10, dtype=torch.long),
            torch.randint(0, 3, (2, 10, 10)),
        )
        wrapper.head_metrics["lr0_01"].update(  # type: ignore[operator]
            torch.zeros(2, 10, 10, dtype=torch.long),
            torch.zeros(2, 10, 10, dtype=torch.long),
        )
        result = wrapper.compute()
        assert "val_metric_head/miou_lr0_001" in result.metrics
        assert "val_metric_head/miou_lr0_01" in result.metrics
        assert result.watch_metric == "val_metric/miou"
        assert result.best_head_name == "lr0_01"

    def test_reset(self) -> None:
        head_metrics = {"lr0_001": _make_head_metric()}
        wrapper = MultiheadTaskMetric(
            head_metrics=head_metrics,  # type: ignore[arg-type]
        )
        torch.manual_seed(0)
        preds = torch.randint(0, 3, (2, 10, 10))
        target = torch.randint(0, 3, (2, 10, 10))
        wrapper.head_metrics["lr0_001"].update(preds, target)  # type: ignore[operator]
        result_before = wrapper.compute()
        wrapper.reset()
        torch.manual_seed(1)
        preds2 = torch.randint(0, 3, (2, 10, 10))
        target2 = torch.randint(0, 3, (2, 10, 10))
        wrapper.head_metrics["lr0_001"].update(preds2, target2)  # type: ignore[operator]
        result_after = wrapper.compute()
        assert result_after.metrics != result_before.metrics


@pytest.mark.parametrize(
    "key,head_name,expected",
    [
        # No slash: append "_head/{head_name}"
        ("val_loss", "lr0_001", "val_loss_head/lr0_001"),
        # With slash, no _classwise: insert "_head" in prefix, append "_{head_name}" to metric
        ("val_metric/miou", "lr0_001", "val_metric_head/miou_lr0_001"),
        ("train_metric/f1_macro", "lr0_001", "train_metric_head/f1_macro_lr0_001"),
        ("val_loss/loss_vfl", "lr0_001", "val_loss_head/loss_vfl_lr0_001"),
        # With slash and _classwise: insert "_head" before "_classwise"
        (
            "val_metric_classwise/iou_dog",
            "lr0_001",
            "val_metric_head_classwise/iou_dog_lr0_001",
        ),
    ],
)
def test__rename_key_for_head(key: str, head_name: str, expected: str) -> None:
    assert _rename_key_for_head(key, head_name) == expected
