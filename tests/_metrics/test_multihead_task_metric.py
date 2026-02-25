#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import torch

from lightly_train._metrics.multihead_task_metric import MultiheadTaskMetric
from lightly_train._metrics.semantic_segmentation.task_metric import (
    SemanticSegmentationTaskMetric,
    SemanticSegmentationTaskMetricArgs,
)


def _make_head_metric(split: str = "val") -> SemanticSegmentationTaskMetric:
    return SemanticSegmentationTaskMetric(
        task_metric_args=SemanticSegmentationTaskMetricArgs(),
        split=split,
        class_names=["cat", "dog", "bird"],
        ignore_index=None,
        log_classwise=False,
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
            best_metric_mode="max",
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
        assert result.best_metric_key == "val_metric/miou"
        assert result.best_head_name == "lr0_01"

    def test_reset(self) -> None:
        head_metrics = {"lr0_001": _make_head_metric()}
        wrapper = MultiheadTaskMetric(
            head_metrics=head_metrics,  # type: ignore[arg-type]
            best_metric_mode="max",
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
