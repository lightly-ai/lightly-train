#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from torch.nn import ModuleDict

from lightly_train._metrics.task_metric import MetricComputeResult, TaskMetric


class MultiheadTaskMetric(TaskMetric):
    """Wrapper that manages metrics for multiple heads and selects the best head.

    This wrapper holds one TaskMetric instance per head. On compute(), it:
    1. Computes metrics for all heads
    2. Renames per-head keys: "val_metric/miou" -> "val_metric_head/miou_{head_name}"
    3. Selects the best head based on each head's watch_metric_value
    4. Promotes the best head's metrics to the top-level prefix (no head suffix)

    The watch_metric and watch_metric_value in the returned MetricComputeResult
    refer to the promoted best-head metric, making it compatible with the training
    loop's checkpointing logic.

    The watch_metric_mode is taken from the head's MetricComputeResult in compute().

    Usage:
        # Create per-head metrics
        head_metrics = {
            "lr0_001": seg_task_metric_args.get_metrics(split="val", ...),
            "lr0_01":  seg_task_metric_args.get_metrics(split="val", ...),
        }
        val_metrics = MultiheadTaskMetric(head_metrics=head_metrics)

        # Update per head during validation
        val_metrics.head_metrics["lr0_001"].update(preds, target)
        val_metrics.head_metrics["lr0_01"].update(preds, target)

        # Compute at end of validation epoch
        result = val_metrics.compute()
        # result.metrics contains:
        #   "val_metric_head/miou_lr0_001": 0.72
        #   "val_metric_head/miou_lr0_01":  0.75  (best)
        #   "val_metric/miou": 0.75              (best head promoted)
        # result.watch_metric == "val_metric/miou"
        # result.watch_metric_value == 0.75
    """

    def __init__(
        self,
        *,
        head_metrics: Mapping[str, TaskMetric],
    ) -> None:
        """Initialize MultiheadTaskMetric.

        Args:
            head_metrics: Mapping from head name to TaskMetric instance.
                Head names should sort alphabetically in order of increasing
                learning rate (e.g., "lr0_001", "lr0_003", "lr0_01").
        """
        task_metric = next(iter(head_metrics.values()))
        super().__init__(task_metric_args=task_metric.task_metric_args)
        self.head_metrics: ModuleDict = ModuleDict(head_metrics)  # type: ignore[arg-type]

    def compute(self) -> MetricComputeResult:
        """Compute metrics for all heads and promote the best head's metrics.

        Returns:
            MetricComputeResult where:
            - metrics contains per-head keys AND best-head top-level keys
            - watch_metric is the top-level key of the best metric
            - watch_metric_value is the value of the best metric
        """
        all_metrics: dict[str, float] = {}
        watch_metric_mode: Literal["min", "max"] | None = None
        best_value: float | None = None
        best_head_name = ""
        best_head_result: MetricComputeResult | None = None

        for head_name, head_metric in self.head_metrics.items():
            head_result = head_metric.compute()  # type: ignore[operator]

            # Rename keys and add to all_metrics
            for key, value in head_result.metrics.items():
                renamed_key = _rename_key_for_head(key, head_name)
                all_metrics[renamed_key] = value

            # Skip heads with no watch metric value or mode
            head_best = head_result.watch_metric_value
            if head_best is None or head_result.watch_metric_mode is None:
                continue
            if watch_metric_mode is None:
                watch_metric_mode = head_result.watch_metric_mode
            is_better = (
                best_value is None
                or (watch_metric_mode == "max" and head_best > best_value)
                or (watch_metric_mode == "min" and head_best < best_value)
            )
            if is_better:
                best_value = head_best
                best_head_name = head_name
                best_head_result = head_result

        # Promote best head's metrics to top-level (without head suffix)
        if best_head_result is not None:
            all_metrics.update(best_head_result.metrics)
            return MetricComputeResult(
                metrics=all_metrics,
                watch_metric=best_head_result.watch_metric,
                watch_metric_value=best_value,
                watch_metric_mode=watch_metric_mode,
                best_head_name=best_head_name,
                best_head_metrics=best_head_result.metrics,
            )

        # Fallback: no heads, or all heads have no matching watch metric
        return MetricComputeResult(
            metrics=all_metrics,
            watch_metric=None,
            watch_metric_value=None,
            watch_metric_mode=None,
            best_head_name=None,
            best_head_metrics=None,
        )


def _rename_key_for_head(key: str, head_name: str) -> str:
    """Rename a metric key to include the head name.

    For keys without a slash (e.g. "val_loss"), appends "_head/{head_name}":
        "val_loss" + "lr0_001" -> "val_loss_head/lr0_001"

    For keys with a slash, inserts "_head" before any "_classwise" suffix in
    the prefix part, then appends "_{head_name}" to the metric part:
        "val_metric/miou" + "lr0_001"            -> "val_metric_head/miou_lr0_001"
        "val_metric_classwise/iou_dog" + "lr0_001" -> "val_metric_head_classwise/iou_dog_lr0_001"
        "train_metric/f1_macro" + "lr0_001"      -> "train_metric_head/f1_macro_lr0_001"
        "val_loss/loss_vfl" + "lr0_001"          -> "val_loss_head/loss_vfl_lr0_001"
    """
    if "/" not in key:
        # No slash: "val_loss" -> "val_loss_head/lr0_001"
        return f"{key}_head/{head_name}"
    slash_idx = key.index("/")
    prefix_part = key[:slash_idx]
    metric_part = key[slash_idx + 1 :]
    # Insert "_head" before "_classwise" if present, otherwise append "_head".
    # e.g. "val_metric_classwise" -> "val_metric_head_classwise"
    # e.g. "val_metric" -> "val_metric_head"
    if "_classwise" in prefix_part:
        new_prefix = prefix_part.replace("_classwise", "_head_classwise", 1)
    else:
        new_prefix = f"{prefix_part}_head"
    return f"{new_prefix}/{metric_part}_{head_name}"
