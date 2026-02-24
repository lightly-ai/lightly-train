#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Literal

import torch

from lightly_train._metrics.task_metric import MetricComputeResult, TaskMetric


class MultiheadTaskMetric(TaskMetric):
    """Wrapper that manages metrics for multiple heads and selects the best head.

    This wrapper holds one TaskMetric instance per head. On compute(), it:
    1. Computes metrics for all heads
    2. Renames per-head keys: "val_metric/miou" -> "val_metric_head/miou_{head_name}"
    3. Selects the best head based on each head's best_metric_value
    4. Promotes the best head's metrics to the top-level prefix (no head suffix)

    The best_metric_key and best_metric_value in the returned MetricComputeResult
    refer to the promoted best-head metric, making it compatible with the training
    loop's checkpointing logic.

    Usage:
        # Create per-head metrics
        head_metrics = {
            "lr0_001": seg_task_metric_args.get_metrics(prefix="val_metric/", ...),
            "lr0_01":  seg_task_metric_args.get_metrics(prefix="val_metric/", ...),
        }
        val_metrics = MultiheadTaskMetric(
            head_metrics=head_metrics,
            best_metric_mode="max",
        )

        # Update per head during validation
        val_metrics.head_metrics["lr0_001"].update(preds, target)
        val_metrics.head_metrics["lr0_01"].update(preds, target)

        # Compute at end of validation epoch
        result = val_metrics.compute()
        # result.metrics contains:
        #   "val_metric_head/miou_lr0_001": 0.72
        #   "val_metric_head/miou_lr0_01":  0.75  (best)
        #   "val_metric/miou": 0.75              (best head promoted)
        # result.best_metric_key == "val_metric/miou"
        # result.best_metric_value == 0.75
    """

    def __init__(
        self,
        *,
        head_metrics: dict[str, TaskMetric],
        best_metric_mode: Literal["min", "max"],
    ) -> None:
        """Initialize MultiheadTaskMetric.

        Args:
            head_metrics: Mapping from head name to TaskMetric instance.
                Head names should sort alphabetically in order of increasing
                learning rate (e.g., "lr0_001", "lr0_003", "lr0_01").
            best_metric_mode: Whether to maximize ("max") or minimize ("min")
                the best_metric_value when selecting the best head.
        """
        super().__init__()
        self.best_metric_mode = best_metric_mode
        # Store as ModuleDict so nn.Module registers them as submodules
        # (enables device transfer, state_dict, etc.)
        self.head_metrics: torch.nn.ModuleDict = torch.nn.ModuleDict(
            head_metrics  # type: ignore[arg-type]
        )

    def compute(self) -> MetricComputeResult:
        """Compute metrics for all heads and promote the best head's metrics.

        Returns:
            MetricComputeResult where:
            - metrics contains per-head keys AND best-head top-level keys
            - best_metric_key is the top-level key of the best metric
            - best_metric_value is the value of the best metric
        """
        all_metrics: dict[str, float] = {}
        best_value = -float("inf") if self.best_metric_mode == "max" else float("inf")
        best_head_name = ""
        best_head_result: MetricComputeResult | None = None

        for head_name, head_metric in self.head_metrics.items():
            head_result = head_metric.compute()  # type: ignore[operator]

            # Rename keys and add to all_metrics
            for key, value in head_result.metrics.items():
                renamed_key = _rename_key_for_head(key, head_name)
                all_metrics[renamed_key] = value

            # Check if this head is best
            head_best = head_result.best_metric_value
            is_better = (self.best_metric_mode == "max" and head_best > best_value) or (
                self.best_metric_mode == "min" and head_best < best_value
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
                best_metric_key=best_head_result.best_metric_key,
                best_metric_value=best_value,
                best_head_name=best_head_name,
                best_head_metrics=best_head_result.metrics,
            )

        # Fallback: no heads (should not happen in practice)
        return MetricComputeResult(
            metrics=all_metrics,
            best_metric_key="",
            best_metric_value=best_value,
            best_head_name="",
            best_head_metrics={},
        )

    def reset(self) -> None:
        """Reset all head metrics."""
        for head_metric in self.head_metrics.values():
            head_metric.reset()  # type: ignore[operator]

    def get_display_names(self) -> dict[str, str]:
        """Get display names for all metrics.

        Returns per-head display names (renamed with head suffix) plus
        the top-level display names from the first head (representing
        the best-head promoted metrics).
        """
        display_names: dict[str, str] = {}

        for head_name, head_metric in self.head_metrics.items():
            for key, display in head_metric.get_display_names().items():  # type: ignore[operator]
                renamed_key = _rename_key_for_head(key, head_name)
                display_names[renamed_key] = display

        # Add top-level display names from any head (all heads share the same metric structure)
        if self.head_metrics:
            first_head = next(iter(self.head_metrics.values()))
            display_names.update(first_head.get_display_names())  # type: ignore[operator]

        return display_names


def _rename_key_for_head(key: str, head_name: str) -> str:
    """Rename a metric key to include the head name.

    Transforms "val_metric/miou" -> "val_metric_head/miou_lr0_001".
    The prefix part before "/" gets "_head" appended (replacing "_metric" with
    "_metric_head"), and the metric name gets "_{head_name}" appended.

    Examples:
        "val_metric/miou" + "lr0_001"       -> "val_metric_head/miou_lr0_001"
        "val_metric_classwise/iou_0" + "lr0_001" -> "val_metric_head_classwise/iou_0_lr0_001"
        "train_metric/f1_macro" + "lr0_001" -> "train_metric_head/f1_macro_lr0_001"
    """
    slash_idx = key.index("/")
    prefix_part = key[:slash_idx]
    metric_part = key[slash_idx + 1 :]
    head_prefix = prefix_part.replace("_metric", "_metric_head", 1)
    return f"{head_prefix}/{metric_part}_{head_name}"
