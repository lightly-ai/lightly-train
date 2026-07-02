#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor
from torchmetrics import Metric as TorchmetricsMetric

from lightly_train._metrics.loss_metric_collection import LossMetricCollection
from lightly_train._metrics.task_metric import (
    AggregatedMetricValues,
    TaskMetric,
    TaskMetricArgs,
)

# Standard relative-error depth metrics and whether they should be minimised or
# maximised for best-model selection.
_METRIC_MODES: dict[str, Literal["min", "max"]] = {
    "abs_rel": "min",
    "rmse": "min",
    "delta1": "max",
}


class DepthEstimationTaskMetricArgs(TaskMetricArgs):
    """Metrics configuration for depth estimation."""

    watch_metric: str = "val_metric/rmse"
    # Whether to also compute quality metrics (not just losses) on the train split.
    train: bool = False


class DepthEstimationTaskMetric(TaskMetric):
    """Stores depth metrics and losses for one split.

    Quality metrics are AbsRel, RMSE and ``delta1`` (the fraction of pixels with
    ``max(pred/target, target/pred) < 1.25``), all computed over valid pixels
    (``target > 0``).
    """

    def __init__(
        self,
        *,
        task_metric_args: DepthEstimationTaskMetricArgs,
        split: str,
        loss_names: Sequence[str],
        train_loss_running_mean_window: int | None = None,
        init_metrics: bool | None = None,
    ) -> None:
        """
        Args:
            task_metric_args: Metrics configuration.
            split: Split name (e.g. "train", "val").
            loss_names: Names of losses to track.
            train_loss_running_mean_window:
                Window size for the running mean of training losses. Required when
                ``split == "train"``, ignored otherwise.
            init_metrics:
                Whether to initialize quality metrics. If None, uses
                ``task_metric_args.train`` for the train split and True otherwise.
        """
        super().__init__(task_metric_args=task_metric_args)
        self.split = split
        self.watch_metric = task_metric_args.watch_metric
        self.watch_metric_mode = _get_watch_metric_mode(
            watch_metric_name=task_metric_args.watch_metric,
            loss_names=list(loss_names),
        )

        if init_metrics is None:
            init_metrics = task_metric_args.train if split == "train" else True
        self._init_metrics = init_metrics
        self.metrics = _DepthMetrics(prefix=f"{split}_metric/")

        self.loss_metrics = LossMetricCollection(
            split=split,
            loss_names=loss_names,
            train_loss_running_mean_window=train_loss_running_mean_window,
        )

    def update_with_predictions(self, preds: Tensor, target: Tensor) -> None:
        """Update quality metrics with a batch of predictions and targets.

        Args:
            preds: Predicted depth of shape ``(B, 1, H, W)``.
            target: Target depth of shape ``(B, 1, H, W)``; pixels with ``target <= 0``
                are treated as invalid and ignored.
        """
        if self._init_metrics:
            self.metrics.update(preds, target)

    def update_with_losses(self, loss_dict: Mapping[str, Tensor], weight: int) -> None:
        """Accumulate loss values (see ``LossMetricCollection``)."""
        self.loss_metrics.update(loss_dict, weight=weight)

    def compute_aggregated_values(self) -> AggregatedMetricValues:
        result = self.loss_metrics.compute()
        if self._init_metrics:
            result.update(
                {name: float(value) for name, value in self.metrics.compute().items()}
            )
        best_val = result.get(self.watch_metric)
        return AggregatedMetricValues(
            metric_values=result,
            watch_metric=self.watch_metric if best_val is not None else None,
            watch_metric_value=float(best_val) if best_val is not None else None,
            watch_metric_mode=self.watch_metric_mode if best_val is not None else None,
            best_head_name=None,
            best_head_metric_values=None,
        )


class _DepthMetrics(TorchmetricsMetric):
    """Accumulates AbsRel, RMSE and ``delta1`` over valid pixels across batches."""

    abs_rel_sum: Tensor
    se_sum: Tensor
    delta1_sum: Tensor
    count: Tensor

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.add_state("abs_rel_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("se_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta1_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        valid = (target > 0) & torch.isfinite(preds) & torch.isfinite(target)
        if not bool(valid.any()):
            return
        pred = preds[valid].float()
        gt = target[valid].float()

        self.abs_rel_sum += torch.sum(torch.abs(pred - gt) / gt)
        self.se_sum += torch.sum((pred - gt) ** 2)
        ratio = torch.maximum(pred / gt, gt / pred)
        self.delta1_sum += torch.sum((ratio < 1.25).float())
        self.count += gt.numel()

    def compute(self) -> dict[str, Tensor]:
        if bool(self.count == 0):
            zero = torch.tensor(0.0)
            return {
                f"{self.prefix}abs_rel": zero,
                f"{self.prefix}rmse": zero,
                f"{self.prefix}delta1": zero,
            }
        return {
            f"{self.prefix}abs_rel": self.abs_rel_sum / self.count,
            f"{self.prefix}rmse": torch.sqrt(self.se_sum / self.count),
            f"{self.prefix}delta1": self.delta1_sum / self.count,
        }


def _get_watch_metric_mode(
    watch_metric_name: str, loss_names: Sequence[str]
) -> Literal["min", "max"]:
    """Returns whether ``watch_metric_name`` should be minimised or maximised.

    Matches the metric suffix (after ``/``) against the known depth metric names, then
    falls back to loss names (always minimised).
    """
    parts = watch_metric_name.split("/")
    if len(parts) > 1:
        suffix = parts[1]
        for metric_name, mode in _METRIC_MODES.items():
            if suffix.startswith(metric_name):
                return mode
    for loss_name in loss_names:
        if loss_name in watch_metric_name:
            return "min"
    raise ValueError(
        f"watch_metric_name {watch_metric_name!r} not found in metric names "
        f"{sorted(_METRIC_MODES)} or loss names {list(loss_names)}."
    )
