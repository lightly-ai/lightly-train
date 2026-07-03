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

# Small constant guarding divisions and the degenerate (near-constant prediction)
# case of the scale-and-shift fit.
_EPS = 1e-8

# Standard relative-error depth metrics and whether they should be minimised or
# maximised for best-model selection.
_METRIC_MODES: dict[str, Literal["min", "max"]] = {
    "abs_rel": "min",
    "rmse": "min",
    "delta1": "max",
}


class DepthEstimationTaskMetricArgs(TaskMetricArgs):
    """Metrics configuration for depth estimation."""

    watch_metric: str = "val_metric/abs_rel"
    # Whether to also compute quality metrics (not just losses) on the train split.
    train: bool = False


class DepthEstimationTaskMetric(TaskMetric):
    """Stores depth metrics and losses for one split.

    Quality metrics are AbsRel, RMSE and ``delta1`` (the fraction of pixels with
    ``max(pred/target, target/pred) < 1.25``), all computed over valid pixels
    (``target > 0``) after a per-image least-squares scale-and-shift alignment of the
    prediction to the target. The student predicts relative depth with an unconstrained
    global scale and shift (the scale-invariant training loss deliberately does not pin
    them down), so raw metrics would be dominated by that arbitrary scale rather than by
    depth quality; aligning first is the standard MiDaS/DepthAnything relative-depth
    evaluation protocol.
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
    """Accumulates AbsRel, RMSE and ``delta1`` over valid pixels across batches.

    Each prediction is aligned to its target by a per-image least-squares scale-and-shift
    fit before the metrics are computed (see ``_align_scale_shift``), so the metrics
    measure relative-depth quality independent of the arbitrary global scale/shift the
    scale-invariant training loss leaves free.
    """

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
        # Align each image independently: a batch-wide fit would let one image's scale
        # contaminate another. Errors are summed and divided by the pixel count at
        # ``compute`` time so the epoch aggregate is a proper pixel-weighted mean.
        for pred_img, target_img in zip(preds.float(), target.float()):
            valid = (
                (target_img > 0) & torch.isfinite(pred_img) & torch.isfinite(target_img)
            )
            if not bool(valid.any()):
                continue
            pred = pred_img[valid]
            gt = target_img[valid]
            pred = _align_scale_shift(pred=pred, target=gt)
            # The aligned prediction can be non-positive where the fit extrapolates
            # below zero; the ratio-based metrics need positive values, so clamp.
            pred = pred.clamp_min(_EPS)

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


def _align_scale_shift(*, pred: Tensor, target: Tensor) -> Tensor:
    """Returns ``pred`` aligned to ``target`` by a least-squares scale and shift.

    Solves ``min_{s, t} sum_i (s * pred_i + t - target_i)^2`` in closed form and returns
    ``s * pred + t``. This is the standard MiDaS/DepthAnything relative-depth evaluation
    alignment: the student's depth has an arbitrary global scale and shift (the
    scale-invariant loss does not constrain them), so the prediction must be aligned to
    the target before an error metric is meaningful.

    Args:
        pred: Predicted depth over valid pixels, shape ``(N,)``.
        target: Target depth over the same valid pixels, shape ``(N,)``.

    Returns:
        The aligned prediction, shape ``(N,)``. If the prediction is (near-)constant the
        fit is degenerate; the prediction is then shifted to match the target mean.
    """
    n = pred.numel()
    mean_pred = pred.mean()
    mean_target = target.mean()
    pred_centered = pred - mean_pred
    var_pred = torch.sum(pred_centered * pred_centered) / n
    if bool(var_pred < _EPS):
        return pred - mean_pred + mean_target
    cov = torch.sum(pred_centered * (target - mean_target)) / n
    scale = cov / var_pred
    shift = mean_target - scale * mean_pred
    return scale * pred + shift


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
