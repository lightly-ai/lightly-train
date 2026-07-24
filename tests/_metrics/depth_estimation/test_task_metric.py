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

from lightly_train._metrics.depth_estimation.task_metric import (
    DepthEstimationTaskMetric,
    DepthEstimationTaskMetricArgs,
)


class TestDepthEstimationTaskMetric:
    def test___init___watch_metric_mode_min_for_abs_rel(self) -> None:
        metric = DepthEstimationTaskMetric(
            task_metric_args=DepthEstimationTaskMetricArgs(),
            split="val",
            loss_names=["loss"],
        )

        assert metric.watch_metric == "val_metric/abs_rel"
        assert metric.watch_metric_mode == "min"

    def test_compute_aggregated_values__perfect_prediction(self) -> None:
        metric = DepthEstimationTaskMetric(
            task_metric_args=DepthEstimationTaskMetricArgs(),
            split="val",
            loss_names=["loss"],
        )
        preds = torch.rand(2, 1, 8, 8) + 1.0
        target = preds.clone()

        metric.update_with_predictions(preds, target)
        metric.update_with_losses({"loss": torch.tensor(0.5)}, weight=2)
        agg = metric.compute_aggregated_values()

        assert agg.metric_values["val_metric/rmse"] == 0.0
        assert agg.metric_values["val_metric/abs_rel"] == 0.0
        assert agg.metric_values["val_metric/delta1"] == 1.0
        assert agg.watch_metric == "val_metric/abs_rel"
        assert agg.watch_metric_mode == "min"

    def test_update_with_predictions__scale_shift_invariant(self) -> None:
        # A prediction that is a global affine transform of the target is a perfect
        # relative-depth prediction: alignment must recover it and report zero error.
        metric = DepthEstimationTaskMetric(
            task_metric_args=DepthEstimationTaskMetricArgs(),
            split="val",
            loss_names=["loss"],
        )
        target = torch.rand(1, 1, 8, 8) + 1.0
        preds = 3.0 * target + 2.0

        metric.update_with_predictions(preds, target)
        agg = metric.compute_aggregated_values()

        assert agg.metric_values["val_metric/abs_rel"] == pytest.approx(0.0, abs=1e-5)
        assert agg.metric_values["val_metric/rmse"] == pytest.approx(0.0, abs=1e-5)
        assert agg.metric_values["val_metric/delta1"] == 1.0

    def test_update_with_predictions__aligns_each_image_independently(self) -> None:
        # Two images with different (target = affine of pred) relationships. A single
        # batch-wide fit could not zero both; per-image alignment does.
        metric = DepthEstimationTaskMetric(
            task_metric_args=DepthEstimationTaskMetricArgs(),
            split="val",
            loss_names=["loss"],
        )
        target = torch.rand(2, 1, 8, 8) + 1.0
        preds = torch.empty_like(target)
        preds[0] = 3.0 * target[0] + 2.0
        preds[1] = 0.5 * target[1] - 1.0

        metric.update_with_predictions(preds, target)
        agg = metric.compute_aggregated_values()

        assert agg.metric_values["val_metric/abs_rel"] == pytest.approx(0.0, abs=1e-5)
        assert agg.metric_values["val_metric/rmse"] == pytest.approx(0.0, abs=1e-5)

    def test_update_with_predictions__no_alignment_penalizes_scale(self) -> None:
        # For a metric model (align=False) a globally scaled/shifted prediction is NOT a
        # perfect prediction: the raw error must be non-zero (contrast with the aligned
        # `test_update_with_predictions__scale_shift_invariant`).
        metric = DepthEstimationTaskMetric(
            task_metric_args=DepthEstimationTaskMetricArgs(),
            split="val",
            loss_names=["loss"],
            align=False,
        )
        target = torch.rand(1, 1, 8, 8) + 1.0
        preds = 3.0 * target + 2.0

        metric.update_with_predictions(preds, target)
        agg = metric.compute_aggregated_values()

        assert agg.metric_values["val_metric/abs_rel"] > 0.0
        assert agg.metric_values["val_metric/rmse"] > 0.0

    def test_update_with_predictions__no_alignment_perfect_prediction(self) -> None:
        # A perfect metric prediction still yields zero error without alignment.
        metric = DepthEstimationTaskMetric(
            task_metric_args=DepthEstimationTaskMetricArgs(),
            split="val",
            loss_names=["loss"],
            align=False,
        )
        preds = torch.rand(2, 1, 8, 8) + 1.0
        target = preds.clone()

        metric.update_with_predictions(preds, target)
        agg = metric.compute_aggregated_values()

        assert agg.metric_values["val_metric/abs_rel"] == pytest.approx(0.0, abs=1e-5)
        assert agg.metric_values["val_metric/rmse"] == pytest.approx(0.0, abs=1e-5)
        assert agg.metric_values["val_metric/delta1"] == 1.0

    def test_update_with_predictions__aligned_diagnostic_for_metric(self) -> None:
        # A metric model (align=False) also emits the aligned diagnostic variants. For a
        # globally scaled/shifted prediction the primary (unaligned) error is non-zero but
        # the aligned diagnostic recovers the affine transform and reports ~zero error.
        metric = DepthEstimationTaskMetric(
            task_metric_args=DepthEstimationTaskMetricArgs(),
            split="val",
            loss_names=["loss"],
            align=False,
        )
        target = torch.rand(1, 1, 8, 8) + 1.0
        preds = 3.0 * target + 2.0

        metric.update_with_predictions(preds, target)
        agg = metric.compute_aggregated_values()

        assert agg.metric_values["val_metric/abs_rel"] > 0.0
        assert agg.metric_values["val_metric/abs_rel_aligned"] == pytest.approx(
            0.0, abs=1e-5
        )
        assert agg.metric_values["val_metric/rmse_aligned"] == pytest.approx(
            0.0, abs=1e-5
        )
        assert agg.metric_values["val_metric/delta1_aligned"] == 1.0

    def test_update_with_predictions__no_aligned_diagnostic_for_relative(self) -> None:
        # Relative models (align=True) do not emit the aligned diagnostic keys: their
        # primary metrics are already aligned.
        metric = DepthEstimationTaskMetric(
            task_metric_args=DepthEstimationTaskMetricArgs(),
            split="val",
            loss_names=["loss"],
        )
        preds = torch.rand(1, 1, 8, 8) + 1.0
        target = preds.clone()

        metric.update_with_predictions(preds, target)
        agg = metric.compute_aggregated_values()

        assert "val_metric/abs_rel_aligned" not in agg.metric_values

    def test_update_with_predictions__masks_invalid(self) -> None:
        metric = DepthEstimationTaskMetric(
            task_metric_args=DepthEstimationTaskMetricArgs(),
            split="val",
            loss_names=["loss"],
        )
        preds = torch.rand(1, 1, 8, 8) + 1.0
        target = preds.clone()
        # Mark half the pixels invalid and corrupt the prediction there.
        target[:, :, :4, :] = 0.0
        preds_corrupt = preds.clone()
        preds_corrupt[:, :, :4, :] = 999.0

        metric.update_with_predictions(preds_corrupt, target)
        agg = metric.compute_aggregated_values()

        assert agg.metric_values["val_metric/rmse"] == 0.0

    def test_update_with_predictions__not_computed_for_train_by_default(self) -> None:
        # By default the train split does not compute quality metrics, only losses.
        metric = DepthEstimationTaskMetric(
            task_metric_args=DepthEstimationTaskMetricArgs(),
            split="train",
            loss_names=["loss"],
            train_loss_running_mean_window=1,
        )
        preds = torch.rand(1, 1, 8, 8) + 1.0
        target = torch.rand(1, 1, 8, 8) + 1.0

        metric.update_with_predictions(preds, target)
        metric.update_with_losses({"loss": torch.tensor(0.5)}, weight=1)
        agg = metric.compute_aggregated_values()

        assert "train_metric/rmse" not in agg.metric_values
        assert "train_loss" in agg.metric_values
