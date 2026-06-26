#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch

from lightly_train._metrics.depth_estimation.task_metric import (
    DepthEstimationTaskMetric,
    DepthEstimationTaskMetricArgs,
)


class TestDepthEstimationTaskMetric:
    def test___init___watch_metric_mode_min_for_rmse(self) -> None:
        metric = DepthEstimationTaskMetric(
            task_metric_args=DepthEstimationTaskMetricArgs(),
            split="val",
            loss_names=["loss"],
        )

        assert metric.watch_metric == "val_metric/rmse"
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
        assert agg.watch_metric == "val_metric/rmse"
        assert agg.watch_metric_mode == "min"

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
