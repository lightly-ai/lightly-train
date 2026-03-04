#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pytest import LogCaptureFixture

from lightly_train._commands.train_task_helpers import BestMetric, get_best_metrics
from lightly_train._metrics.task_metric import MetricComputeResult, TaskMetricArgs


def test_get_best_metrics__no_previous_best() -> None:
    last = MetricComputeResult(
        metrics={"val_metric/acc": 0.8},
        watch_metric="val_metric/acc",
        watch_metric_value=0.8,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metrics=None,
    )
    result = get_best_metrics(
        best_metrics=None,
        last_metrics=last,
        step=0,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result.metrics is last
    assert result.step == 0


def test_get_best_metrics__max_mode_improvement() -> None:
    prev = MetricComputeResult(
        metrics={"val_metric/acc": 0.5},
        watch_metric="val_metric/acc",
        watch_metric_value=0.5,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metrics=None,
    )
    best = BestMetric(metrics=prev, step=0)
    last = MetricComputeResult(
        metrics={"val_metric/acc": 0.8},
        watch_metric="val_metric/acc",
        watch_metric_value=0.8,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metrics=None,
    )
    result = get_best_metrics(
        best_metrics=best,
        last_metrics=last,
        step=1,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result.metrics is last
    assert result.step == 1


def test_get_best_metrics__max_mode_no_improvement() -> None:
    prev = MetricComputeResult(
        metrics={"val_metric/acc": 0.9},
        watch_metric="val_metric/acc",
        watch_metric_value=0.9,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metrics=None,
    )
    best = BestMetric(metrics=prev, step=0)
    last = MetricComputeResult(
        metrics={"val_metric/acc": 0.7},
        watch_metric="val_metric/acc",
        watch_metric_value=0.7,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metrics=None,
    )
    result = get_best_metrics(
        best_metrics=best,
        last_metrics=last,
        step=1,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result is best


def test_get_best_metrics__min_mode_improvement() -> None:
    prev = MetricComputeResult(
        metrics={"val_loss": 0.8},
        watch_metric="val_loss",
        watch_metric_value=0.8,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metrics=None,
    )
    best = BestMetric(metrics=prev, step=0)
    last = MetricComputeResult(
        metrics={"val_loss": 0.3},
        watch_metric="val_loss",
        watch_metric_value=0.3,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metrics=None,
    )
    result = get_best_metrics(
        best_metrics=best,
        last_metrics=last,
        step=2,
        metric_args=TaskMetricArgs(watch_metric="val_loss"),
    )
    assert result.metrics is last
    assert result.step == 2


def test_get_best_metrics__min_mode_no_improvement() -> None:
    prev = MetricComputeResult(
        metrics={"val_loss": 0.3},
        watch_metric="val_loss",
        watch_metric_value=0.3,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metrics=None,
    )
    best = BestMetric(metrics=prev, step=0)
    last = MetricComputeResult(
        metrics={"val_loss": 0.9},
        watch_metric="val_loss",
        watch_metric_value=0.9,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metrics=None,
    )
    result = get_best_metrics(
        best_metrics=best,
        last_metrics=last,
        step=2,
        metric_args=TaskMetricArgs(watch_metric="val_loss"),
    )
    assert result is best


def test_get_best_metrics__missing_watch_metric(caplog: LogCaptureFixture) -> None:
    # watch_metric configured but not present in computed metrics
    # last is returned as best since no valid best exists.
    prev = MetricComputeResult(
        metrics={"val_metric/acc": 0.9},
        watch_metric=None,
        watch_metric_value=None,
        watch_metric_mode=None,
        best_head_name=None,
        best_head_metrics=None,
    )
    best = BestMetric(metrics=prev, step=0)
    last = MetricComputeResult(
        metrics={"val_metric/acc": 0.95},
        watch_metric=None,
        watch_metric_value=None,
        watch_metric_mode=None,
        best_head_name=None,
        best_head_metrics=None,
    )
    with caplog.at_level("WARNING"):
        result = get_best_metrics(
            best_metrics=best,
            last_metrics=last,
            step=1,
            metric_args=TaskMetricArgs(watch_metric="val_metric/nonexistent"),
        )
    assert "Unknown watch metric" in caplog.text
    assert result.metrics is last
    assert result.step == 1
