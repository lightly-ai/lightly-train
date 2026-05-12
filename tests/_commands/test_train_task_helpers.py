#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pytest import LogCaptureFixture

from lightly_train._commands.train_task_helpers import (
    BestAggregatedMetricValues,
    get_best_metrics,
    get_train_model_args,
    get_transform_args,
)
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._metrics.task_metric import AggregatedMetricValues, TaskMetricArgs
from lightly_train._task_models.dinov3_ltdetr_object_detection.train_model import (
    DINOv3LTDETRObjectDetectionTrain,
    DINOv3LTDETRObjectDetectionTrainArgs,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.transforms import (
    DINOv3LTDETRObjectDetectionTrainTransformArgs,
)


def test_get_train_model_args_and_transform_args__propagate_patch_size_to_scale_jitter() -> None:
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
    )

    train_model_args = cast(
        DINOv3LTDETRObjectDetectionTrainArgs,
        get_train_model_args(
            model_args={"patch_size": 14},
            model_args_cls=DINOv3LTDETRObjectDetectionTrainArgs,
            total_steps=1000,
            model_name="dinov3/vitt16-notpretrained-ltdetr",
            model_init_args={},
            data_args=data_args,
        ),
    )

    resolved_model_init_args: dict[str, int] = {}
    if isinstance(train_model_args.patch_size, int):
        resolved_model_init_args["patch_size"] = train_model_args.patch_size

    train_transform_args, _ = get_transform_args(
        train_model_cls=DINOv3LTDETRObjectDetectionTrain,
        transform_args=None,
        ignore_index=None,
        model_init_args=resolved_model_init_args,
        total_steps=1000,
        train_num_batches=100,
        gradient_accumulation_steps=1,
    )

    train_transform_args = cast(
        DINOv3LTDETRObjectDetectionTrainTransformArgs, train_transform_args
    )

    assert train_model_args.patch_size == 14
    assert train_transform_args.scale_jitter is not None
    assert train_transform_args.scale_jitter.divisible_by == 28


def test_get_best_metrics__no_previous_best() -> None:
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.8},
        watch_metric="val_metric/acc",
        watch_metric_value=0.8,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=None,
        last_agg_metric_values=last,
        step=0,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result.agg_metric_values is last
    assert result.step == 0


def test_get_best_metrics__max_mode_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.5},
        watch_metric="val_metric/acc",
        watch_metric_value=0.5,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.8},
        watch_metric="val_metric/acc",
        watch_metric_value=0.8,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=1,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result.agg_metric_values is last
    assert result.step == 1


def test_get_best_metrics__max_mode_no_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.9},
        watch_metric="val_metric/acc",
        watch_metric_value=0.9,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.7},
        watch_metric="val_metric/acc",
        watch_metric_value=0.7,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=1,
        metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
    )
    assert result is best


def test_get_best_metrics__min_mode_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_loss": 0.8},
        watch_metric="val_loss",
        watch_metric_value=0.8,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_loss": 0.3},
        watch_metric="val_loss",
        watch_metric_value=0.3,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=2,
        metric_args=TaskMetricArgs(watch_metric="val_loss"),
    )
    assert result.agg_metric_values is last
    assert result.step == 2


def test_get_best_metrics__min_mode_no_improvement() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_loss": 0.3},
        watch_metric="val_loss",
        watch_metric_value=0.3,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_loss": 0.9},
        watch_metric="val_loss",
        watch_metric_value=0.9,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )
    result = get_best_metrics(
        best_agg_metric_values=best,
        last_agg_metric_values=last,
        step=2,
        metric_args=TaskMetricArgs(watch_metric="val_loss"),
    )
    assert result is best


def test_get_best_metrics__missing_watch_metric(caplog: LogCaptureFixture) -> None:
    # watch_metric configured but not present in computed metrics
    # last is returned as best since no valid best exists.
    prev = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.9},
        watch_metric=None,
        watch_metric_value=None,
        watch_metric_mode=None,
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.95},
        watch_metric=None,
        watch_metric_value=None,
        watch_metric_mode=None,
        best_head_name=None,
        best_head_metric_values=None,
    )
    with caplog.at_level("WARNING"):
        result = get_best_metrics(
            best_agg_metric_values=best,
            last_agg_metric_values=last,
            step=1,
            metric_args=TaskMetricArgs(watch_metric="val_metric/nonexistent"),
        )
    assert "Unknown watch metric" in caplog.text
    assert result.agg_metric_values is last
    assert result.step == 1


def test_get_best_metrics__different_watch_metric_raises() -> None:
    prev = AggregatedMetricValues(
        metric_values={"val_metric/acc": 0.9},
        watch_metric="val_metric/acc",
        watch_metric_value=0.9,
        watch_metric_mode="max",
        best_head_name=None,
        best_head_metric_values=None,
    )
    best = BestAggregatedMetricValues(agg_metric_values=prev, step=0)
    last = AggregatedMetricValues(
        metric_values={"val_loss": 0.1},
        watch_metric="val_loss",
        watch_metric_value=0.1,
        watch_metric_mode="min",
        best_head_name=None,
        best_head_metric_values=None,
    )

    with pytest.raises(
        RuntimeError,
        match="Best and last aggregated metrics use different watch metrics",
    ):
        get_best_metrics(
            best_agg_metric_values=best,
            last_agg_metric_values=last,
            step=1,
            metric_args=TaskMetricArgs(watch_metric="val_metric/acc"),
        )
