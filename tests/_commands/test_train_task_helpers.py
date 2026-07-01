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
    get_train_model_cls,
    get_transform_args,
    log_step,
)
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._metrics.task_metric import AggregatedMetricValues, TaskMetricArgs
from lightly_train._task_models.dinov2_ltdetr_object_detection.train_model import (
    DINOv2LTDETRObjectDetectionTrain,
)
from lightly_train._task_models.ltdetr_object_detection.train_model import (
    DINOv2LTDETRObjectDetectionTrainArgsV2,
    LTDETRObjectDetectionTrain,
    LTDETRObjectDetectionTrainArgs,
)
from lightly_train._task_models.ltdetr_object_detection.transforms import (
    LTDETRObjectDetectionTrainTransformArgs,
)
from lightly_train._training_step_timer import TimerAggregateMetrics


def test_get_train_model_args_and_transform_args__propagate_patch_size_to_scale_jitter() -> (
    None
):
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
    )

    train_model_args = cast(
        LTDETRObjectDetectionTrainArgs,
        get_train_model_args(
            model_args={"patch_size": 14},
            model_args_cls=LTDETRObjectDetectionTrainArgs,
            total_steps=1000,
            gradient_accumulation_steps=1,
            train_num_batches=100,
            model_name="dinov3/vitt16-notpretrained-ltdetr",
            model_init_args={},
            data_args=data_args,
        ),
    )

    resolved_model_init_args: dict[str, int] = {}
    if isinstance(train_model_args.patch_size, int):
        resolved_model_init_args["patch_size"] = train_model_args.patch_size

    train_transform_args, _ = get_transform_args(
        train_model_cls=LTDETRObjectDetectionTrain,
        transform_args=None,
        ignore_index=None,
        model_init_args=resolved_model_init_args,
        total_steps=1000,
        train_num_batches=100,
        gradient_accumulation_steps=1,
    )

    train_transform_args = cast(
        LTDETRObjectDetectionTrainTransformArgs, train_transform_args
    )

    assert train_model_args.patch_size == 14
    assert train_transform_args.scale_jitter is not None
    assert train_transform_args.scale_jitter.divisible_by == 28


def test_get_train_model_args_and_transform_args__propagate_dinov2_patch_size_to_scale_jitter() -> (
    None
):
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
    )

    train_model_args = cast(
        DINOv2LTDETRObjectDetectionTrainArgsV2,
        get_train_model_args(
            model_args={},
            model_args_cls=DINOv2LTDETRObjectDetectionTrainArgsV2,
            total_steps=1000,
            gradient_accumulation_steps=1,
            train_num_batches=100,
            model_name="dinov2/vits14-ltdetr",
            model_init_args={},
            data_args=data_args,
        ),
    )

    resolved_model_init_args: dict[str, int] = {
        "patch_size": train_model_args.patch_size
    }

    train_transform_args, _ = get_transform_args(
        train_model_cls=LTDETRObjectDetectionTrain,
        transform_args=None,
        ignore_index=None,
        model_init_args=resolved_model_init_args,
        total_steps=1000,
        train_num_batches=100,
        gradient_accumulation_steps=1,
    )

    train_transform_args = cast(
        LTDETRObjectDetectionTrainTransformArgs, train_transform_args
    )

    assert train_model_args.patch_size == 14
    assert train_transform_args.scale_jitter is not None
    assert train_transform_args.scale_jitter.divisible_by == 28


def test_get_train_model_cls__dinov2_ltdetr_routes_to_generic_pipeline() -> None:
    assert (
        get_train_model_cls(model_name="dinov2/vits14-ltdetr", task="object_detection")
        is LTDETRObjectDetectionTrain
    )


def test_get_train_model_cls__dinov2_ltdetr_dsp_is_unsupported() -> None:
    # The DSP variant has no equivalent config in the generic pipeline's
    # LTDETR_MODEL_REGISTRY yet, and DINOv2LTDETRObjectDetectionTrain (standalone) was
    # never wired up to accept the "-ltdetr-dsp" suffix either. This must keep raising
    # after removing DINOv2LTDETRObjectDetectionTrain from the training dispatch list.
    with pytest.raises(ValueError, match="Unsupported model name"):
        get_train_model_cls(
            model_name="dinov2/vits14-ltdetr-dsp", task="object_detection"
        )
    # Sanity check the standalone class itself still exists and is constructible
    # directly (e.g. for loading old checkpoints), it's just no longer reachable via
    # get_train_model_cls.
    assert DINOv2LTDETRObjectDetectionTrain.task == "object_detection"


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


def _make_timer_agg() -> TimerAggregateMetrics:
    return TimerAggregateMetrics(
        step_total_times={"train_step": 1.0, "train_dataload": 0.2},
        step_counts={"train_step": 1, "train_dataload": 1},
        phase_gpu_utils={},
        phase_gpu_max_mem={},
    )


def test_log_step__includes_gradient_norm(caplog: LogCaptureFixture) -> None:
    with caplog.at_level("INFO"):
        log_step(
            split="train",
            step=0,
            max_steps=10,
            epoch=0,
            agg_metric_values=None,
            task="image_classification",
            timer_agg=_make_timer_agg(),
            global_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            gradient_norm=12.3456,
        )
    assert "grad_norm: 12.3456" in caplog.text


def test_log_step__omits_gradient_norm_when_none(caplog: LogCaptureFixture) -> None:
    with caplog.at_level("INFO"):
        log_step(
            split="train",
            step=0,
            max_steps=10,
            epoch=0,
            agg_metric_values=None,
            task="image_classification",
            timer_agg=_make_timer_agg(),
            global_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            gradient_norm=None,
        )
    assert "grad_norm" not in caplog.text
