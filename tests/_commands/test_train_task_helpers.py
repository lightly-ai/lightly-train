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
import torch
from pytest import LogCaptureFixture
from pytest_mock import MockerFixture

from lightly_train._commands.train_task_helpers import (
    BestAggregatedMetricValues,
    get_best_metrics,
    get_train_model_args,
    get_train_model_cls,
    get_transform_args,
    load_checkpoint,
    log_step,
)
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._metrics.task_metric import AggregatedMetricValues, TaskMetricArgs
from lightly_train._task_models.ltdetr_object_detection.train_model import (
    DINOv2LTDETRObjectDetectionTrainArgsV2,
    LTDETRObjectDetectionTrain,
    LTDETRObjectDetectionTrainArgs,
)
from lightly_train._task_models.ltdetr_object_detection.transforms import (
    DINOv2LTDETRObjectDetectionTrainTransformArgsV2,
    DINOv2LTDETRObjectDetectionTrainTransformV2,
    DINOv2LTDETRObjectDetectionValTransformV2,
    LTDETRObjectDetectionTrainTransform,
    LTDETRObjectDetectionTrainTransformArgs,
    LTDETRObjectDetectionValTransform,
)
from lightly_train._training_step_timer import TimerAggregateMetrics


@pytest.mark.parametrize(
    ("model", "task", "checkpoint_model_name"),
    [
        (
            "dinov2/vits14-noreg-ltdetr-coco",
            "object_detection",
            "dinov2/vits14-noreg-ltdetr",
        ),
        ("ltdetrv2-s-coco", "object_detection", "edgecrafter/ecvitt-ltdetr"),
        (
            "dinov3/vits16-eomt-coco",
            "semantic_segmentation",
            "dinov3/vits16-eomt",
        ),
    ],
)
def test_load_checkpoint__downloadable_supported_alias_loads_checkpoint(
    mocker: MockerFixture,
    tmp_path: Path,
    model: str,
    task: str,
    checkpoint_model_name: str,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.touch()
    train_model_state_dict = {"model.weight": "pretrained"}
    checkpoint_dict = {
        "train_model": train_model_state_dict,
        "model_class_path": "package.TaskModel",
        "model_init_args": {"model_name": checkpoint_model_name},
    }
    fabric = mocker.MagicMock()
    fabric.load.return_value = checkpoint_dict
    download_checkpoint = mocker.patch(
        "lightly_train._commands.train_task_helpers.task_model_helpers.download_checkpoint",
        return_value=checkpoint_path,
    )

    checkpoint, resolved_path, resolved_model, model_init_args = load_checkpoint(
        fabric=fabric,
        out_dir=tmp_path / "out",
        resume_interrupted=False,
        model=model,
        checkpoint=None,
        task=task,
    )

    download_checkpoint.assert_called_once_with(checkpoint=model)
    fabric.load.assert_called_once_with(path=checkpoint_path)
    assert resolved_path == checkpoint_path
    assert resolved_model == checkpoint_model_name
    assert model_init_args == checkpoint_dict["model_init_args"]
    assert checkpoint == {
        "train_model_state_dict": train_model_state_dict,
        "model_class_path": "package.TaskModel",
        "model_init_args": checkpoint_dict["model_init_args"],
    }


def test_load_checkpoint__supported_architecture_does_not_load_checkpoint(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    fabric = mocker.MagicMock()
    download_checkpoint = mocker.patch(
        "lightly_train._commands.train_task_helpers.task_model_helpers.download_checkpoint"
    )

    result = load_checkpoint(
        fabric=fabric,
        out_dir=tmp_path / "out",
        resume_interrupted=False,
        model="ltdetrv2-s",
        checkpoint=None,
        task="object_detection",
    )

    assert result == (None, None, "ltdetrv2-s", None)
    download_checkpoint.assert_not_called()
    fabric.load.assert_not_called()


def test_load_checkpoint__local_model_path_loads_checkpoint(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.touch()
    checkpoint_dict = {
        "train_model": {"model.weight": "pretrained"},
        "model_class_path": "package.TaskModel",
        "model_init_args": {"model_name": "ltdetrv2-s"},
    }
    fabric = mocker.MagicMock()
    fabric.load.return_value = checkpoint_dict
    download_checkpoint = mocker.patch(
        "lightly_train._commands.train_task_helpers.task_model_helpers.download_checkpoint",
        return_value=model_path,
    )

    checkpoint, resolved_path, resolved_model, model_init_args = load_checkpoint(
        fabric=fabric,
        out_dir=tmp_path / "out",
        resume_interrupted=False,
        model=str(model_path),
        checkpoint=None,
        task="object_detection",
    )

    download_checkpoint.assert_called_once_with(checkpoint=str(model_path))
    fabric.load.assert_called_once_with(path=model_path)
    assert resolved_path == model_path
    assert resolved_model == "ltdetrv2-s"
    assert model_init_args == checkpoint_dict["model_init_args"]
    assert checkpoint is not None
    assert checkpoint["train_model_state_dict"] == checkpoint_dict["train_model"]


def test_load_checkpoint__explicit_checkpoint_overrides_downloadable_model(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "explicit.pt"
    checkpoint_path.touch()
    checkpoint_dict = {
        "train_model": {"model.weight": "explicit"},
        "model_class_path": "package.TaskModel",
        "model_init_args": {"model_name": "edgecrafter/ecvitt-ltdetr"},
    }
    fabric = mocker.MagicMock()
    fabric.load.return_value = checkpoint_dict
    download_checkpoint = mocker.patch(
        "lightly_train._commands.train_task_helpers.task_model_helpers.download_checkpoint"
    )

    _, resolved_path, resolved_model, _ = load_checkpoint(
        fabric=fabric,
        out_dir=tmp_path / "out",
        resume_interrupted=False,
        model="ltdetrv2-s-coco",
        checkpoint=checkpoint_path,
        task="object_detection",
    )

    assert resolved_path == checkpoint_path.resolve()
    assert resolved_model == "edgecrafter/ecvitt-ltdetr"
    download_checkpoint.assert_not_called()


def test_load_checkpoint__explicit_checkpoint_rejects_incompatible_model_family(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "explicit.pt"
    checkpoint_path.touch()
    fabric = mocker.MagicMock()
    fabric.load.return_value = {
        "train_model": {"model.weight": "explicit"},
        "model_class_path": "package.TaskModel",
        "model_init_args": {"model_name": "edgecrafter/ecvitt-ltdetr"},
    }

    with pytest.raises(ValueError, match="incompatible task-model families"):
        load_checkpoint(
            fabric=fabric,
            out_dir=tmp_path / "out",
            resume_interrupted=False,
            model="picodet/s-416",
            checkpoint=checkpoint_path,
            task="object_detection",
        )


def test_load_checkpoint__explicit_legacy_checkpoint_uses_model_fallback(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "legacy.ckpt"
    checkpoint_path.touch()
    fabric = mocker.MagicMock()
    fabric.load.return_value = {
        "train_model": {"model.weight": "legacy"},
        "model_class_path": "",
    }

    checkpoint, _, resolved_model, model_init_args = load_checkpoint(
        fabric=fabric,
        out_dir=tmp_path / "out",
        resume_interrupted=False,
        model="ltdetrv2-s",
        checkpoint=checkpoint_path,
        task="object_detection",
    )

    assert resolved_model == "ltdetrv2-s"
    assert model_init_args == {}
    assert checkpoint is not None
    assert checkpoint["train_model_state_dict"] == {"model.weight": "legacy"}


def test_load_checkpoint__resume_ignores_downloadable_model(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    checkpoint_path = out_dir / "checkpoints" / "last.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {"model_init_args": {"model_name": "edgecrafter/ecvitt-ltdetr"}},
        checkpoint_path,
    )
    fabric = mocker.MagicMock()
    download_checkpoint = mocker.patch(
        "lightly_train._commands.train_task_helpers.task_model_helpers.download_checkpoint"
    )

    result = load_checkpoint(
        fabric=fabric,
        out_dir=out_dir,
        resume_interrupted=True,
        model="ltdetrv2-s-coco",
        checkpoint=None,
        task="object_detection",
    )

    assert result == (
        None,
        checkpoint_path,
        "edgecrafter/ecvitt-ltdetr",
        {"model_name": "edgecrafter/ecvitt-ltdetr"},
    )
    download_checkpoint.assert_not_called()
    fabric.load.assert_not_called()


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
        model_name="dinov3/vitt16-notpretrained-ltdetr",
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
        model_name="dinov2/vits14-ltdetr",
        transform_args=None,
        ignore_index=None,
        model_init_args=resolved_model_init_args,
        total_steps=1000,
        train_num_batches=100,
        gradient_accumulation_steps=1,
    )

    # "dinov2/..." model names dispatch to DINOv2LTDETRObjectDetectionTrainTransformArgsV2,
    # not the generic LTDETRObjectDetectionTrainTransformArgs. This carries the original
    # patch-14-tuned scale-jitter sizes directly rather than deriving them from the
    # generic patch-16-tuned list via rounding, which previously silently dropped the
    # 560 and 784 sizes.
    train_transform_args = cast(
        DINOv2LTDETRObjectDetectionTrainTransformArgsV2, train_transform_args
    )

    assert train_model_args.patch_size == 14
    assert train_transform_args.scale_jitter is not None
    assert train_transform_args.scale_jitter.divisible_by is None
    assert train_transform_args.scale_jitter.sizes == [
        (476, 476),
        (504, 504),
        (532, 532),
        (560, 560),
        (588, 588),
        (616, 616),
        *([(644, 644)] * 20),
        (672, 672),
        (700, 700),
        (728, 728),
        (756, 756),
        (784, 784),
        (812, 812),
    ]


def test_get_train_model_cls__dinov2_ltdetr_routes_to_generic_pipeline() -> None:
    assert (
        get_train_model_cls(model_name="dinov2/vits14-ltdetr", task="object_detection")
        is LTDETRObjectDetectionTrain
    )
    assert (
        get_train_model_cls(
            model_name="dinov2/vits14-notpretrained-ltdetr",
            task="object_detection",
        )
        is LTDETRObjectDetectionTrain
    )


def test_get_train_transform_cls__dinov2_ltdetr_routes_to_dinov2_transform() -> None:
    assert (
        LTDETRObjectDetectionTrain.get_train_transform_cls("dinov2/vits14-ltdetr")
        is DINOv2LTDETRObjectDetectionTrainTransformV2
    )
    assert (
        LTDETRObjectDetectionTrain.get_val_transform_cls("dinov2/vits14-ltdetr")
        is DINOv2LTDETRObjectDetectionValTransformV2
    )


def test_get_train_transform_cls__dinov3_ltdetr_routes_to_generic_transform() -> None:
    assert (
        LTDETRObjectDetectionTrain.get_train_transform_cls(
            "dinov3/vitt16-notpretrained-ltdetr"
        )
        is LTDETRObjectDetectionTrainTransform
    )
    assert (
        LTDETRObjectDetectionTrain.get_val_transform_cls(
            "dinov3/vitt16-notpretrained-ltdetr"
        )
        is LTDETRObjectDetectionValTransform
    )


def test_get_train_model_cls__dinov2_ltdetr_dsp_is_unsupported() -> None:
    # The DSP variant has no equivalent config in the generic pipeline's
    # LTDETR_MODEL_REGISTRY. This must keep raising.
    with pytest.raises(ValueError, match="Unsupported model name"):
        get_train_model_cls(
            model_name="dinov2/vits14-ltdetr-dsp", task="object_detection"
        )


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
