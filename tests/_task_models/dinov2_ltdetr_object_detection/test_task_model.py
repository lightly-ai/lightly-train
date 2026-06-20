#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from pytest_mock import MockerFixture
from torch import nn
from torch.optim.lr_scheduler import LinearLR

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._metrics.detection.task_metric import ObjectDetectionTaskMetricArgs
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._task_models.dinov2_ltdetr_object_detection.task_model import (
    DINOv2LTDETRDSPObjectDetection,
    DINOv2LTDETRObjectDetection,
)
from lightly_train._task_models.dinov2_ltdetr_object_detection.train_model import (
    DINOv2LTDETRObjectDetectionTrain,
    DINOv2LTDETRObjectDetectionTrainArgs,
)
from lightly_train._task_models.dinov2_ltdetr_object_detection.transforms import (
    DINOv2LTDETRObjectDetectionTrainTransformArgs,
    DINOv2LTDETRObjectDetectionValTransformArgs,
)
from lightly_train._task_models.object_detection_components.flat_cosine import (
    FlatCosineLRScheduler,
)


@pytest.mark.parametrize("use_ema_model", [True, False])
def test_load_train_state_dict__from_exported(use_ema_model: bool) -> None:
    model_args = DINOv2LTDETRObjectDetectionTrainArgs(use_ema_model=use_ema_model)
    train_model = _create_train_model(model_args)
    task_model = train_model.model
    state_dict = train_model.get_export_state_dict()
    task_model.load_train_state_dict(state_dict)


def test_load_train_state_dict__no_ema_weights() -> None:
    model_args = DINOv2LTDETRObjectDetectionTrainArgs(use_ema_model=True)
    train_model = _create_train_model(model_args)
    task_model = train_model.model
    state_dict = train_model.state_dict()
    # Drop all EMA weights from the state dict. This is for backwards compatibility
    # with older checkpoints. The model should still be able to load the weights by
    # copying the non-EMA weights to the EMA model.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ema_model.")}
    task_model.load_train_state_dict(state_dict)


def _is_module_frozen(m: nn.Module) -> bool:
    return all(not param.requires_grad for param in m.parameters())


@pytest.mark.parametrize("should_freeze", [True, False])
def test_freeze_backbone_on_set_train_mode(should_freeze: bool) -> None:
    model_args = DINOv2LTDETRObjectDetectionTrainArgs(
        use_ema_model=True, backbone_freeze=should_freeze
    )
    train_model = _create_train_model(model_args)
    task_model_backbone = train_model.model.backbone
    assert isinstance(task_model_backbone, nn.Module), "Backbone should be a nn.Module"

    train_model.set_train_mode()

    assert _is_module_frozen(task_model_backbone) == should_freeze, (
        f"Backbone should be frozen: {should_freeze}, but got frozen={_is_module_frozen(task_model_backbone)}"
    )
    assert not task_model_backbone.training == should_freeze, (
        "Backbone should be in eval mode after set_train_mode()"
    )


def _create_train_model(
    train_model_args: DINOv2LTDETRObjectDetectionTrainArgs,
) -> DINOv2LTDETRObjectDetectionTrain:
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
    )
    train_model_args.resolve_auto(
        total_steps=1000,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name="dinov2/_vittest14-ltdetr",
        model_init_args={},
        data_args=data_args,
    )
    train_transform_args = DINOv2LTDETRObjectDetectionTrainTransformArgs()
    train_transform_args.resolve_auto(model_init_args={})
    val_transform_args = DINOv2LTDETRObjectDetectionValTransformArgs()
    val_transform_args.resolve_auto(model_init_args={})

    train_model = DINOv2LTDETRObjectDetectionTrain(
        model_name="dinov2/_vittest14-ltdetr",
        model_args=train_model_args,
        data_args=data_args,
        train_transform_args=train_transform_args,
        val_transform_args=val_transform_args,
        metric_args=ObjectDetectionTaskMetricArgs(),
        load_weights=False,
        gradient_accumulation_steps=1,
    )
    return train_model


@pytest.mark.parametrize(
    ("scheduler_name", "scheduler_cls"),
    [
        ("linear", LinearLR),
        ("flat-cosine", FlatCosineLRScheduler),
    ],
)
def test_get_optimizer__scheduler_modes(
    scheduler_name: Literal["linear", "flat-cosine"],
    scheduler_cls: type[LinearLR] | type[FlatCosineLRScheduler],
) -> None:
    train_model = _create_train_model(
        DINOv2LTDETRObjectDetectionTrainArgs(
            scheduler_name=scheduler_name,
            lr_warmup_steps=500,
            scheduler_flat_steps=550,
            scheduler_no_aug_steps=150,
        )
    )
    optimizer, scheduler = train_model.get_optimizer(
        total_steps=1000,
        global_batch_size=16,
    )

    assert isinstance(scheduler, scheduler_cls)
    optimizer.step()
    scheduler.step()
    assert len(scheduler.get_last_lr()) == len(optimizer.param_groups)
    scheduler.load_state_dict(scheduler.state_dict())  # type: ignore[no-untyped-call]


def test_get_optimizer__flat_cosine_raises_when_cosine_phase_collapses() -> None:
    train_model = _create_train_model(
        DINOv2LTDETRObjectDetectionTrainArgs(
            scheduler_name="flat-cosine",
            lr_warmup_steps=1000,
        )
    )

    with pytest.raises(ValueError, match="non-empty cosine phase"):
        train_model.get_optimizer(total_steps=1000, global_batch_size=16)


def test_get_optimizer__linear_warns_when_warmup_exceeds_training(
    caplog: pytest.LogCaptureFixture,
) -> None:
    train_model = _create_train_model(
        DINOv2LTDETRObjectDetectionTrainArgs(
            scheduler_name="linear",
            lr_warmup_steps=1001,
        )
    )

    with caplog.at_level(
        logging.WARNING,
        logger="lightly_train._task_models.dinov2_ltdetr_object_detection.train_model",
    ):
        train_model.get_optimizer(total_steps=1000, global_batch_size=16)

    assert "the schedule will not complete as intended" in caplog.text


def test_predict_batch__composes_stages_in_order(mocker: MockerFixture) -> None:
    model = DINOv2LTDETRObjectDetection(
        model_name="dinov2/vits14-ltdetr",
        classes={0: "class_0", 1: "class_1"},
        image_size=(224, 224),
        load_weights=False,
    )

    preprocess_image_spy = mocker.spy(model, "preprocess_image")
    preprocess_batch_spy = mocker.spy(model, "preprocess_batch")
    forward_backend_spy = mocker.spy(model, "forward_backend")
    postprocess_spy = mocker.spy(model, "postprocess")

    images = [torch.rand(3, 480, 640), torch.rand(3, 720, 1280)]
    result = model.predict_batch(images=images)

    # Each input image goes through preprocess_image once.
    assert preprocess_image_spy.call_count == 2

    # The stacked batch is preprocessed in a single call with shape (B, C, H, W).
    assert preprocess_batch_spy.call_count == 1
    (batch_in,) = preprocess_batch_spy.call_args.args
    assert batch_in.shape == (2, 3, 224, 224)

    # forward_backend receives the output of preprocess_batch.
    assert forward_backend_spy.call_count == 1
    (forward_in,) = forward_backend_spy.call_args.args
    assert forward_in is preprocess_batch_spy.spy_return

    # postprocess receives forward_backend's output and per-image metadata.
    assert postprocess_spy.call_count == 1
    raw_in, metadata = postprocess_spy.call_args.args
    assert raw_in is forward_backend_spy.spy_return
    assert len(metadata) == 2

    # predict_batch returns whatever postprocess produced.
    assert result is postprocess_spy.spy_return


def _build_dsp_model(mocker: MockerFixture) -> DINOv2LTDETRDSPObjectDetection:
    # The DSP variant overrides __init__ and deliberately skips the base LTDETR
    # __init__ via super(_DINOv2LTDETRBase, self). Unlike the regular variant it
    # builds its backbone without exposing load_weights, so patch get_model to
    # avoid downloading pretrained weights.
    original_get_model = DINOV2_VIT_PACKAGE.get_model

    def _get_model_no_download(*args: Any, **kwargs: Any) -> Any:
        kwargs["load_weights"] = False
        return original_get_model(*args, **kwargs)

    mocker.patch.object(
        DINOV2_VIT_PACKAGE, "get_model", side_effect=_get_model_no_download
    )

    return DINOv2LTDETRDSPObjectDetection(
        model_name="dinov2/vits14-ltdetr-dsp",
        classes={0: "class_0", 1: "class_1"},
        image_size=(224, 224),
    )


def test_dsp_variant__builds_full_stack(mocker: MockerFixture) -> None:
    # Guards the refactoring: the __init__ override that skips the base class init
    # must still wire up a complete, correctly-configured object graph.
    model = _build_dsp_model(mocker)

    assert isinstance(model.backbone, nn.Module)
    assert isinstance(model.encoder, nn.Module)
    assert isinstance(model.decoder, nn.Module)
    assert isinstance(model.postprocessor, nn.Module)

    # The class-id lookup buffer registered by the override must match `classes`.
    assert model.internal_class_to_class.tolist() == [0, 1]


def test_dsp_variant__forward(mocker: MockerFixture) -> None:
    model = _build_dsp_model(mocker)
    model.deploy()
    with torch.no_grad():
        labels, boxes, scores = model(torch.rand(1, 3, 224, 224))
    assert labels.shape[0] == boxes.shape[0] == scores.shape[0] == 1
    assert boxes.shape[-1] == 4
