#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.train_model import (
    DINOv3LTDETRObjectDetectionTrain,
    DINOv3LTDETRObjectDetectionTrainArgs,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.transforms import (
    DINOv3LTDETRObjectDetectionTrainTransformArgs,
    DINOv3LTDETRObjectDetectionValTransformArgs,
)


@pytest.mark.parametrize("use_ema_model", [True, False])
def test_load_train_state_dict__from_exported(use_ema_model: bool) -> None:
    model_args = DINOv3LTDETRObjectDetectionTrainArgs(use_ema_model=use_ema_model)
    train_model = _create_train_model(model_args)
    task_model = train_model.model
    state_dict = train_model.get_export_state_dict()
    task_model.load_train_state_dict(state_dict)


def test_load_train_state_dict__no_ema_weights() -> None:
    model_args = DINOv3LTDETRObjectDetectionTrainArgs(use_ema_model=True)
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
    model_args = DINOv3LTDETRObjectDetectionTrainArgs(
        use_ema_model=True,
        backbone_freeze=should_freeze,
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
    train_model_args: DINOv3LTDETRObjectDetectionTrainArgs,
) -> DINOv3LTDETRObjectDetectionTrain:
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
    )
    train_model_args.resolve_auto(
        total_steps=1000,
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        model_init_args={},
        data_args=data_args,
    )
    train_transform_args = DINOv3LTDETRObjectDetectionTrainTransformArgs()
    train_transform_args.resolve_auto(model_init_args={})
    val_transform_args = DINOv3LTDETRObjectDetectionValTransformArgs()
    val_transform_args.resolve_auto(model_init_args={})

    train_model = DINOv3LTDETRObjectDetectionTrain(
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        model_args=train_model_args,
        data_args=data_args,
        train_transform_args=train_transform_args,
        val_transform_args=val_transform_args,
        load_weights=False,
    )
    return train_model
