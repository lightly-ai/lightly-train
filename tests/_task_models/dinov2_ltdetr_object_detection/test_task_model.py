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

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._task_models.dinov2_ltdetr_object_detection.train_model import (
    DINOv2LTDETRObjectDetectionTrain,
    DINOv2LTDETRObjectDetectionTrainArgs,
)
from lightly_train._task_models.dinov2_ltdetr_object_detection.transforms import (
    DINOv2LTDETRObjectDetectionTrainTransformArgs,
    DINOv2LTDETRObjectDetectionValTransformArgs,
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


def _create_train_model(
    train_model_args: DINOv2LTDETRObjectDetectionTrainArgs,
) -> DINOv2LTDETRObjectDetectionTrain:
    train_model_args.resolve_auto(
        total_steps=1000,
        model_name="dinov2/_vittest14-ltdetr",
        model_init_args={},
    )
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
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
        load_weights=False,
    )
    return train_model
