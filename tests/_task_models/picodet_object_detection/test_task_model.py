#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._task_models.picodet_object_detection.train_model import (
    PicoDetObjectDetectionTrain,
    PicoDetObjectDetectionTrainArgs,
)
from lightly_train._task_models.picodet_object_detection.transforms import (
    PicoDetObjectDetectionTrainTransformArgs,
    PicoDetObjectDetectionValTransformArgs,
)


def test_load_train_state_dict__from_exported() -> None:
    model_args = PicoDetObjectDetectionTrainArgs()
    train_model = _create_train_model(model_args)
    task_model = train_model.model
    state_dict = train_model.get_export_state_dict()
    task_model.load_train_state_dict(state_dict)


def test_load_train_state_dict__no_ema_weights() -> None:
    model_args = PicoDetObjectDetectionTrainArgs()
    train_model = _create_train_model(model_args)
    task_model = train_model.model
    state_dict = train_model.state_dict()
    # Drop all EMA weights from the state dict. This is for backwards compatibility
    # with older checkpoints. The model should still be able to load the weights by
    # copying the non-EMA weights to the EMA model.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ema_model.")}
    task_model.load_train_state_dict(state_dict)


def _create_train_model(
    train_model_args: PicoDetObjectDetectionTrainArgs,
) -> PicoDetObjectDetectionTrain:
    train_model_args.resolve_auto(
        total_steps=1000,
        model_name="picodet/s-320",
        model_init_args={},
    )
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
    )
    train_transform_args = PicoDetObjectDetectionTrainTransformArgs()
    train_transform_args.resolve_auto(model_init_args={"image_size": (320, 320)})
    val_transform_args = PicoDetObjectDetectionValTransformArgs()
    val_transform_args.resolve_auto(model_init_args={"image_size": (320, 320)})

    train_model = PicoDetObjectDetectionTrain(
        model_name="picodet/s-320",
        model_args=train_model_args,
        data_args=data_args,
        train_transform_args=train_transform_args,
        val_transform_args=val_transform_args,
        load_weights=False,
    )
    return train_model
