#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import ClassVar, Literal

from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
from lightly_train._task_models.dinov3_ltdetr_object_detection.task_model import (
    DINOv3LTDetrObjectDetectionTaskModel,
)
from lightly_train._task_models.train_model import TrainModel, TrainModelArgs


class DINOv3LTDetrObjectDetectionTaskSaveCheckpointArgs(TaskSaveCheckpointArgs):
    watch_metric: str = "val_metric/mAP_50-95"
    mode: Literal["min", "max"] = "max"


class DINOv3LTDetrObjectDetectionTrainModelArgs(TrainModelArgs):
    save_checkpoint_args_cls: ClassVar[type[TaskSaveCheckpointArgs]] = (
        DINOv3LTDetrObjectDetectionTaskSaveCheckpointArgs
    )


class DINOv3LTDetrObjectDetectionTrainModel(TrainModel):
    task = "object_detection"
    train_model_args_cls = DINOv3LTDetrObjectDetectionTrainModelArgs
    task_model_cls = DINOv3LTDetrObjectDetectionTaskModel
