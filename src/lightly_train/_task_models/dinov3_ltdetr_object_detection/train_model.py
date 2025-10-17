#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from lightly_train._task_models.dinov3_ltdetr_object_detection.task_model import (
    DINOv3LTDetrObjectDetectionTaskModel,
)
from lightly_train._task_models.train_model import TrainModel, TrainModelArgs


class DINOv3LTDetrObjectDetectionTrainModelArgs(TrainModelArgs):
    pass


class DINOv3LTDetrObjectDetectionTrainModel(TrainModel):
    task = "object_detection"
    train_model_args_cls = DINOv3LTDetrObjectDetectionTrainModelArgs
    task_model_cls = DINOv3LTDetrObjectDetectionTaskModel
