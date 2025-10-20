#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
from typing import Any, ClassVar

import torch
from lightning_fabric import Fabric
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._task_models.dinov2_ltdetr_object_detection.task_model import (
    DINOv2LTDetrObjectDetectionTaskModel,
)
from lightly_train._task_models.dinov2_ltdetr_object_detection.transforms import (
    DINOv2LTDetrObjectDetectionTrainTransform,
    DINOv2LTDetrObjectDetectionValTransform,
    DINOv2LTDetrObjectDetectionValTransformArgs,
)
from lightly_train._task_models.object_detection_components.matcher import (
    HungarianMatcher,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_criterion import (
    RTDETRCriterionv2,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train.types import ObjectDetectionBatch, PathLike


class DINOv2LTDetrObjectDetectionTrainModelArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = (
        100_000 // 16 * 72
    )  # TODO (Lionel, 10/25): Adjust default steps.

    backbone_weights: PathLike | None = None
    backbone_url: str = ""
    backbone_args: dict[str, Any] = {}


class DINOv2LTDetrObjectDetectionTrain(TrainModel):
    task = "object_detection"
    train_model_args_cls = DINOv2LTDetrObjectDetectionTrainModelArgs
    task_model_cls = DINOv2LTDetrObjectDetectionTaskModel
    train_transform_cls = DINOv2LTDetrObjectDetectionTrainTransform
    val_transform_cls = DINOv2LTDetrObjectDetectionValTransform

    def __init__(
        self,
        *,
        model_name: str,
        model_args: DINOv2LTDetrObjectDetectionTrainModelArgs,
        data_args: YOLOObjectDetectionDataArgs,
        val_transform_args: DINOv2LTDetrObjectDetectionValTransformArgs,
    ) -> None:
        super().__init__()
        self.model_args = model_args
        self.model = DINOv2LTDetrObjectDetectionTaskModel(
            model_name=model_name,
            image_size=val_transform_args.image_size,
            classes=data_args.names,
            image_normalize=None,  # TODO (Lionel, 10/25): Allow custom normalization.
            backbone_weights=model_args.backbone_weights,
            backbone_args=model_args.backbone_args,  # TODO (Lionel, 10/25): Potentially remove in accordance with EoMT.
        )

        matcher = HungarianMatcher(
            weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
            alpha=0.25,
            gamma=2.0,
        )

        self.criterion = RTDETRCriterionv2(
            matcher=matcher,
            weight_dict={"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2},
            losses=["vfl", "boxes"],
            alpha=0.75,
            gamma=2.0,
        )

    def set_train_mode(self) -> None:
        super().set_train_mode()
        self.criterion.train()  # TODO (Lionel, 10/25): Check if this is necessary.

    def training_step(
        self, fabric: Fabric, batch: ObjectDetectionBatch, step: int
    ) -> TaskStepResult:
        samples, boxes, classes = batch["image"], batch["bboxes"], batch["classes"]
        boxes = _yolo_to_xyxy(boxes)
        targets = [
            {"boxes": boxes, "labels": classes}
            for boxes, classes in zip(boxes, classes)
        ]
        outputs = self.model._forward_train(
            x=samples,
            targets=targets,
        )
        loss_dict = self.criterion(
            outputs=outputs, targets=targets, epoch=None, step=None, global_step=step, fabric=fabric
        )
        return TaskStepResult(
            loss=sum(loss_dict.values()),
            log_dict=loss_dict,
        )

    def validation_step(
        self, fabric: Fabric, batch: ObjectDetectionBatch, step: int
    ) -> TaskStepResult:
        raise NotImplementedError()

    def get_optimizer(self, total_steps: int) -> tuple[Optimizer, LRScheduler]:
        param_groups = [
            {
                "name": "backbone",
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if re.match(r"^(?=.*backbone)(?!.*norm).*$", n)
                ],
                "lr": 1e-5,
            },
            {
                "name": "detector",
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if re.match(r"^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", n)
                ],
                "weight_decay": 0.0,
            },
        ]
        optim = AdamW(
            param_groups,
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        scheduler = MultiStepLR(optimizer=optim, milestones=[1000], gamma=0.1)
        # warmup_scheduler = None
        return optim, scheduler

    def get_task_model(self) -> TaskModel:
        return self.model

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        pass


def _yolo_to_xyxy(boxes: list[Tensor]) -> list[Tensor]:
    """Convert bounding boxes from YOLO (normalized cx, cy, w, h) format to
    (normalized x_min, y_min, x_max, y_max) format.

    Args:
        boxes: Bounding boxes in YOLO format of shape (n_boxes, 4) with values
            normalized between 0 and 1.

    Returns:
        Bounding boxes in (normalized x_min, y_min, x_max, y_max) format.
    """
    converted_boxes = []
    for box in boxes:
        x_c, y_c, w, h = box.unbind(-1)
        x_min = x_c - 0.5 * w
        y_min = y_c - 0.5 * h
        x_max = x_c + 0.5 * w
        y_max = y_c + 0.5 * h
        converted_box = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        converted_boxes.append(converted_box)
    return converted_boxes
