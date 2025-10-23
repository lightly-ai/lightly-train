#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import re
from dataclasses import field
from typing import Any, ClassVar, Literal

import torch
from lightning_fabric import Fabric
from torch import Tensor
from torch.optim import AdamW, Optimizer  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._distributed import reduce_dict
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
from lightly_train._task_models.dinov3_ltdetr_object_detection.task_model import (
    DINOv3LTDETRObjectDetectionTaskModel,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.transforms import (
    DINOv3LTDETRObjectDetectionTrainTransform,
    DINOv3LTDETRObjectDetectionValTransform,
    DINOv3LTDETRObjectDetectionValTransformArgs,
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


class DINOv3LTDETRObjectDetectionTaskSaveCheckpointArgs(TaskSaveCheckpointArgs):
    watch_metric: str = "val_total_loss"
    mode: Literal["min", "max"] = "min"


class DINOv3LTDETRObjectDetectionTrainModelArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = (
        100_000 // 16 * 72
    )  # TODO (Lionel, 10/25): Adjust default steps.

    save_checkpoint_args_cls: ClassVar[type[TaskSaveCheckpointArgs]] = (
        DINOv3LTDETRObjectDetectionTaskSaveCheckpointArgs
    )

    backbone_weights: PathLike | None = None
    backbone_url: str = ""
    backbone_args: dict[str, Any] = {}

    # Matcher configuration
    matcher_weight_dict: dict[str, float] = field(
        default_factory=lambda: {"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}
    )
    matcher_use_focal_loss: bool = True
    matcher_alpha: float = 0.25
    matcher_gamma: float = 2.0

    # Criterion configuration
    criterion_weight_dict: dict[str, float] = field(
        default_factory=lambda: {"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2}
    )
    criterion_losses: list[str] = field(default_factory=lambda: ["vfl", "boxes"])
    criterion_alpha: float = 0.75
    criterion_gamma: float = 2.0

    # Miscellaneous
    clip_max_norm: float = 0.1

    # Optimizer configuration
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_weight_decay: float = 1e-4

    # Per-parameter-group overrides
    backbone_lr: float = 1e-6
    backbone_weight_decay: float | None = None  # Use default if None

    detector_weight_decay: float = 0.0

    # Scheduler configuration
    scheduler_milestones: list[int] = field(default_factory=lambda: [1000])
    scheduler_gamma: float = 0.1
    scheduler_warmup_steps: int | None = (
        None  # TODO (Thomas, 10/25): Change to flat-cosine with warmup.
    )


class DINOv3LTDETRObjectDetectionTrain(TrainModel):
    task = "object_detection"
    train_model_args_cls = DINOv3LTDETRObjectDetectionTrainModelArgs
    task_model_cls = DINOv3LTDETRObjectDetectionTaskModel
    train_transform_cls = DINOv3LTDETRObjectDetectionTrainTransform
    val_transform_cls = DINOv3LTDETRObjectDetectionValTransform
    save_checkpoint_args_cls = DINOv3LTDETRObjectDetectionTaskSaveCheckpointArgs

    def __init__(
        self,
        *,
        model_name: str,
        model_args: DINOv3LTDETRObjectDetectionTrainModelArgs,
        data_args: YOLOObjectDetectionDataArgs,
        val_transform_args: DINOv3LTDETRObjectDetectionValTransformArgs,
    ) -> None:
        super().__init__()
        self.model_args = model_args
        self.model = DINOv3LTDETRObjectDetectionTaskModel(
            model_name=model_name,
            image_size=val_transform_args.image_size,
            classes=data_args.names,
            image_normalize=None,  # TODO (Lionel, 10/25): Allow custom normalization.
            backbone_weights=model_args.backbone_weights,
            backbone_args=model_args.backbone_args,  # TODO (Lionel, 10/25): Potentially remove in accordance with EoMT.
        )

        matcher = HungarianMatcher(  # type: ignore[no-untyped-call]
            weight_dict=model_args.matcher_weight_dict,
            use_focal_loss=model_args.matcher_use_focal_loss,
            alpha=model_args.matcher_alpha,
            gamma=model_args.matcher_gamma,
        )

        self.criterion = RTDETRCriterionv2(  # type: ignore[no-untyped-call]
            matcher=matcher,
            weight_dict=model_args.criterion_weight_dict,
            losses=model_args.criterion_losses,
            alpha=model_args.criterion_alpha,
            gamma=model_args.criterion_gamma,
        )

        self.clip_max_norm = model_args.clip_max_norm

    def set_train_mode(self) -> None:
        super().set_train_mode()
        self.criterion.train()  # TODO (Lionel, 10/25): Check if this is necessary.

    def training_step(
        self, fabric: Fabric, batch: ObjectDetectionBatch, step: int
    ) -> TaskStepResult:
        samples, boxes, classes = batch["image"], batch["bboxes"], batch["classes"]
        boxes = _yolo_to_xyxy(boxes)
        targets: list[dict[str, Tensor]] = [
            {"boxes": boxes, "labels": classes}
            for boxes, classes in zip(boxes, classes)
        ]
        outputs = self.model._forward_train(
            x=samples,
            targets=targets,
        )
        # Additional kwargs are anyway ignore in RTDETRCriterionv2.
        loss_dict = self.criterion(
            outputs=outputs,
            targets=targets,
            epoch=None,
            step=None,
            global_step=None,
            world_size=fabric.world_size,
        )
        total_loss = sum(loss_dict.values())

        # Average loss dict across devices.
        loss_dict = reduce_dict(loss_dict)

        return TaskStepResult(
            loss=total_loss,
            log_dict={**{"train_total_loss": total_loss.item()}, **loss_dict},
        )

    def validation_step(
        self,
        fabric: Fabric,
        batch: ObjectDetectionBatch,
    ) -> TaskStepResult:
        samples, boxes, classes = batch["image"], batch["bboxes"], batch["classes"]
        boxes = _yolo_to_xyxy(boxes)
        targets = [
            {"boxes": boxes, "labels": classes}
            for boxes, classes in zip(boxes, classes)
        ]
        with torch.no_grad():
            outputs = self.model._forward_train(
                x=samples,
                targets=targets,
            )
            # TODO (Lionel, 10/25): Pass epoch, step, global_step.
            loss_dict = self.criterion(
                outputs=outputs,
                targets=targets,
                epoch=None,
                step=None,
                global_step=None,
                world_size=fabric.world_size,
            )

        total_loss = sum(loss_dict.values())

        # Average loss dict across devices.
        loss_dict = reduce_dict(loss_dict)

        return TaskStepResult(
            loss=total_loss,
            log_dict={**{"val_total_loss": total_loss.item()}, **loss_dict},
        )

    def get_optimizer(self, total_steps: int) -> tuple[Optimizer, LRScheduler]:
        # TODO (Thomas, 10/25): Update groups as done for DINOv3 backbones.
        param_groups = [
            {
                "name": "backbone",
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if re.match(r"^(?=.*backbone)(?!.*norm).*$", n)
                ],
                "lr": self.model_args.backbone_lr,
            },
            {
                "name": "detector",
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if re.match(r"^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", n)
                ],
                "weight_decay": self.model_args.detector_weight_decay,
            },
        ]
        optim = AdamW(
            param_groups,
            lr=self.model_args.optimizer_lr,
            betas=self.model_args.optimizer_betas,
            weight_decay=self.model_args.optimizer_weight_decay,
        )
        scheduler = MultiStepLR(
            optimizer=optim,
            milestones=self.model_args.scheduler_milestones,
            gamma=self.model_args.scheduler_gamma,
        )
        # TODO (Lionel, 10/25): Use the warmup scheduler.
        return optim, scheduler

    def get_task_model(self) -> TaskModel:
        return self.model

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        if self.clip_max_norm > 0:
            fabric.clip_gradients(
                self.model,
                optimizer=optimizer,
                max_norm=self.clip_max_norm,
            )


def _yolo_to_xyxy(batch_boxes: list[Tensor]) -> list[Tensor]:
    """Convert bounding boxes from YOLO (normalized cx, cy, w, h) format to
    (normalized x_min, y_min, x_max, y_max) format.

    Args:
        boxes: Bounding boxes in YOLO format of shape (n_boxes, 4) with values
            normalized between 0 and 1.

    Returns:
        Bounding boxes in (normalized x_min, y_min, x_max, y_max) format.
    """
    converted_boxes = []
    for sample_boxes in batch_boxes:
        cxcywh = sample_boxes
        x_min = cxcywh[:, 0] - cxcywh[:, 2] / 2
        y_min = cxcywh[:, 1] - cxcywh[:, 3] / 2
        x_max = cxcywh[:, 0] + cxcywh[:, 2] / 2
        y_max = cxcywh[:, 1] + cxcywh[:, 3] / 2
        converted_boxes.append(torch.stack([x_min, y_min, x_max, y_max], dim=-1))
    return converted_boxes
