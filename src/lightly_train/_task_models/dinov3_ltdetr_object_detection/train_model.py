#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
import re
from dataclasses import field
from typing import Any, ClassVar, Literal

import torch
from lightning_fabric import Fabric
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import AdamW, Optimizer  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import (  # type: ignore[attr-defined]
    LinearLR,
    LRScheduler,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from lightly_train._configs.validate import no_auto
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._distributed import reduce_dict
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
from lightly_train._task_models.dinov3_ltdetr_object_detection.task_model import (
    DINOv3LTDETRObjectDetection,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.transforms import (
    DINOv3LTDETRObjectDetectionTrainTransform,
    DINOv3LTDETRObjectDetectionTrainTransformArgs,
    DINOv3LTDETRObjectDetectionValTransform,
    DINOv3LTDETRObjectDetectionValTransformArgs,
)
from lightly_train._task_models.object_detection_components.ema import ModelEMA
from lightly_train._task_models.object_detection_components.matcher import (
    HungarianMatcher,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_criterion import (
    RTDETRCriterionv2,
)
from lightly_train._task_models.object_detection_components.utils import (
    _denormalize_xyxy_boxes,
    _yolo_to_xyxy,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train.types import ObjectDetectionBatch, PathLike


class DINOv3LTDETRObjectDetectionTaskSaveCheckpointArgs(TaskSaveCheckpointArgs):
    watch_metric: str = "val_metric/map"
    mode: Literal["min", "max"] = "max"


class DINOv3LTDETRObjectDetectionTrainArgs(TrainModelArgs):
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

    use_ema_model: bool = True
    ema_momentum: float = 0.9999
    ema_warmup_steps: int = 2000

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
    scheduler_start_factor: float = 0.01
    scheduler_warmup_steps: int = 2000


class DINOv3LTDETRObjectDetectionTrain(TrainModel):
    task = "object_detection"
    train_model_args_cls = DINOv3LTDETRObjectDetectionTrainArgs
    task_model_cls = DINOv3LTDETRObjectDetection
    train_transform_cls = DINOv3LTDETRObjectDetectionTrainTransform
    val_transform_cls = DINOv3LTDETRObjectDetectionValTransform
    save_checkpoint_args_cls = DINOv3LTDETRObjectDetectionTaskSaveCheckpointArgs

    def __init__(
        self,
        *,
        model_name: str,
        model_args: DINOv3LTDETRObjectDetectionTrainArgs,
        data_args: YOLOObjectDetectionDataArgs,
        train_transform_args: DINOv3LTDETRObjectDetectionTrainTransformArgs,
        val_transform_args: DINOv3LTDETRObjectDetectionValTransformArgs,
        load_weights: bool,
    ) -> None:
        super().__init__()

        self.model_args = model_args

        # Get the normalization.
        normalize = no_auto(val_transform_args.normalize)
        normalize_dict: dict[str, Any] | None
        if normalize is None:
            normalize_dict = None
        else:
            normalize_dict = normalize.model_dump()

        self.model_state_dict_key_prefix = "model."
        self.model = DINOv3LTDETRObjectDetection(
            model_name=model_name,
            image_size=no_auto(val_transform_args.image_size),
            classes=data_args.included_classes,
            image_normalize=normalize_dict,
            backbone_args=model_args.backbone_args,  # TODO (Lionel, 10/25): Potentially remove in accordance with EoMT.
            backbone_weights=model_args.backbone_weights,
            load_weights=load_weights,
        )

        self.ema_model_state_dict_key_prefix = "ema_model."
        self.ema_model: ModelEMA | None = None
        if model_args.use_ema_model:
            self.ema_model = ModelEMA(
                model=self.model,
                decay=model_args.ema_momentum,
                warmups=model_args.ema_warmup_steps,
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
            num_classes=len(data_args.included_classes),
        )

        self.clip_max_norm = model_args.clip_max_norm

        # Validation metric.
        self.map_metric = MeanAveragePrecision()
        self.map_metric.warn_on_many_detections = False

    def load_train_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Loads the model state dict.

        Overloads the default implementation to use the task model's loading logic.
        This allows loading weights from an EMA model into the training model.
        """
        missing_keys, unexpected_keys = self.model.load_train_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )
        if self.ema_model is not None:
            missing_keys_ema, unexpected_keys_ema = (
                self.ema_model.model.load_train_state_dict(  # type: ignore
                    # Copy to avoid assigning the same weights to both models
                    copy.deepcopy(state_dict),
                    strict=strict,
                    assign=assign,
                )
            )
            missing_keys.extend(missing_keys_ema)
            unexpected_keys.extend(unexpected_keys_ema)
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def get_export_state_dict(self) -> dict[str, Any]:
        """Returns the state dict for exporting."""
        state_dict = super().get_export_state_dict()
        if self.ema_model is not None:
            # Only keep EMA weights for export
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k.startswith(self.ema_model_state_dict_key_prefix)
            }
        return state_dict

    def set_train_mode(self) -> None:
        super().set_train_mode()
        self.criterion.train()  # TODO (Lionel, 10/25): Check if this is necessary.

    def training_step(
        self, fabric: Fabric, batch: ObjectDetectionBatch, step: int
    ) -> TaskStepResult:
        samples, boxes, classes = batch["image"], batch["bboxes"], batch["classes"]
        targets: list[dict[str, Tensor]] = [
            {"boxes": boxes, "labels": classes}
            for boxes, classes in zip(boxes, classes)
        ]
        outputs = self.model._forward_train(
            x=samples,
            targets=targets,
        )

        # Additional kwargs are anyway ignored in RTDETRCriterionv2.
        # The loss expects gt boxes in cxcywh format normalized in [0,1].
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
            log_dict={
                "train_loss": total_loss.item(),
                "train_loss/loss_vfl": loss_dict["loss_vfl"],
                "train_loss/loss_bbox": loss_dict["loss_bbox"],
                "train_loss/loss_giou": loss_dict["loss_giou"],
            },
        )

    def on_train_batch_end(self) -> None:
        if self.ema_model is not None:
            self.ema_model.update(self.model)

    def validation_step(
        self,
        fabric: Fabric,
        batch: ObjectDetectionBatch,
    ) -> TaskStepResult:
        samples, boxes, classes, orig_target_sizes = (
            batch["image"],
            batch["bboxes"],
            batch["classes"],
            batch["original_size"],
        )
        targets = [
            {"boxes": boxes, "labels": classes}
            for boxes, classes in zip(boxes, classes)
        ]

        if self.ema_model is not None:
            model_to_use = self.ema_model.model
        else:
            model_to_use = self.model

        with torch.no_grad():
            outputs = model_to_use._forward_train(  # type: ignore[operator]
                x=samples,
                targets=targets,
            )
            # TODO (Lionel, 10/25): Pass epoch, step, global_step.
            # The loss expects gt boxes in cxcywh format normalized in [0,1].
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

        # Convert to xyxy format and de-normalize the boxes.
        boxes = _yolo_to_xyxy(boxes)
        boxes_denormalized = _denormalize_xyxy_boxes(boxes, orig_target_sizes)
        for target, sample_denormalized_boxes in zip(targets, boxes_denormalized):
            target["boxes"] = sample_denormalized_boxes

        orig_target_sizes_tensor = torch.tensor(
            orig_target_sizes, device=samples.device
        )
        results = self.model.postprocessor(
            outputs, orig_target_sizes=orig_target_sizes_tensor
        )

        # Update mAP metric
        self.map_metric.update(results, targets)

        metrics: dict[str, Any] = {
            "val_metric/map": self.map_metric,
        }

        return TaskStepResult(
            loss=total_loss,
            log_dict={
                "val_loss": total_loss.item(),
                "val_loss/loss_vfl": loss_dict["loss_vfl"],
                "val_loss/loss_bbox": loss_dict["loss_bbox"],
                "val_loss/loss_giou": loss_dict["loss_giou"],
                **metrics,
            },
        )

    def get_optimizer(self, total_steps: int) -> tuple[Optimizer, LRScheduler]:
        # TODO (Thomas, 10/25): Update groups as done for DINOv3 backbones.
        param_groups = []
        base_weight_decay = self.model_args.optimizer_weight_decay
        base_lr = self.model_args.optimizer_lr
        backbone_lr = self.model_args.backbone_lr
        detector_weight_decay = self.model_args.detector_weight_decay

        # Backbone group without normalization layers.
        backbone_params = [
            p
            for n, p in self.model.named_parameters()
            if re.match(r"^(?=.*backbone)(?!.*(norm|bn)).*$", n)
        ]
        if backbone_params:
            param_groups.append(
                {
                    "name": "backbone",
                    "params": backbone_params,
                    "lr": backbone_lr,
                    "weight_decay": base_weight_decay,
                }
            )

        # Normalization layers from the detector (encoder/decoder).
        # TODO (Thomas, 10/25): Add one extra group for the normalization layers in the backbone.
        detector_params = [
            p
            for n, p in self.model.named_parameters()
            if re.match(r"^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", n)
        ]
        if detector_params:
            param_groups.append(
                {
                    "name": "detector",
                    "params": detector_params,
                    "lr": base_lr,
                    "weight_decay": detector_weight_decay,
                }
            )

        # Default group for all remaining parameters.
        used_params = set(
            p
            for param_group in param_groups
            for p in param_group["params"]  # type: ignore[attr-defined]
        )
        default_params = [
            p for _, p in self.model.named_parameters() if p not in used_params
        ]

        if default_params:
            param_groups.append(
                {
                    "name": "default",
                    "params": default_params,
                    "lr": base_lr,
                    "weight_decay": base_weight_decay,
                }
            )
        optim = AdamW(
            param_groups,
            lr=self.model_args.optimizer_lr,
            betas=self.model_args.optimizer_betas,
            weight_decay=self.model_args.optimizer_weight_decay,
        )
        # TODO (Thomas, 11/25): Change to flat-cosine with warmup.
        scheduler = LinearLR(
            optimizer=optim,
            total_iters=self.model_args.scheduler_warmup_steps,
            start_factor=self.model_args.scheduler_start_factor,
        )

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
