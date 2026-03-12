#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning_fabric import Fabric
from matplotlib.patches import Rectangle
from pydantic import AliasChoices, Field
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import AdamW, Optimizer  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import (  # type: ignore[attr-defined]
    LinearLR,
    LRScheduler,
)

from lightly_train._configs.validate import no_auto
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._distributed import reduce_dict
from lightly_train._metrics.detection.task_metric import (
    ObjectDetectionTaskMetric,
    ObjectDetectionTaskMetricArgs,
)
from lightly_train._optim import optimizer_helpers
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


class DINOv3LTDETRObjectDetectionTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = (
        100_000 // 16 * 72
    )  # TODO (Lionel, 10/25): Adjust default steps.

    backbone_weights: PathLike | None = None
    backbone_url: str = ""
    backbone_args: dict[str, Any] = {}
    backbone_freeze: bool = False

    use_ema_model: bool = True
    ema_momentum: float = 0.9999
    ema_warmup_steps: int = 2000

    # Matcher configuration
    matcher_weight_dict: dict[str, float] = Field(
        default_factory=lambda: {"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0}
    )
    matcher_use_focal_loss: bool = True
    matcher_alpha: float = 0.25
    matcher_gamma: float = 2.0

    # Criterion configuration
    loss_weight_dict: dict[str, float] = Field(
        default_factory=lambda: {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    )
    losses: list[str] = Field(default_factory=lambda: ["vfl", "boxes"])
    loss_alpha: float = 0.75
    loss_gamma: float = 2.0

    # Miscellaneous
    gradient_clip_val: float = 0.1

    # Optimizer configuration
    lr: float = Field(
        default=1e-4,
        validation_alias=AliasChoices("lr", "optimizer_lr"),
    )
    weight_decay: float = Field(
        default=1e-4,
        validation_alias=AliasChoices("weight_decay", "optimizer_weight_decay"),
    )
    optimizer_betas: tuple[float, float] = (0.9, 0.999)

    # Per-parameter-group overrides
    backbone_lr_factor: float = 1e-2
    backbone_weight_decay: float | None = None  # Use default if None

    detector_weight_decay: float = 0.0

    # Scheduler configuration
    scheduler_start_factor: float = 0.01
    lr_warmup_steps: int = Field(
        default=2000,
        validation_alias=AliasChoices("lr_warmup_steps", "scheduler_warmup_steps"),
    )


class DINOv3LTDETRObjectDetectionTrain(TrainModel):
    task = "object_detection"
    train_model_args_cls = DINOv3LTDETRObjectDetectionTrainArgs
    task_metric_args_cls = ObjectDetectionTaskMetricArgs
    task_model_cls = DINOv3LTDETRObjectDetection
    train_transform_cls = DINOv3LTDETRObjectDetectionTrainTransform
    val_transform_cls = DINOv3LTDETRObjectDetectionValTransform

    def __init__(
        self,
        *,
        model_name: str,
        model_args: DINOv3LTDETRObjectDetectionTrainArgs,
        data_args: YOLOObjectDetectionDataArgs,
        train_transform_args: DINOv3LTDETRObjectDetectionTrainTransformArgs,
        val_transform_args: DINOv3LTDETRObjectDetectionValTransformArgs,
        load_weights: bool,
        metric_args: ObjectDetectionTaskMetricArgs,
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

        self.model = DINOv3LTDETRObjectDetection(
            model_name=model_name,
            image_size=no_auto(val_transform_args.image_size),
            classes=data_args.included_classes,
            image_normalize=normalize_dict,
            backbone_freeze=model_args.backbone_freeze,
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
            weight_dict=model_args.loss_weight_dict,
            losses=model_args.losses,
            alpha=model_args.loss_alpha,
            gamma=model_args.loss_gamma,
            num_classes=len(data_args.included_classes),
        )

        self.clip_max_norm = model_args.gradient_clip_val

        self._normalize_mean: list[float] | None = None
        self._normalize_std: list[float] | None = None
        if normalize_dict is not None:
            self._normalize_mean = normalize_dict["mean"]
            self._normalize_std = normalize_dict["std"]

        class_names = list(data_args.included_classes.values())
        self.loss_names = ["loss", "loss_vfl", "loss_bbox", "loss_giou"]
        self.metric_args = metric_args
        self.train_metrics = ObjectDetectionTaskMetric(
            task_metric_args=metric_args,
            split="train",
            class_names=class_names,
            box_format="xyxy",
            loss_names=self.loss_names,
        )
        self.val_metrics = ObjectDetectionTaskMetric(
            task_metric_args=metric_args,
            split="val",
            class_names=class_names,
            box_format="xyxy",
            loss_names=self.loss_names,
        )

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
        if self.model_args.backbone_freeze:
            self.model.freeze_backbone()

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

        # Metrics
        self.train_metrics.update_with_losses(
            loss_dict={
                "loss": total_loss.detach(),
                "loss_vfl": loss_dict["loss_vfl"].detach(),
                "loss_bbox": loss_dict["loss_bbox"].detach(),
                "loss_giou": loss_dict["loss_giou"].detach(),
            },
            weight=samples.shape[0],
        )
        if self.metric_args.train:
            orig_target_sizes = batch["original_size"]
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
            self.train_metrics.update_with_predictions(results, targets)

        return TaskStepResult(
            loss=total_loss,
            log_dict={},
            metrics=self.train_metrics,
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

        # Metrics
        self.val_metrics.update_with_losses(
            loss_dict={
                "loss": total_loss.detach(),
                "loss_vfl": loss_dict["loss_vfl"].detach(),
                "loss_bbox": loss_dict["loss_bbox"].detach(),
                "loss_giou": loss_dict["loss_giou"].detach(),
            },
            weight=samples.shape[0],
        )
        self.val_metrics.update_with_predictions(results, targets)

        return TaskStepResult(
            loss=total_loss,
            log_dict={},
            metrics=self.val_metrics,
        )

    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
    ) -> tuple[Optimizer, LRScheduler]:
        _, params_no_wd_list = optimizer_helpers.get_weight_decay_parameters(
            modules=[self.model]
        )
        params_no_wd = set(params_no_wd_list)

        param_groups = []
        lr = self.model_args.lr * math.sqrt(
            global_batch_size / self.model_args.default_batch_size
        )
        backbone_lr = lr * self.model_args.backbone_lr_factor
        backbone_weight_decay = (
            self.model_args.backbone_weight_decay
            if self.model_args.backbone_weight_decay is not None
            else self.model_args.weight_decay
        )
        detector_weight_decay = self.model_args.detector_weight_decay

        backbone_params = list(self.model.backbone.parameters())
        backbone_params_wd = [p for p in backbone_params if p not in params_no_wd]
        backbone_params_no_wd = [p for p in backbone_params if p in params_no_wd]
        if backbone_params_wd:
            param_groups.append(
                {
                    "name": "backbone",
                    "params": backbone_params_wd,
                    "lr": backbone_lr,
                    "weight_decay": backbone_weight_decay,
                }
            )
        if backbone_params_no_wd:
            param_groups.append(
                {
                    "name": "backbone_no_wd",
                    "params": backbone_params_no_wd,
                    "lr": backbone_lr,
                    "weight_decay": 0.0,
                }
            )

        detector_params = list(self.model.encoder.parameters()) + list(
            self.model.decoder.parameters()
        )
        detector_params_wd = [p for p in detector_params if p not in params_no_wd]
        detector_params_no_wd = [p for p in detector_params if p in params_no_wd]
        if detector_params_wd:
            param_groups.append(
                {
                    "name": "detector",
                    "params": detector_params_wd,
                    "weight_decay": detector_weight_decay,
                }
            )
        if detector_params_no_wd:
            param_groups.append(
                {
                    "name": "detector_no_wd",
                    "params": detector_params_no_wd,
                    "weight_decay": 0.0,
                }
            )

        # Default group for all remaining parameters.
        used_params = set(backbone_params + detector_params)
        default_params = [p for p in self.model.parameters() if p not in used_params]
        default_params_wd = [p for p in default_params if p not in params_no_wd]
        default_params_no_wd = [p for p in default_params if p in params_no_wd]
        if default_params_wd:
            param_groups.append(
                {
                    "name": "default",
                    "params": default_params_wd,
                }
            )
        if default_params_no_wd:
            param_groups.append(
                {
                    "name": "default_no_wd",
                    "params": default_params_no_wd,
                    "weight_decay": 0.0,
                }
            )

        optim = AdamW(
            param_groups,
            lr=lr,
            betas=self.model_args.optimizer_betas,
            weight_decay=self.model_args.weight_decay,
        )
        # TODO (Thomas, 11/25): Change to flat-cosine with warmup.
        scheduler = LinearLR(
            optimizer=optim,
            total_iters=self.model_args.lr_warmup_steps,
            start_factor=self.model_args.scheduler_start_factor,
        )

        return optim, scheduler

    def get_task_model(self) -> TaskModel:
        return self.model

    def save_labeled_images(
        self,
        *,
        batch: ObjectDetectionBatch,
        step: int,
        output_dir: Path,
        split: str,
    ) -> None:
        images = batch["image"]  # (B, C, H, W)
        bboxes = batch["bboxes"]  # GT boxes, normalized cxcywh.
        classes = batch["classes"]  # GT class ids.

        # Run inference to get predicted boxes.
        model_to_use = (
            self.ema_model.model if self.ema_model is not None else self.model
        )
        with torch.no_grad():
            targets = [
                {"boxes": gt_boxes, "labels": gt_cls}
                for gt_boxes, gt_cls in zip(bboxes, classes)
            ]
            outputs = model_to_use._forward_train(  # type: ignore[operator]
                x=images, targets=targets
            )
        # Use the model input size so predicted box coordinates align with the
        # displayed image (img_np), which is the resized model input, not the
        # original image.
        input_h, input_w = images.shape[-2:]
        input_sizes_tensor = torch.tensor(
            [[input_w, input_h]] * len(images), device=images.device
        )
        predictions = self.model.postprocessor(
            outputs, orig_target_sizes=input_sizes_tensor
        )

        # Build a stable color palette: one distinct color per class id.
        class_ids = sorted(self.model.classes.keys())
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        class_colors = {
            cls_id: color_cycle[idx % len(color_cycle)]
            for idx, cls_id in enumerate(class_ids)
        }

        n = min(4, len(images))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Decode images once for reuse in both figures.
        imgs_np: list[np.ndarray[Any, Any]] = []
        for i in range(n):
            img = images[i].cpu().float()
            if self._normalize_mean is not None and self._normalize_std is not None:
                mean = torch.tensor(self._normalize_mean, dtype=torch.float32).view(
                    -1, 1, 1
                )
                std = torch.tensor(self._normalize_std, dtype=torch.float32).view(
                    -1, 1, 1
                )
                img = img * std + mean
            imgs_np.append(img.clamp(0, 1).permute(1, 2, 0).numpy())

        # --- GT label figure ---
        fig_gt, axes_gt = plt.subplots(2, 2, figsize=(12, 12))
        for i in range(4):
            ax = axes_gt[i // 2][i % 2]
            if i < n:
                img_np = imgs_np[i]
                h, w = img_np.shape[:2]
                ax.imshow(img_np)
                for box, cls_id in zip(bboxes[i], classes[i]):
                    cx, cy, bw, bh = box.tolist()
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    cls_id_int = int(cls_id.item())
                    color = class_colors.get(cls_id_int, color_cycle[0])
                    rect = Rectangle(
                        xy=(x1, y1),
                        width=bw * w,
                        height=bh * h,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    cls_name = self.model.classes.get(cls_id_int, str(cls_id_int))
                    ax.text(
                        x1,
                        y1 - 2,
                        cls_name,
                        color=color,
                        fontsize=8,
                        verticalalignment="bottom",
                    )
            ax.axis("off")
        fig_gt.savefig(
            output_dir / f"{split}_label{step}.jpg", bbox_inches="tight", dpi=150
        )
        plt.close(fig_gt)

        # --- Prediction figure ---
        fig_pred, axes_pred = plt.subplots(2, 2, figsize=(12, 12))
        for i in range(4):
            ax = axes_pred[i // 2][i % 2]
            if i < n:
                img_np = imgs_np[i]
                ax.imshow(img_np)
                pred = predictions[i]
                for box, cls_id, score in zip(
                    pred["boxes"], pred["labels"], pred["scores"]
                ):
                    if score < 0.5:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    cls_id_int = int(cls_id.item())
                    color = class_colors.get(cls_id_int, color_cycle[0])
                    rect = Rectangle(
                        xy=(x1, y1),
                        width=x2 - x1,
                        height=y2 - y1,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    cls_name = self.model.classes.get(cls_id_int, str(cls_id_int))
                    ax.text(
                        x1,
                        y1 - 2,
                        f"{cls_name} {score.item():.2f}",
                        color=color,
                        fontsize=8,
                        verticalalignment="bottom",
                    )
            ax.axis("off")
        fig_pred.savefig(
            output_dir / f"{split}_predict{step}.jpg", bbox_inches="tight", dpi=150
        )
        plt.close(fig_pred)

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        if self.clip_max_norm > 0:
            fabric.clip_gradients(
                self.model,
                optimizer=optimizer,
                max_norm=self.clip_max_norm,
            )
