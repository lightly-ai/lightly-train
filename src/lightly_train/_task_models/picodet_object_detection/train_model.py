#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
from typing import Any, ClassVar, Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightly.utils.scheduler import CosineWarmupScheduler
from lightning_fabric import Fabric
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from lightly_train._configs.validate import no_auto
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._distributed import reduce_dict
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
from lightly_train._task_models.object_detection_components.ema import ModelEMA
from lightly_train._task_models.object_detection_components.utils import (
    _denormalize_xyxy_boxes,
    _yolo_to_xyxy,
)
from lightly_train._task_models.picodet_object_detection.losses import (
    DistributionFocalLoss,
    GIoULoss,
    VarifocalLoss,
    box_iou_aligned,
)
from lightly_train._task_models.picodet_object_detection.pico_head import (
    Integral,
    bbox2distance,
    distance2bbox,
)
from lightly_train._task_models.picodet_object_detection.sim_ota_assigner import (
    SimOTAAssigner,
    TaskAlignedTop1Assigner,
)
from lightly_train._task_models.picodet_object_detection.task_model import (
    PicoDetObjectDetection,
)
from lightly_train._task_models.picodet_object_detection.transforms import (
    PicoDetObjectDetectionTrainTransform,
    PicoDetObjectDetectionTrainTransformArgs,
    PicoDetObjectDetectionValTransform,
    PicoDetObjectDetectionValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train.types import ObjectDetectionBatch, PathLike


class PicoDetObjectDetectionTaskSaveCheckpointArgs(TaskSaveCheckpointArgs):
    """Checkpoint saving configuration for PicoDet."""

    watch_metric: str = "val_metric/map"
    mode: Literal["min", "max"] = "max"


class PicoDetObjectDetectionTrainArgs(TrainModelArgs):
    """Training arguments for PicoDet-S.

    Args:
        backbone_weights: Optional path to backbone checkpoint to load.
        lr: Learning rate for SGD optimizer.
        momentum: Momentum for SGD optimizer.
        weight_decay: Weight decay for SGD optimizer.
        lr_warmup_steps: Number of warmup iterations with linear LR increase.
        warmup_ratio: Starting LR ratio during warmup (e.g., 0.1 = start at 10% of base LR).
        loss_vfl_weight: Weight for varifocal loss.
        loss_giou_weight: Weight for GIoU loss.
        loss_dfl_weight: Weight for distribution focal loss.
        loss_o2o_dfl_weight: Weight for the o2o DFL loss term.
        simota_center_radius: Center radius for SimOTA assignment.
        simota_candidate_topk: Top-k candidates for dynamic k in SimOTA.
        simota_iou_weight: IoU weight in SimOTA cost matrix.
        ema_momentum: EMA momentum for model averaging.
        ema_warmup_steps: Warmup steps before applying EMA momentum.
    """

    default_batch_size: ClassVar[int] = 80
    default_steps: ClassVar[int] = 90_000
    save_checkpoint_args_cls: ClassVar[type[TaskSaveCheckpointArgs]] = (
        PicoDetObjectDetectionTaskSaveCheckpointArgs
    )

    backbone_weights: PathLike | None = None

    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 4e-5

    lr_warmup_steps: int = 300
    warmup_ratio: float = 0.1

    loss_vfl_weight: float = 1.0
    loss_giou_weight: float = 2.0
    loss_dfl_weight: float = 0.25
    loss_o2o_dfl_weight: float = 0.25

    simota_center_radius: float = 2.5
    simota_candidate_topk: int = 10
    simota_iou_weight: float = 6.0
    ema_momentum: float = 0.9998
    ema_warmup_steps: int = 2000


class PicoDetObjectDetectionTrain(TrainModel):
    """Training implementation for PicoDet-S.

    This class wraps the PicoDetObjectDetection task model and implements
    the training and validation steps with SimOTA assignment and
    VFL + GIoU + DFL losses.
    """

    task: ClassVar[str] = "object_detection"
    train_model_args_cls = PicoDetObjectDetectionTrainArgs
    task_model_cls = PicoDetObjectDetection
    train_transform_cls = PicoDetObjectDetectionTrainTransform
    val_transform_cls = PicoDetObjectDetectionValTransform

    def __init__(
        self,
        *,
        model_name: str,
        model_args: PicoDetObjectDetectionTrainArgs,
        data_args: YOLOObjectDetectionDataArgs,
        train_transform_args: PicoDetObjectDetectionTrainTransformArgs,
        val_transform_args: PicoDetObjectDetectionValTransformArgs,
        load_weights: bool,
    ) -> None:
        super().__init__()
        self.model_args = model_args

        num_classes = len(data_args.included_classes)
        resolved_image_size = no_auto(val_transform_args.image_size)

        normalize = no_auto(val_transform_args.normalize)
        if normalize is None:
            image_normalize = None
        else:
            image_normalize = normalize.model_dump()

        self.model = PicoDetObjectDetection(
            model_name=model_name,
            image_size=resolved_image_size,
            num_classes=num_classes,
            classes=data_args.included_classes,
            image_normalize=image_normalize,
            backbone_weights=model_args.backbone_weights,
            load_weights=load_weights,
        )

        self.num_classes = num_classes
        self.strides = (8, 16, 32, 64)
        self.reg_max = self.model.head.reg_max

        self.vfl_loss = VarifocalLoss(alpha=0.75, gamma=2.0)
        self.dfl_loss = DistributionFocalLoss()
        self.giou_loss = GIoULoss()

        self.integral = Integral(self.reg_max)

        self.assigner = SimOTAAssigner(
            center_radius=model_args.simota_center_radius,
            candidate_topk=model_args.simota_candidate_topk,
            iou_weight=model_args.simota_iou_weight,
            cls_weight=1.0,
            num_classes=num_classes,
        )
        self.o2o_assigner = TaskAlignedTop1Assigner(alpha=0.5, beta=6.0)

        # EMA model setup (following LTDETR pattern for consistency)
        # EMA is always enabled
        self._ema_model_state_dict_key_prefix = "ema_model."
        self.ema_model: ModelEMA
        self.ema_model = ModelEMA(
            model=self.model,
            decay=model_args.ema_momentum,
            warmups=model_args.ema_warmup_steps,
        )

        self.map_metric = MeanAveragePrecision()
        self.map_metric.warn_on_many_detections = False

    def get_task_model(self) -> PicoDetObjectDetection:
        """Return the task model for inference/export.

        Returns the EMA model which is used for inference.
        """
        return self.ema_model.model  # type: ignore[return-value]

    def training_step(
        self,
        fabric: Fabric,
        batch: ObjectDetectionBatch,
        step: int,
    ) -> TaskStepResult:
        """Perform a training step following reference PicoDet implementation."""
        images = batch["image"]
        gt_bboxes_yolo = batch["bboxes"]
        gt_labels_list = batch["classes"]

        batch_size = images.shape[0]
        img_h, img_w = images.shape[-2:]

        outputs = self.model._forward_train(images)
        cls_scores_list = cast(list[Tensor], outputs["cls_scores"])
        bbox_preds_list = cast(list[Tensor], outputs["bbox_preds"])
        o2o_cls_scores = cast(list[Tensor], outputs["o2o_cls_scores"])
        o2o_bbox_preds = cast(list[Tensor], outputs["o2o_bbox_preds"])

        # Convert GT from YOLO format to pixel xyxy
        gt_boxes_xyxy_norm = _yolo_to_xyxy(gt_bboxes_yolo)
        sizes = [(img_w, img_h)] * batch_size
        gt_boxes_xyxy_list = _denormalize_xyxy_boxes(gt_boxes_xyxy_norm, sizes)

        dense_loss, loss_vfl, loss_giou, loss_dfl = self._compute_losses(
            fabric=fabric,
            cls_scores=cls_scores_list,
            bbox_preds=bbox_preds_list,
            gt_boxes_xyxy_list=gt_boxes_xyxy_list,
            gt_labels_list=gt_labels_list,
        )
        (
            o2o_loss,
            loss_o2o_obj,
            loss_o2o_cls,
            loss_o2o_box,
            loss_o2o_dfl,
            o2o_stats,
        ) = self._compute_o2o_losses(
            cls_scores=o2o_cls_scores,
            bbox_preds=o2o_bbox_preds,
            gt_boxes_xyxy_list=gt_boxes_xyxy_list,
            gt_labels_list=gt_labels_list,
            image_size=(img_h, img_w),
        )

        total_loss = o2o_loss + 0.5 * dense_loss

        loss_dict = reduce_dict(
            {
                "train_loss": total_loss,
                "train_loss/loss_o2o": o2o_loss,
                "train_loss/loss_o2o_obj": loss_o2o_obj,
                "train_loss/loss_o2o_cls": loss_o2o_cls,
                "train_loss/loss_o2o_box": loss_o2o_box,
                "train_loss/loss_o2o_dfl": loss_o2o_dfl,
                "train_loss/loss_dense": dense_loss,
                "train_loss/loss_vfl": loss_vfl,
                "train_loss/loss_giou": loss_giou,
                "train_loss/loss_dfl": loss_dfl,
                # Debug o2o stats; remove after investigation.
                "debug/o2o_num_pos": o2o_stats["o2o_num_pos"],
                "debug/o2o_mean_iou": o2o_stats["o2o_mean_iou"],
                "debug/o2o_cls_target_sum": o2o_stats["o2o_cls_target_sum"],
                "debug/o2o_gt_small": o2o_stats["o2o_gt_small"],
                "debug/o2o_gt_medium": o2o_stats["o2o_gt_medium"],
                "debug/o2o_gt_large": o2o_stats["o2o_gt_large"],
                "debug/o2o_gt_center_small": o2o_stats["o2o_gt_center_small"],
                "debug/o2o_gt_center_medium": o2o_stats["o2o_gt_center_medium"],
                "debug/o2o_gt_center_large": o2o_stats["o2o_gt_center_large"],
                "debug/o2o_gt_matched_small": o2o_stats["o2o_gt_matched_small"],
                "debug/o2o_gt_matched_medium": o2o_stats["o2o_gt_matched_medium"],
                "debug/o2o_gt_matched_large": o2o_stats["o2o_gt_matched_large"],
            }
        )

        return TaskStepResult(
            loss=total_loss,
            log_dict={k: v.item() for k, v in loss_dict.items()},
        )

    def on_train_batch_end(self) -> None:
        """Called at the end of each training batch."""
        self.ema_model.update(self.model)

    def validation_step(
        self,
        fabric: Fabric,
        batch: ObjectDetectionBatch,
    ) -> TaskStepResult:
        """Perform a validation step."""
        images = batch["image"]
        gt_bboxes_yolo = batch["bboxes"]
        gt_labels_list = batch["classes"]

        batch_size = images.shape[0]
        device = images.device

        # Use EMA model for validation
        model_to_use = self.ema_model.model
        model_to_use.eval()
        with torch.no_grad():
            outputs = model_to_use._forward_train(images)  # type: ignore[operator]

        cls_scores_list = cast(list[Tensor], outputs["cls_scores"])
        bbox_preds_list = cast(list[Tensor], outputs["bbox_preds"])
        o2o_cls_scores = cast(list[Tensor], outputs["o2o_cls_scores"])
        o2o_bbox_preds = cast(list[Tensor], outputs["o2o_bbox_preds"])

        gt_boxes_xyxy_norm = _yolo_to_xyxy(gt_bboxes_yolo)
        img_h, img_w = images.shape[-2:]
        sizes = [(img_w, img_h)] * batch_size
        gt_boxes_xyxy_list = _denormalize_xyxy_boxes(gt_boxes_xyxy_norm, sizes)

        dense_loss, loss_vfl, loss_giou, loss_dfl = self._compute_losses(
            fabric=fabric,
            cls_scores=cls_scores_list,
            bbox_preds=bbox_preds_list,
            gt_boxes_xyxy_list=gt_boxes_xyxy_list,
            gt_labels_list=gt_labels_list,
        )
        (
            o2o_loss,
            loss_o2o_obj,
            loss_o2o_cls,
            loss_o2o_box,
            loss_o2o_dfl,
            o2o_stats,
        ) = self._compute_o2o_losses(
            cls_scores=o2o_cls_scores,
            bbox_preds=o2o_bbox_preds,
            gt_boxes_xyxy_list=gt_boxes_xyxy_list,
            gt_labels_list=gt_labels_list,
            image_size=(img_h, img_w),
        )
        total_loss = o2o_loss + 0.5 * dense_loss

        boxes_xyxy, cls_logits = model_to_use._decode_o2o_predictions(
            cls_scores_list=o2o_cls_scores,
            bbox_preds_list=o2o_bbox_preds,
            image_size=(img_h, img_w),
            input_size=(int(img_h), int(img_w)),
        )
        cls_labels = cls_logits.argmax(dim=-1)
        cls_scores = cls_logits.gather(2, cls_labels.unsqueeze(-1)).squeeze(-1)
        scores = torch.sigmoid(cls_scores)
        cls_labels = self.model.internal_class_to_class[cls_labels]

        preds = []
        targets = []
        max_detections = model_to_use.postprocessor.max_detections

        for i in range(batch_size):
            pred_boxes = boxes_xyxy[i].detach()
            pred_scores = scores[i].detach()
            pred_labels = cls_labels[i].detach()
            gt_boxes = gt_boxes_xyxy_list[i].to(device).detach()
            gt_labels_i = gt_labels_list[i].to(device).long().detach()

            if pred_scores.numel() > max_detections:
                topk_scores, topk_idx = torch.topk(
                    pred_scores, k=max_detections, largest=True
                )
                pred_boxes = pred_boxes[topk_idx]
                pred_labels = pred_labels[topk_idx]
                pred_scores = topk_scores

            preds.append(
                {
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels,
                }
            )
            targets.append(
                {
                    "boxes": gt_boxes,
                    "labels": gt_labels_i,
                }
            )

        self.map_metric.update(preds, targets)

        return TaskStepResult(
            loss=total_loss,
            log_dict={
                "val_loss": total_loss.item(),
                "val_loss/loss_o2o": o2o_loss.item(),
                "val_loss/loss_o2o_obj": loss_o2o_obj.item(),
                "val_loss/loss_o2o_cls": loss_o2o_cls.item(),
                "val_loss/loss_o2o_box": loss_o2o_box.item(),
                "val_loss/loss_o2o_dfl": loss_o2o_dfl.item(),
                "val_loss/loss_dense": dense_loss.item(),
                "val_loss/loss_vfl": loss_vfl.item(),
                "val_loss/loss_giou": loss_giou.item(),
                "val_loss/loss_dfl": loss_dfl.item(),
                # Debug o2o stats; remove after investigation.
                "debug/o2o_num_pos": o2o_stats["o2o_num_pos"].item(),
                "debug/o2o_mean_iou": o2o_stats["o2o_mean_iou"].item(),
                "debug/o2o_cls_target_sum": o2o_stats["o2o_cls_target_sum"].item(),
                "debug/o2o_gt_small": o2o_stats["o2o_gt_small"].item(),
                "debug/o2o_gt_medium": o2o_stats["o2o_gt_medium"].item(),
                "debug/o2o_gt_large": o2o_stats["o2o_gt_large"].item(),
                "debug/o2o_gt_center_small": o2o_stats["o2o_gt_center_small"].item(),
                "debug/o2o_gt_center_medium": o2o_stats["o2o_gt_center_medium"].item(),
                "debug/o2o_gt_center_large": o2o_stats["o2o_gt_center_large"].item(),
                "debug/o2o_gt_matched_small": o2o_stats["o2o_gt_matched_small"].item(),
                "debug/o2o_gt_matched_medium": o2o_stats[
                    "o2o_gt_matched_medium"
                ].item(),
                "debug/o2o_gt_matched_large": o2o_stats["o2o_gt_matched_large"].item(),
                "val_metric/map": self.map_metric,
            },
        )

    def _compute_losses(
        self,
        *,
        fabric: Fabric,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        gt_boxes_xyxy_list: list[Tensor],
        gt_labels_list: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size = cls_scores[0].shape[0]
        device = cls_scores[0].device

        decode_bbox_preds_pixel: list[Tensor] = []
        center_and_strides: list[Tensor] = []
        flatten_cls_preds: list[Tensor] = []
        flatten_bbox_preds: list[Tensor] = []

        for level_idx, (cls_score, bbox_pred) in enumerate(zip(cls_scores, bbox_preds)):
            stride = self.strides[level_idx]
            _, _, h, w = cls_score.shape
            num_points = h * w

            y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
            x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
            priors = torch.cat(
                [points, torch.full((num_points, 2), stride, device=device)], dim=-1
            )
            center_and_stride = priors.unsqueeze(0).expand(batch_size, -1, -1)
            center_and_strides.append(center_and_stride)

            center_in_feature = points / stride
            bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(
                batch_size, num_points, 4 * (self.reg_max + 1)
            )
            pred_corners = self.integral(bbox_pred_flat)
            decode_bbox_pred = distance2bbox(
                center_in_feature.unsqueeze(0).expand(batch_size, -1, -1), pred_corners
            )
            decode_bbox_preds_pixel.append(decode_bbox_pred * stride)

            cls_pred_flat = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, num_points, self.num_classes
            )
            flatten_cls_preds.append(cls_pred_flat)
            flatten_bbox_preds.append(bbox_pred_flat)

        all_center_and_strides = torch.cat(center_and_strides, dim=1)
        all_decoded_bboxes_pixel = torch.cat(decode_bbox_preds_pixel, dim=1)
        all_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        all_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        assert all_cls_preds.shape[1] == all_bbox_preds.shape[1]
        assert all_cls_preds.shape[1] == all_center_and_strides.shape[1]

        all_vfl_losses: list[Tensor] = []
        all_giou_losses: list[Tensor] = []
        all_dfl_losses: list[Tensor] = []
        num_pos_per_image: list[int] = []
        total_weight_sum = 0.0

        for img_idx in range(batch_size):
            gt_bboxes = gt_boxes_xyxy_list[img_idx].to(device)
            gt_labels = gt_labels_list[img_idx].to(device).long()

            cls_pred = all_cls_preds[img_idx]
            decoded_bboxes_pixel = all_decoded_bboxes_pixel[img_idx]
            priors = all_center_and_strides[img_idx]
            bbox_pred = all_bbox_preds[img_idx]

            if gt_bboxes.numel() == 0:
                vfl_target = cls_pred.new_zeros(cls_pred.shape)
                vfl_loss = self.vfl_loss(cls_pred, vfl_target)
                all_vfl_losses.append(vfl_loss)
                num_pos_per_image.append(0)
                continue

            assigned_gt_inds, _matched_pred_ious = self.assigner.assign(
                pred_scores=cls_pred.detach().sigmoid(),
                priors=priors,
                decoded_bboxes=decoded_bboxes_pixel.detach(),
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
            )

            pos_mask = assigned_gt_inds > 0
            num_pos = int(pos_mask.sum().item())
            num_pos_per_image.append(num_pos)

            vfl_target = cls_pred.new_zeros(cls_pred.shape)

            if num_pos > 0:
                pos_inds = torch.where(pos_mask)[0]
                pos_assigned_gt_inds = assigned_gt_inds[pos_mask] - 1
                pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]
                pos_gt_labels = gt_labels[pos_assigned_gt_inds]

                pos_priors = priors[pos_mask]
                pos_strides = pos_priors[:, 2:3]
                pos_centers = pos_priors[:, :2]
                pos_centers_feature = pos_centers / pos_strides

                pos_bbox_pred = bbox_pred[pos_mask]

                pos_pred_corners = self.integral(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(
                    pos_centers_feature, pos_pred_corners
                )

                pos_gt_bboxes_feature = pos_gt_bboxes / pos_strides

                pos_ious = box_iou_aligned(
                    pos_decode_bbox_pred.detach(), pos_gt_bboxes_feature.detach()
                ).clamp(min=1e-6)

                vfl_target[pos_inds, pos_gt_labels] = pos_ious.detach().to(
                    vfl_target.dtype
                )

                weight_targets = cls_pred.detach().sigmoid().max(dim=1)[0][pos_inds]
                total_weight_sum += weight_targets.sum().item()

                giou_loss = self.giou_loss(
                    pos_decode_bbox_pred,
                    pos_gt_bboxes_feature.detach(),
                    weight=weight_targets,
                )
                all_giou_losses.append(giou_loss)

                pos_gt_distances = bbox2distance(
                    pos_centers_feature,
                    pos_gt_bboxes_feature,
                    reg_max=float(self.reg_max),
                )
                dfl_weight = weight_targets.unsqueeze(-1).expand(-1, 4).reshape(-1)
                dfl_loss = self.dfl_loss(
                    pos_bbox_pred.reshape(-1, self.reg_max + 1),
                    pos_gt_distances.reshape(-1),
                    weight=dfl_weight,
                )
                dfl_loss = dfl_loss / 4.0
                all_dfl_losses.append(dfl_loss)

            vfl_loss = self.vfl_loss(cls_pred, vfl_target)
            all_vfl_losses.append(vfl_loss)

        num_total_pos = sum(max(n, 1) for n in num_pos_per_image)

        num_pos_tensor = torch.as_tensor(
            [num_total_pos], dtype=torch.float, device=device
        )
        weight_sum_tensor = torch.as_tensor(
            [total_weight_sum], dtype=torch.float, device=device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_pos_tensor)
            dist.all_reduce(weight_sum_tensor)

        num_pos_avg = torch.clamp(num_pos_tensor / fabric.world_size, min=1).item()
        weight_sum_avg = torch.clamp(
            weight_sum_tensor / fabric.world_size, min=1
        ).item()

        zero = torch.tensor(0.0, device=device)
        loss_vfl = sum(all_vfl_losses, zero) / num_pos_avg
        loss_giou = sum(all_giou_losses, zero) / weight_sum_avg
        loss_dfl = sum(all_dfl_losses, zero) / weight_sum_avg

        total_loss = (
            self.model_args.loss_vfl_weight * loss_vfl
            + self.model_args.loss_giou_weight * loss_giou
            + self.model_args.loss_dfl_weight * loss_dfl
        )

        return total_loss, loss_vfl, loss_giou, loss_dfl

    def _compute_o2o_losses(
        self,
        *,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        gt_boxes_xyxy_list: list[Tensor],
        gt_labels_list: list[Tensor],
        image_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        batch_size = cls_scores[0].shape[0]
        device = cls_scores[0].device

        total_obj = torch.zeros((), device=device)
        total_cls = torch.zeros((), device=device)
        total_box = torch.zeros((), device=device)
        total_dfl = torch.zeros((), device=device)
        total_pos = torch.zeros((), device=device)
        total_iou = torch.zeros((), device=device)
        total_cls_target = torch.zeros((), device=device)
        total_gt_small = torch.zeros((), device=device)
        total_gt_medium = torch.zeros((), device=device)
        total_gt_large = torch.zeros((), device=device)
        total_gt_center_small = torch.zeros((), device=device)
        total_gt_center_medium = torch.zeros((), device=device)
        total_gt_center_large = torch.zeros((), device=device)
        total_gt_matched_small = torch.zeros((), device=device)
        total_gt_matched_medium = torch.zeros((), device=device)
        total_gt_matched_large = torch.zeros((), device=device)

        decode_bbox_preds_pixel: list[Tensor] = []
        center_and_strides: list[Tensor] = []
        flatten_cls_preds: list[Tensor] = []
        flatten_bbox_preds: list[Tensor] = []

        for level_idx, (cls_score, bbox_pred) in enumerate(zip(cls_scores, bbox_preds)):
            stride = self.strides[level_idx]
            _, _, h, w = cls_score.shape
            num_points = h * w

            y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
            x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
            priors = torch.cat(
                [points, torch.full((num_points, 2), stride, device=device)], dim=-1
            )
            center_and_stride = priors.unsqueeze(0).expand(batch_size, -1, -1)
            center_and_strides.append(center_and_stride)

            center_in_feature = points / stride
            bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(
                batch_size, num_points, 4 * (self.reg_max + 1)
            )
            pred_corners = self.integral(bbox_pred_flat)
            decode_bbox_pred = distance2bbox(
                center_in_feature.unsqueeze(0).expand(batch_size, -1, -1), pred_corners
            )
            decode_bbox_preds_pixel.append(decode_bbox_pred * stride)

            cls_pred_flat = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, num_points, self.num_classes
            )
            flatten_cls_preds.append(cls_pred_flat)
            flatten_bbox_preds.append(bbox_pred_flat)

        all_center_and_strides = torch.cat(center_and_strides, dim=1)
        all_decoded_bboxes_pixel = torch.cat(decode_bbox_preds_pixel, dim=1)
        all_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        all_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        assert all_cls_preds.shape[1] == all_bbox_preds.shape[1]
        assert all_cls_preds.shape[1] == all_center_and_strides.shape[1]

        img_h, img_w = image_size
        scale_limit = all_decoded_bboxes_pixel.new_tensor([img_w, img_h, img_w, img_h])
        all_decoded_bboxes_pixel = torch.min(
            all_decoded_bboxes_pixel, scale_limit
        ).clamp(min=0)

        for img_idx in range(batch_size):
            pred_boxes = all_decoded_bboxes_pixel[img_idx]
            pred_cls_logits = all_cls_preds[img_idx]
            pred_bbox = all_bbox_preds[img_idx]
            priors = all_center_and_strides[img_idx]
            gt_boxes = gt_boxes_xyxy_list[img_idx].to(device)
            gt_labels = gt_labels_list[img_idx].to(device).long()

            if gt_boxes.numel() > 0:
                gt_wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=0)
                gt_area = gt_wh[:, 0] * gt_wh[:, 1]
                small = gt_area < 32**2
                medium = (gt_area >= 32**2) & (gt_area < 96**2)
                large = gt_area >= 96**2

                total_gt_small = total_gt_small + small.sum()
                total_gt_medium = total_gt_medium + medium.sum()
                total_gt_large = total_gt_large + large.sum()

                centers = priors[:, :2]
                cx = centers[:, 0].unsqueeze(1)
                cy = centers[:, 1].unsqueeze(1)
                in_gt = (cx >= gt_boxes[:, 0]) & (cx <= gt_boxes[:, 2])
                in_gt = in_gt & (cy >= gt_boxes[:, 1]) & (cy <= gt_boxes[:, 3])
                has_center = in_gt.any(dim=0)
                total_gt_center_small = total_gt_center_small + has_center[small].sum()
                total_gt_center_medium = (
                    total_gt_center_medium + has_center[medium].sum()
                )
                total_gt_center_large = total_gt_center_large + has_center[large].sum()

            assigned_gt, assigned_labels, assigned_ious = self.o2o_assigner.assign(
                pred_boxes_xyxy=pred_boxes,
                pred_cls_logits=pred_cls_logits,
                gt_boxes_xyxy=gt_boxes,
                gt_labels=gt_labels,
                prior_centers=priors[:, :2],
            )

            pos_mask = assigned_gt >= 0
            if gt_boxes.numel() > 0 and pos_mask.any():
                matched = torch.zeros(
                    (gt_boxes.shape[0],), device=device, dtype=torch.bool
                )
                matched[assigned_gt[pos_mask]] = True
                gt_wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=0)
                gt_area = gt_wh[:, 0] * gt_wh[:, 1]
                small = gt_area < 32**2
                medium = (gt_area >= 32**2) & (gt_area < 96**2)
                large = gt_area >= 96**2
                total_gt_matched_small = total_gt_matched_small + matched[small].sum()
                total_gt_matched_medium = (
                    total_gt_matched_medium + matched[medium].sum()
                )
                total_gt_matched_large = total_gt_matched_large + matched[large].sum()
            cls_target = pred_cls_logits.new_zeros(pred_cls_logits.shape)
            num_pos = pos_mask.sum()
            total_pos = total_pos + num_pos
            if pos_mask.any():
                cls_target[pos_mask] = F.one_hot(
                    assigned_labels[pos_mask], num_classes=self.num_classes
                ).to(dtype=pred_cls_logits.dtype) * assigned_ious[pos_mask].unsqueeze(
                    -1
                )
                total_iou = total_iou + assigned_ious[pos_mask].sum()
                total_cls_target = total_cls_target + cls_target[pos_mask].sum()

            cls_loss_raw = self.vfl_loss(pred_cls_logits, cls_target)
            cls_norm = cls_target.sum().clamp(min=1.0)
            cls_loss = cls_loss_raw / cls_norm

            if pos_mask.any():
                target_boxes = gt_boxes[assigned_gt[pos_mask]]
                giou_loss = self.giou_loss(pred_boxes[pos_mask], target_boxes)
                pos_priors = priors[pos_mask]
                pos_strides = pos_priors[:, 2:3]
                pos_centers = pos_priors[:, :2]
                pos_centers_feature = pos_centers / pos_strides
                pos_gt_bboxes_feature = target_boxes / pos_strides
                pos_bbox_pred = pred_bbox[pos_mask]
                pos_gt_distances = bbox2distance(
                    pos_centers_feature,
                    pos_gt_bboxes_feature,
                    reg_max=float(self.reg_max),
                )
                dfl_weight = (
                    assigned_ious[pos_mask].unsqueeze(-1).expand(-1, 4).reshape(-1)
                )
                dfl_loss = self.dfl_loss(
                    pos_bbox_pred.reshape(-1, self.reg_max + 1),
                    pos_gt_distances.reshape(-1),
                    weight=dfl_weight,
                )
                dfl_loss = dfl_loss / 4.0
                box_loss = giou_loss + self.model_args.loss_o2o_dfl_weight * dfl_loss
            else:
                box_loss = torch.zeros((), device=device)
                dfl_loss = torch.zeros((), device=device)

            obj_loss = torch.zeros((), device=device)

            total_obj = total_obj + obj_loss
            total_cls = total_cls + cls_loss
            total_box = total_box + box_loss
            total_dfl = total_dfl + dfl_loss

        total_obj = total_obj / batch_size
        total_cls = total_cls / batch_size
        total_box = total_box / batch_size
        total_loss = total_obj + total_cls + total_box
        stats = {
            "o2o_num_pos": total_pos,
            "o2o_mean_iou": total_iou / total_pos.clamp(min=1),
            "o2o_cls_target_sum": total_cls_target,
            "o2o_gt_small": total_gt_small,
            "o2o_gt_medium": total_gt_medium,
            "o2o_gt_large": total_gt_large,
            "o2o_gt_center_small": total_gt_center_small,
            "o2o_gt_center_medium": total_gt_center_medium,
            "o2o_gt_center_large": total_gt_center_large,
            "o2o_gt_matched_small": total_gt_matched_small,
            "o2o_gt_matched_medium": total_gt_matched_medium,
            "o2o_gt_matched_large": total_gt_matched_large,
        }
        total_dfl = total_dfl / batch_size
        return total_loss, total_obj, total_cls, total_box, total_dfl, stats

    def get_optimizer(self, total_steps: int) -> tuple[Optimizer, LRScheduler]:
        """Create optimizer and learning rate scheduler.

        Uses cosine schedule with warmup steps.
        """
        param_groups = [
            {
                "name": "default",
                "params": list(self.model.parameters()),
                "lr": self.model_args.lr,
                "weight_decay": self.model_args.weight_decay,
            }
        ]
        optimizer = SGD(
            param_groups,
            lr=self.model_args.lr,
            momentum=self.model_args.momentum,
            weight_decay=self.model_args.weight_decay,
        )

        warmup_steps = self.model_args.lr_warmup_steps
        max_steps = total_steps
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_steps,
            max_epochs=max_steps,
        )

        return optimizer, scheduler

    def get_export_state_dict(self) -> dict[str, Any]:
        """Return the state dict for exporting.

        Only exports EMA weights if available, following LTDETR pattern.
        This ensures the exported model is ~1x size instead of ~2x.
        """
        state_dict = super().get_export_state_dict()
        if self.ema_model is not None:
            # Only keep EMA weights for export
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k.startswith(self._ema_model_state_dict_key_prefix)
            }
        return state_dict

    def load_train_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load a training state dict.

        Handles loading from checkpoints that may have EMA weights.
        """

        missing_keys, unexpected_keys = self.model.load_train_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

        if self.ema_model is not None:
            missing_keys_ema, unexpected_keys_ema = (
                self.ema_model.model.load_train_state_dict(  # type: ignore[operator]
                    copy.deepcopy(state_dict),
                    strict=strict,
                    assign=assign,
                )
            )
            missing_keys.extend(missing_keys_ema)
            unexpected_keys.extend(unexpected_keys_ema)

        return _IncompatibleKeys(missing_keys, unexpected_keys)
