#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, ClassVar, Literal

import torch
from lightning_fabric import Fabric
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.optim.optimizer import Optimizer

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
from lightly_train.types import ObjectDetectionBatch


class PicoDetObjectDetectionTaskSaveCheckpointArgs(TaskSaveCheckpointArgs):
    """Checkpoint saving configuration for PicoDet."""

    watch_metric: str = "val_metric/map"
    mode: Literal["min", "max"] = "max"


class PicoDetObjectDetectionTrainArgs(TrainModelArgs):
    """Training arguments for PicoDet-S.

    Args:
        learning_rate: Learning rate for SGD optimizer.
        momentum: Momentum for SGD optimizer.
        weight_decay: Weight decay for SGD optimizer.
        vfl_weight: Weight for varifocal loss.
        giou_weight: Weight for GIoU loss.
        dfl_weight: Weight for distribution focal loss.
        simota_center_radius: Center radius for SimOTA assignment.
        simota_candidate_topk: Top-k candidates for dynamic k in SimOTA.
        simota_iou_weight: IoU weight in SimOTA cost matrix.
    """

    default_batch_size: ClassVar[int] = 80
    default_steps: ClassVar[int] = 90_000
    save_checkpoint_args_cls: ClassVar[type[TaskSaveCheckpointArgs]] = (
        PicoDetObjectDetectionTaskSaveCheckpointArgs
    )

    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 4e-5

    vfl_weight: float = 1.0
    giou_weight: float = 2.0
    dfl_weight: float = 0.25

    simota_center_radius: float = 2.5
    simota_candidate_topk: int = 10
    simota_iou_weight: float = 6.0


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

    # Training debug state
    _train_debug_step: int = 0
    _train_debug_logged: bool = False

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
        resolved_image_size: tuple[int, int]
        if val_transform_args.image_size == "auto":
            from lightly_train._task_models.picodet_object_detection.task_model import (
                _MODEL_CONFIGS,
            )

            config = _MODEL_CONFIGS.get(model_name, {})
            config_size_raw = config.get("image_size", (416, 416))
            if isinstance(config_size_raw, tuple) and len(config_size_raw) == 2:
                resolved_image_size = (int(config_size_raw[0]), int(config_size_raw[1]))
            else:
                resolved_image_size = (416, 416)
        else:
            resolved_image_size = val_transform_args.image_size

        image_normalize: dict[str, list[float]] | None = None
        normalize_args = val_transform_args.normalize
        if normalize_args is not None and normalize_args != "auto":
            from lightly_train._transforms.transform import NormalizeArgs

            if isinstance(normalize_args, NormalizeArgs):
                image_normalize = {
                    "mean": list(normalize_args.mean),
                    "std": list(normalize_args.std),
                }

        self.model = PicoDetObjectDetection(
            model_name=model_name,
            image_size=resolved_image_size,
            num_classes=num_classes,
            image_normalize=image_normalize,
            load_weights=load_weights,
        )

        self.num_classes = num_classes
        self.strides = (8, 16, 32, 64)
        self.reg_max = self.model.head.reg_max

        self.vfl_loss = VarifocalLoss(alpha=0.75, gamma=2.0)
        self.dfl_loss = DistributionFocalLoss()
        self.giou_loss = GIoULoss()

        # Integral for decoding bbox predictions
        self.integral = Integral(self.reg_max)

        self.assigner = SimOTAAssigner(
            center_radius=model_args.simota_center_radius,
            candidate_topk=model_args.simota_candidate_topk,
            iou_weight=model_args.simota_iou_weight,
            cls_weight=1.0,
            num_classes=num_classes,
        )

        # EMA model setup (following LTDETR pattern for consistency)
        # EMA is always enabled - it's essential for training quality
        self._ema_model_state_dict_key_prefix = "ema_model."
        self.ema_model = ModelEMA(
            model=self.model,
            decay=0.9998,
            warmups=2000,
        )

        self._map_metric: Any = None

        # EMA for num_pos to smooth out batch size variations
        self._num_pos_ema: float = 100.0
        self._num_pos_ema_decay: float = 0.9

    @property
    def map_metric(self) -> Any:
        """Lazy load mAP metric."""
        if self._map_metric is None:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision

            self._map_metric = MeanAveragePrecision()
            self._map_metric.warn_on_many_detections = False
        return self._map_metric

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
        """Perform a training step."""
        import os

        debug_mode = os.environ.get("PICODET_DEBUG", "0") == "1"

        images = batch["image"]
        gt_bboxes_yolo = batch["bboxes"]
        gt_labels_list = batch["classes"]

        batch_size = images.shape[0]
        device = images.device
        img_h, img_w = images.shape[-2:]

        outputs = self.model(images)
        cls_scores = outputs["cls_scores"]
        bbox_preds = outputs["bbox_preds"]

        # Convert GT from YOLO format to pixel xyxy
        gt_boxes_xyxy_norm = _yolo_to_xyxy(gt_bboxes_yolo)
        sizes = [(img_w, img_h)] * batch_size
        gt_boxes_xyxy_list = _denormalize_xyxy_boxes(gt_boxes_xyxy_norm, sizes)

        # Decode predictions for each level
        decode_bbox_preds_pixel: list[Tensor] = []
        center_and_strides: list[Tensor] = []
        flatten_cls_preds: list[Tensor] = []
        flatten_bbox_preds: list[Tensor] = []

        for level_idx, (cls_score, bbox_pred) in enumerate(zip(cls_scores, bbox_preds)):
            stride = self.strides[level_idx]
            _, _, h, w = cls_score.shape
            num_points = h * w

            # Generate priors: (H*W, 4) as [cx, cy, stride, stride]
            y = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * stride
            x = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * stride
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
            priors = torch.cat(
                [points, torch.full((num_points, 2), stride, device=device)], dim=-1
            )
            center_and_stride = priors.unsqueeze(0).expand(batch_size, -1, -1)
            center_and_strides.append(center_and_stride)

            # Decode bbox predictions to pixel space
            center_in_feature = points / stride
            bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(
                batch_size, num_points, 4 * (self.reg_max + 1)
            )
            pred_corners = self.integral(bbox_pred_flat)
            decode_bbox_pred = distance2bbox(
                center_in_feature.unsqueeze(0).expand(batch_size, -1, -1), pred_corners
            )
            decode_bbox_preds_pixel.append(decode_bbox_pred * stride)

            # Flatten for assignment
            cls_pred_flat = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, num_points, self.num_classes
            )
            flatten_cls_preds.append(cls_pred_flat)
            flatten_bbox_preds.append(bbox_pred_flat)

        # Concatenate across levels
        all_center_and_strides = torch.cat(center_and_strides, dim=1)
        all_decoded_bboxes_pixel = torch.cat(decode_bbox_preds_pixel, dim=1)
        all_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        all_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

        # Process each image
        all_vfl_losses: list[Tensor] = []
        all_giou_losses: list[Tensor] = []
        all_dfl_losses: list[Tensor] = []
        total_num_pos = 0
        total_weight_sum = 0.0

        for img_idx in range(batch_size):
            gt_bboxes = gt_boxes_xyxy_list[img_idx].to(device)
            gt_labels = gt_labels_list[img_idx].to(device).long()

            if gt_bboxes.numel() == 0:
                continue

            cls_pred = all_cls_preds[img_idx]
            decoded_bboxes_pixel = all_decoded_bboxes_pixel[img_idx]
            priors = all_center_and_strides[img_idx]
            bbox_pred = all_bbox_preds[img_idx]

            # SimOTA assignment with sigmoid scores
            assigned_gt_inds, matched_pred_ious = self.assigner.assign(
                pred_scores=cls_pred.detach().sigmoid(),
                priors=priors,
                decoded_bboxes=decoded_bboxes_pixel.detach(),
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
            )

            pos_mask = assigned_gt_inds > 0
            num_pos = int(pos_mask.sum().item())
            total_num_pos += num_pos

            if num_pos == 0:
                continue

            pos_inds = torch.where(pos_mask)[0]
            pos_assigned_gt_inds = assigned_gt_inds[pos_mask] - 1
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds]
            pos_gt_labels = gt_labels[pos_assigned_gt_inds]

            pos_priors = priors[pos_mask]
            pos_strides = pos_priors[:, 2:3]
            pos_centers = pos_priors[:, :2]
            pos_centers_feature = pos_centers / pos_strides

            pos_bbox_pred = bbox_pred[pos_mask]

            # Decode predictions in feature space
            pos_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(pos_centers_feature, pos_pred_corners)

            # GT boxes in feature space
            pos_gt_bboxes_feature = pos_gt_bboxes / pos_strides

            # Compute IoU in feature space for VFL targets
            pos_ious = box_iou_aligned(
                pos_decode_bbox_pred.detach(), pos_gt_bboxes_feature.detach()
            ).clamp(min=1e-6)

            # Debug logging for training (first batch only)
            if debug_mode and not self._train_debug_logged and img_idx == 0:
                self._train_debug_logged = True
                self._log_training_debug_info(
                    step=step,
                    img_h=img_h,
                    img_w=img_w,
                    num_pos=num_pos,
                    pos_ious=pos_ious,
                    pos_strides=pos_strides,
                    pos_decode_bbox_pred=pos_decode_bbox_pred,
                    pos_gt_bboxes_feature=pos_gt_bboxes_feature,
                    gt_bboxes=gt_bboxes,
                    cls_pred=cls_pred,
                )

            # Weight targets
            weight_targets = cls_pred.detach().sigmoid().max(dim=1)[0][pos_inds]
            total_weight_sum += weight_targets.sum().item()

            # VFL targets
            vfl_target = cls_pred.new_zeros(cls_pred.shape)
            vfl_target[pos_inds, pos_gt_labels] = pos_ious.detach()

            # VFL loss
            vfl_loss = self.vfl_loss(cls_pred, vfl_target)
            all_vfl_losses.append(vfl_loss)

            # GIoU loss
            giou_loss = self.giou_loss(
                pos_decode_bbox_pred,
                pos_gt_bboxes_feature.detach(),
                weight=weight_targets,
            )
            all_giou_losses.append(giou_loss)

            # DFL targets
            pos_gt_distances = bbox2distance(
                pos_centers_feature, pos_gt_bboxes_feature, reg_max=float(self.reg_max)
            )

            # DFL loss
            dfl_weight = weight_targets.unsqueeze(-1).expand(-1, 4).reshape(-1)
            dfl_loss = self.dfl_loss(
                pos_bbox_pred.reshape(-1, self.reg_max + 1),
                pos_gt_distances.reshape(-1),
                weight=dfl_weight,
            )
            dfl_loss = dfl_loss / 4.0
            all_dfl_losses.append(dfl_loss)

        # Aggregate losses
        self._num_pos_ema = (
            self._num_pos_ema_decay * self._num_pos_ema
            + (1 - self._num_pos_ema_decay) * max(total_num_pos, 1)
        )
        num_pos_avg = max(self._num_pos_ema, 1.0)
        weight_sum_avg = max(total_weight_sum, 1.0)

        zero = torch.tensor(0.0, device=device)
        loss_vfl = sum(all_vfl_losses, zero) / num_pos_avg
        loss_giou = sum(all_giou_losses, zero) / weight_sum_avg
        loss_dfl = sum(all_dfl_losses, zero) / weight_sum_avg

        total_loss = (
            self.model_args.vfl_weight * loss_vfl
            + self.model_args.giou_weight * loss_giou
            + self.model_args.dfl_weight * loss_dfl
        )

        # Print loss values for debugging
        if debug_mode and step % 100 == 0:
            print(f"\n[Step {step}] Loss values:")
            print(f"  VFL loss: {loss_vfl.item():.6f} (weight: {self.model_args.vfl_weight})")
            print(f"  GIoU loss: {loss_giou.item():.6f} (weight: {self.model_args.giou_weight})")
            print(f"  DFL loss: {loss_dfl.item():.6f} (weight: {self.model_args.dfl_weight})")
            print(f"  Total loss: {total_loss.item():.6f}")
            print(f"  num_pos: {total_num_pos}, num_pos_avg (EMA): {num_pos_avg:.2f}")
            print(f"  weight_sum: {total_weight_sum:.4f}, weight_sum_avg: {weight_sum_avg:.4f}")

            # Per-class analysis for first image in batch
            if batch_size > 0:
                self._log_per_class_debug(
                    step=step,
                    all_cls_preds=all_cls_preds,
                    gt_boxes_xyxy_list=gt_boxes_xyxy_list,
                    gt_labels_list=gt_labels_list,
                    device=device,
                )

            # Gradient analysis (requires backward to have been called on a previous step)
            self._log_gradient_info()

        # Average losses across devices for logging (distributed training support)
        loss_dict = {
            "train_loss": total_loss,
            "loss_vfl": loss_vfl,
            "loss_giou": loss_giou,
            "loss_dfl": loss_dfl,
        }
        loss_dict = reduce_dict(loss_dict)

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
        import os

        debug_mode = os.environ.get("PICODET_DEBUG", "0") == "1"

        images = batch["image"]
        gt_bboxes_yolo = batch["bboxes"]
        gt_labels_list = batch["classes"]

        batch_size = images.shape[0]
        device = images.device

        # Use EMA model for validation
        self.ema_model.model.eval()
        with torch.no_grad():
            outputs = self.ema_model.model(images)

        cls_scores = outputs["cls_scores"]
        bbox_preds = outputs["bbox_preds"]

        gt_boxes_xyxy_norm = _yolo_to_xyxy(gt_bboxes_yolo)
        img_h, img_w = images.shape[-2:]
        sizes = [(img_w, img_h)] * batch_size
        gt_boxes_xyxy_list = _denormalize_xyxy_boxes(gt_boxes_xyxy_norm, sizes)

        from lightly_train._task_models.picodet_object_detection.postprocessor import (
            PicoDetPostProcessor,
        )

        postprocessor = self.model.postprocessor
        assert isinstance(postprocessor, PicoDetPostProcessor)
        predictions = postprocessor.forward_batch(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            original_sizes=torch.tensor([[img_h, img_w]] * batch_size, device=device),
            score_threshold=0.001,
        )

        preds = []
        targets = []

        for i in range(batch_size):
            pred_boxes = predictions[i]["bboxes"].detach()
            pred_scores = predictions[i]["scores"].detach()
            pred_labels = predictions[i]["labels"].detach()
            gt_boxes = gt_boxes_xyxy_list[i].to(device).detach()
            gt_labels_i = gt_labels_list[i].to(device).long().detach()

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

            # Debug logging for first image in first batch
            if debug_mode and i == 0 and not hasattr(self, "_debug_logged"):
                self._debug_logged = True
                self._log_debug_info(
                    img_h=img_h,
                    img_w=img_w,
                    pred_boxes=pred_boxes,
                    pred_scores=pred_scores,
                    pred_labels=pred_labels,
                    gt_boxes=gt_boxes,
                    gt_labels=gt_labels_i,
                    cls_scores=cls_scores,
                    bbox_preds=bbox_preds,
                )

        self.map_metric.to(device)
        self.map_metric.update(preds, targets)

        return TaskStepResult(
            loss=torch.tensor(0.0, device=device),
            log_dict={"val_metric/map": self.map_metric},
        )

    def _log_debug_info(
        self,
        img_h: int,
        img_w: int,
        pred_boxes: Tensor,
        pred_scores: Tensor,
        pred_labels: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
    ) -> None:
        """Log debug information for diagnosing low mAP."""
        print("\n" + "=" * 80)
        print("PICODET DEBUG INFO (first validation image)")
        print("=" * 80)

        print(f"\n[Image Info]")
        print(f"  Image size: {img_h} x {img_w}")

        print(f"\n[Ground Truth]")
        print(f"  Number of GT boxes: {len(gt_boxes)}")
        if len(gt_boxes) > 0:
            print(f"  GT boxes (first 5): {gt_boxes[:5].tolist()}")
            print(f"  GT labels (first 5): {gt_labels[:5].tolist()}")
            print(f"  GT label range: [{gt_labels.min().item()}, {gt_labels.max().item()}]")
            # Box statistics
            widths = gt_boxes[:, 2] - gt_boxes[:, 0]
            heights = gt_boxes[:, 3] - gt_boxes[:, 1]
            print(f"  GT box width range: [{widths.min().item():.1f}, {widths.max().item():.1f}]")
            print(f"  GT box height range: [{heights.min().item():.1f}, {heights.max().item():.1f}]")

        print(f"\n[Predictions]")
        print(f"  Number of predictions: {len(pred_boxes)}")
        if len(pred_boxes) > 0:
            print(f"  Pred boxes (first 5): {pred_boxes[:5].tolist()}")
            print(f"  Pred scores (first 5): {pred_scores[:5].tolist()}")
            print(f"  Pred labels (first 5): {pred_labels[:5].tolist()}")
            print(f"  Pred score range: [{pred_scores.min().item():.4f}, {pred_scores.max().item():.4f}]")
            print(f"  Pred label range: [{pred_labels.min().item()}, {pred_labels.max().item()}]")
            # Box statistics
            widths = pred_boxes[:, 2] - pred_boxes[:, 0]
            heights = pred_boxes[:, 3] - pred_boxes[:, 1]
            print(f"  Pred box width range: [{widths.min().item():.1f}, {widths.max().item():.1f}]")
            print(f"  Pred box height range: [{heights.min().item():.1f}, {heights.max().item():.1f}]")
            # Check for invalid boxes
            invalid_w = (widths <= 0).sum().item()
            invalid_h = (heights <= 0).sum().item()
            if invalid_w > 0 or invalid_h > 0:
                print(f"  WARNING: {invalid_w} boxes with width <= 0, {invalid_h} with height <= 0")

        print(f"\n[Raw Model Outputs]")
        for level_idx, (cls_score, bbox_pred) in enumerate(zip(cls_scores, bbox_preds)):
            stride = self.strides[level_idx]
            print(f"  Level {level_idx} (stride {stride}):")
            print(f"    cls_score shape: {cls_score.shape}")
            print(f"    bbox_pred shape: {bbox_pred.shape}")
            # Score statistics (after sigmoid)
            scores_sigmoid = cls_score[0].sigmoid()
            print(f"    cls score range (sigmoid): [{scores_sigmoid.min().item():.4f}, {scores_sigmoid.max().item():.4f}]")
            print(f"    cls score mean (sigmoid): {scores_sigmoid.mean().item():.4f}")
            # DFL statistics
            print(f"    bbox_pred range: [{bbox_pred[0].min().item():.2f}, {bbox_pred[0].max().item():.2f}]")

        # Check IoU between predictions and GT
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            from lightly_train._task_models.picodet_object_detection.losses import box_iou
            ious = box_iou(pred_boxes, gt_boxes)
            max_ious_per_pred = ious.max(dim=1)[0]
            max_ious_per_gt = ious.max(dim=0)[0]
            print(f"\n[IoU Analysis]")
            print(f"  Max IoU per prediction: mean={max_ious_per_pred.mean().item():.3f}, max={max_ious_per_pred.max().item():.3f}")
            print(f"  Max IoU per GT: mean={max_ious_per_gt.mean().item():.3f}, max={max_ious_per_gt.max().item():.3f}")
            print(f"  Predictions with IoU > 0.5: {(max_ious_per_pred > 0.5).sum().item()}/{len(pred_boxes)}")
            print(f"  GT boxes with IoU > 0.5: {(max_ious_per_gt > 0.5).sum().item()}/{len(gt_boxes)}")

        print("=" * 80 + "\n")

    def _log_training_debug_info(
        self,
        step: int,
        img_h: int,
        img_w: int,
        num_pos: int,
        pos_ious: Tensor,
        pos_strides: Tensor,
        pos_decode_bbox_pred: Tensor,
        pos_gt_bboxes_feature: Tensor,
        gt_bboxes: Tensor,
        cls_pred: Tensor,
    ) -> None:
        """Log debug information during training."""
        print("\n" + "=" * 80)
        print(f"PICODET TRAINING DEBUG INFO (step {step}, first image)")
        print("=" * 80)

        print(f"\n[Image Info]")
        print(f"  Image size: {img_h} x {img_w}")
        print(f"  reg_max: {self.reg_max}")

        print(f"\n[Positive Samples]")
        print(f"  Number of positive samples: {num_pos}")
        if num_pos > 0:
            print(f"  VFL target (IoU) range: [{pos_ious.min().item():.4f}, {pos_ious.max().item():.4f}]")
            print(f"  VFL target (IoU) mean: {pos_ious.mean().item():.4f}")
            print(f"  VFL target (IoU) distribution:")
            print(f"    IoU < 0.1: {(pos_ious < 0.1).sum().item()}/{num_pos}")
            print(f"    IoU 0.1-0.3: {((pos_ious >= 0.1) & (pos_ious < 0.3)).sum().item()}/{num_pos}")
            print(f"    IoU 0.3-0.5: {((pos_ious >= 0.3) & (pos_ious < 0.5)).sum().item()}/{num_pos}")
            print(f"    IoU 0.5-0.7: {((pos_ious >= 0.5) & (pos_ious < 0.7)).sum().item()}/{num_pos}")
            print(f"    IoU >= 0.7: {(pos_ious >= 0.7).sum().item()}/{num_pos}")

        print(f"\n[Stride Distribution of Positive Samples]")
        if num_pos > 0:
            unique_strides = pos_strides.squeeze(-1).unique()
            for stride in unique_strides:
                count = (pos_strides.squeeze(-1) == stride).sum().item()
                stride_mask = pos_strides.squeeze(-1) == stride
                stride_ious = pos_ious[stride_mask]
                print(f"  Stride {int(stride.item())}: {count} samples, IoU mean={stride_ious.mean().item():.4f}, max={stride_ious.max().item():.4f}")

        print(f"\n[Ground Truth Boxes (pixel space)]")
        print(f"  Number of GT boxes: {len(gt_bboxes)}")
        if len(gt_bboxes) > 0:
            widths = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            heights = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            print(f"  GT boxes (first 3): {gt_bboxes[:3].tolist()}")
            print(f"  GT box width range: [{widths.min().item():.1f}, {widths.max().item():.1f}]")
            print(f"  GT box height range: [{heights.min().item():.1f}, {heights.max().item():.1f}]")

        print(f"\n[Decoded Predictions vs GT (feature space)]")
        if num_pos > 0:
            # Sample a few positive samples to show
            n_show = min(5, num_pos)
            print(f"  Showing first {n_show} positive samples:")
            for i in range(n_show):
                pred_box = pos_decode_bbox_pred[i].tolist()
                gt_box = pos_gt_bboxes_feature[i].tolist()
                stride = pos_strides[i, 0].item()
                iou = pos_ious[i].item()
                # Convert to pixel space for comparison
                pred_box_pixel = [v * stride for v in pred_box]
                gt_box_pixel = [v * stride for v in gt_box]
                print(f"    Sample {i}: stride={int(stride)}, IoU={iou:.4f}")
                print(f"      Pred (feature): [{pred_box[0]:.2f}, {pred_box[1]:.2f}, {pred_box[2]:.2f}, {pred_box[3]:.2f}]")
                print(f"      GT (feature):   [{gt_box[0]:.2f}, {gt_box[1]:.2f}, {gt_box[2]:.2f}, {gt_box[3]:.2f}]")
                print(f"      Pred (pixel):   [{pred_box_pixel[0]:.1f}, {pred_box_pixel[1]:.1f}, {pred_box_pixel[2]:.1f}, {pred_box_pixel[3]:.1f}]")
                print(f"      GT (pixel):     [{gt_box_pixel[0]:.1f}, {gt_box_pixel[1]:.1f}, {gt_box_pixel[2]:.1f}, {gt_box_pixel[3]:.1f}]")

        print(f"\n[Classification Predictions (raw)]")
        cls_sigmoid = cls_pred.sigmoid()
        print(f"  cls_pred shape: {cls_pred.shape}")
        print(f"  cls_pred (logit) range: [{cls_pred.min().item():.4f}, {cls_pred.max().item():.4f}]")
        print(f"  cls_pred (sigmoid) range: [{cls_sigmoid.min().item():.4f}, {cls_sigmoid.max().item():.4f}]")
        print(f"  cls_pred (sigmoid) mean: {cls_sigmoid.mean().item():.4f}")

        # Check if reg_max is limiting predictions
        print(f"\n[reg_max Analysis]")
        max_box_sizes_pixel = [self.reg_max * 2 * s for s in self.strides]
        print(f"  Max box size per stride (in pixels):")
        for s, max_size in zip(self.strides, max_box_sizes_pixel):
            print(f"    Stride {s}: {max_size} pixels")
        if len(gt_bboxes) > 0:
            max_gt_dim = max(widths.max().item(), heights.max().item())
            print(f"  Largest GT dimension: {max_gt_dim:.1f} pixels")
            for s, max_size in zip(self.strides, max_box_sizes_pixel):
                if max_gt_dim > max_size:
                    print(f"    WARNING: Stride {s} cannot fully represent GT (max {max_size} < GT {max_gt_dim:.1f})")

        print("=" * 80 + "\n")

    def _log_gradient_info(self) -> None:
        """Log gradient norms for classification vs bbox head."""
        print(f"\n  [Gradient Analysis (from previous step)]")

        # Get the head module
        head = self.model.head

        # Check gradients for classification output (first num_classes channels of gfl_cls)
        cls_grad_norms: list[float] = []
        bbox_grad_norms: list[float] = []

        for level_idx, gfl_cls in enumerate(head.gfl_cls):
            if gfl_cls is None or not isinstance(gfl_cls, torch.nn.Conv2d):
                continue

            # Weight gradients
            if gfl_cls.weight.grad is not None:
                # For shared head: first num_classes outputs are cls, rest are bbox
                cls_weight_grad = gfl_cls.weight.grad[: head.num_classes]
                bbox_weight_grad = gfl_cls.weight.grad[head.num_classes :]

                cls_grad_norm = cls_weight_grad.norm().item()
                bbox_grad_norm = bbox_weight_grad.norm().item()

                cls_grad_norms.append(cls_grad_norm)
                bbox_grad_norms.append(bbox_grad_norm)

                print(f"    Level {level_idx} gfl_cls weight grad:")
                print(f"      cls grad norm: {cls_grad_norm:.6f}")
                print(f"      bbox grad norm: {bbox_grad_norm:.6f}")
                print(f"      ratio (cls/bbox): {cls_grad_norm / max(bbox_grad_norm, 1e-8):.4f}")

            # Bias gradients
            if gfl_cls.bias is not None and gfl_cls.bias.grad is not None:
                cls_bias_grad = gfl_cls.bias.grad[: head.num_classes]
                bbox_bias_grad = gfl_cls.bias.grad[head.num_classes :]

                print(f"    Level {level_idx} gfl_cls bias grad:")
                print(f"      cls bias grad norm: {cls_bias_grad.norm().item():.6f}")
                print(f"      bbox bias grad norm: {bbox_bias_grad.norm().item():.6f}")

        if cls_grad_norms and bbox_grad_norms:
            avg_cls_grad = sum(cls_grad_norms) / len(cls_grad_norms)
            avg_bbox_grad = sum(bbox_grad_norms) / len(bbox_grad_norms)
            print(f"    Average across levels:")
            print(f"      cls grad norm: {avg_cls_grad:.6f}")
            print(f"      bbox grad norm: {avg_bbox_grad:.6f}")
            print(f"      ratio (cls/bbox): {avg_cls_grad / max(avg_bbox_grad, 1e-8):.4f}")
        else:
            print(f"    No gradients available (first step or gradients cleared)")

    def _log_per_class_debug(
        self,
        step: int,
        all_cls_preds: Tensor,
        gt_boxes_xyxy_list: list[Tensor],
        gt_labels_list: list[Tensor],
        device: torch.device,
    ) -> None:
        """Log per-class statistics for debugging."""
        from collections import Counter

        print(f"\n  [Per-Class Analysis (batch)]")

        # Collect GT class distribution across batch
        all_gt_labels: list[int] = []
        for gt_labels in gt_labels_list:
            if gt_labels.numel() > 0:
                all_gt_labels.extend(gt_labels.cpu().tolist())

        gt_class_counts = Counter(all_gt_labels)
        print(f"    GT class distribution (top 10):")
        for cls_id, count in gt_class_counts.most_common(10):
            print(f"      Class {cls_id}: {count} instances")

        # Analyze model predictions (across all priors)
        # all_cls_preds shape: (batch_size, num_priors, num_classes)
        cls_sigmoid = all_cls_preds.detach().sigmoid()

        # Get predicted class for each prior (argmax)
        pred_classes = cls_sigmoid.argmax(dim=-1)  # (batch_size, num_priors)
        pred_class_counts = Counter(pred_classes.flatten().cpu().tolist())

        print(f"    Predicted class distribution (top 10, across all priors):")
        for cls_id, count in pred_class_counts.most_common(10):
            pct = 100.0 * count / pred_classes.numel()
            print(f"      Class {cls_id}: {count} priors ({pct:.1f}%)")

        # Check max score per class (across all priors in batch)
        max_score_per_class = cls_sigmoid.max(dim=0)[0].max(dim=0)[0]  # (num_classes,)
        print(f"    Max score per class (top 10):")
        top_scores, top_classes = max_score_per_class.topk(min(10, self.num_classes))
        for score, cls_id in zip(top_scores.cpu().tolist(), top_classes.cpu().tolist()):
            print(f"      Class {cls_id}: {score:.4f}")

        # Check if GT classes have high scores
        print(f"    GT class scores (max score for each GT class):")
        for cls_id in sorted(gt_class_counts.keys())[:10]:
            if cls_id < self.num_classes:
                max_score = max_score_per_class[cls_id].item()
                print(f"      Class {cls_id}: max_score={max_score:.4f}, GT_count={gt_class_counts[cls_id]}")

        # Check prediction vs GT alignment for high-confidence predictions
        high_conf_mask = cls_sigmoid.max(dim=-1)[0] > 0.1  # Priors with >0.1 confidence
        if high_conf_mask.any():
            high_conf_preds = pred_classes[high_conf_mask]
            high_conf_counts = Counter(high_conf_preds.cpu().tolist())
            print(f"    High-confidence (>0.1) predictions:")
            for cls_id, count in high_conf_counts.most_common(5):
                print(f"      Class {cls_id}: {count} priors")
        else:
            print(f"    No predictions with confidence > 0.1")

    def get_optimizer(self, total_steps: int) -> tuple[Optimizer, LRScheduler]:
        """Create optimizer and learning rate scheduler."""
        param_groups = [
            {
                "name": "default",
                "params": list(self.model.parameters()),
                "lr": self.model_args.learning_rate,
                "weight_decay": self.model_args.weight_decay,
            }
        ]
        optimizer = SGD(
            param_groups,
            lr=self.model_args.learning_rate,
            momentum=self.model_args.momentum,
            weight_decay=self.model_args.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
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
        import copy

        from torch.nn.modules.module import _IncompatibleKeys

        # Load into the main model
        missing_keys, unexpected_keys = self.model.load_train_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

        # Also load into EMA model if present
        if self.ema_model is not None:
            missing_keys_ema, unexpected_keys_ema = (
                self.ema_model.model.load_train_state_dict(  # type: ignore[operator]
                    # Copy to avoid assigning the same weights to both models
                    copy.deepcopy(state_dict),
                    strict=strict,
                    assign=assign,
                )
            )
            missing_keys.extend(missing_keys_ema)
            unexpected_keys.extend(unexpected_keys_ema)

        return _IncompatibleKeys(missing_keys, unexpected_keys)
