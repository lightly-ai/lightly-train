#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, ClassVar

import torch
from lightly.models.utils import get_weight_decay_parameters
from lightly.utils.scheduler import CosineWarmupScheduler
from lightning_fabric import Fabric
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
)
from lightly_train._task_models.dinov2_linear_semantic_segmentation.task_model import (
    DINOv2LinearSemanticSegmentation,
)
from lightly_train._task_models.dinov2_linear_semantic_segmentation.transforms import (
    DINOv2LinearSemanticSegmentationTrainTransform,
    DINOv2LinearSemanticSegmentationValTransform,
    DINOv2LinearSemanticSegmentationValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train.types import MaskSemanticSegmentationBatch, PathLike


class DINOv2LinearSemanticSegmentationTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    # Default comes from PVOC12
    default_steps: ClassVar[int] = 80_000

    backbone_freeze: bool = True
    backbone_weights: PathLike | None = None
    drop_path_rate: float = 0.0

    # Gradient clipping. Same value as DINOv2.
    gradient_clip_val: float = 3.0

    # Optim
    lr: float = 0.001
    weight_decay: float = 0.01

    # Metrics
    metric_log_classwise: bool = True
    metric_log_debug: bool = False

    def resolve_auto(self, total_steps: int, model_name: str) -> None:
        pass


class DINOv2LinearSemanticSegmentationTrain(TrainModel):
    task = "semantic_segmentation"
    train_model_args_cls = DINOv2LinearSemanticSegmentationTrainArgs
    task_model_cls = DINOv2LinearSemanticSegmentation
    train_transform_cls = DINOv2LinearSemanticSegmentationTrainTransform
    val_transform_cls = DINOv2LinearSemanticSegmentationValTransform

    def __init__(
        self,
        *,
        model_name: str,
        model_args: DINOv2LinearSemanticSegmentationTrainArgs,
        data_args: MaskSemanticSegmentationDataArgs,
        val_transform_args: DINOv2LinearSemanticSegmentationValTransformArgs,
    ) -> None:
        super().__init__()
        # Lazy import because torchmetrics is an optional dependency.
        from torchmetrics import ClasswiseWrapper, JaccardIndex, MeanMetric
        from torchmetrics.classification import (  # type: ignore[attr-defined]
            MulticlassJaccardIndex,
        )

        self.model_args = model_args
        self.model = DINOv2LinearSemanticSegmentation(
            model_name=model_name,
            classes=data_args.included_classes,
            class_ignore_index=(
                data_args.ignore_index if data_args.ignore_classes else None
            ),
            backbone_freeze=self.model_args.backbone_freeze,
            backbone_weights=model_args.backbone_weights,
            backbone_args={
                "drop_path_rate": model_args.drop_path_rate,
            },
            image_size=val_transform_args.image_size,
            image_normalize=val_transform_args.normalize.model_dump(),
        )
        self.criterion = CrossEntropyLoss(ignore_index=data_args.ignore_index)

        # Metrics
        self.val_loss = MeanMetric()

        # TODO(Guarin, 08/25): Speed up metric calculation by not calculating
        # mIoU and classwise IoU separately. mIoU can be derived from the classwise IoU.
        self.train_miou = JaccardIndex(
            task="multiclass",  # type: ignore[arg-type]
            num_classes=data_args.num_included_classes,
            ignore_index=data_args.ignore_index,
        )
        self.val_miou = self.train_miou.clone()

        # Classwise MeanIoU
        class_labels = list(data_args.included_classes.values())
        self.train_classwise_iou = ClasswiseWrapper(  # type: ignore[call-arg]
            MulticlassJaccardIndex(
                num_classes=data_args.num_included_classes,
                validate_args=False,
                ignore_index=data_args.ignore_index,
                average=None,
            ),
            prefix="_",
            labels=class_labels,
        )
        self.val_classwise_iou = self.train_classwise_iou.clone()

    def get_task_model(self) -> DINOv2LinearSemanticSegmentation:
        return self.model

    def training_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch, step: int
    ) -> TaskStepResult:
        images = batch["image"]
        assert isinstance(images, Tensor), "Images must be a single tensor for training"
        masks = batch["mask"]
        assert isinstance(masks, Tensor), "Masks must be a single tensor for training"

        logits = self.model.forward_train(images)
        if self.model.class_ignore_index is not None:
            logits = logits[:, :-1]  # Drop logits for the ignored class.
        loss = self.criterion(logits, masks)

        self.train_miou.update(logits, masks)
        log_dict = {
            "train_loss": loss.detach(),
            "train_metric/miou": self.train_miou,
        }
        if self.model_args.metric_log_debug or self.model_args.metric_log_classwise:
            self.train_classwise_iou.update(logits, masks)
            log_dict["train_metric_classwise/miou"] = self.train_classwise_iou

        return TaskStepResult(loss=loss, log_dict=log_dict)

    def validation_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> TaskStepResult:
        images = batch["image"]
        masks = batch["mask"]
        image_sizes = [(image.shape[-2], image.shape[-1]) for image in images]

        # Tile the images.
        crops_list, origins = self.model.tile(images)
        crops = torch.stack(crops_list)

        crop_logits = self.model.forward_train(crops)
        if self.model.class_ignore_index is not None:
            crop_logits = crop_logits[:, :-1]

        # Un-tile the predictions.
        logits = self.model.untile(
            crop_logits=crop_logits, origins=origins, image_sizes=image_sizes
        )

        loss = torch.tensor(0.0, device=crop_logits.device)
        for image_logits, image_mask in zip(logits, masks):
            image_logits = image_logits.unsqueeze(0)  # Add batch dimension.
            image_mask = image_mask.unsqueeze(0)  # Add batch dimension.
            loss += self.criterion(image_logits, image_mask)
            self.val_miou.update(image_logits, image_mask)
            if self.model_args.metric_log_debug or self.model_args.metric_log_classwise:
                self.val_classwise_iou.update(image_logits, image_mask)
        loss /= len(images)

        # Metrics
        self.val_loss.update(loss, weight=len(images))
        log_dict = {
            "val_loss": loss.detach(),
            "val_metric/miou": self.val_miou,
        }
        if self.model_args.metric_log_debug or self.model_args.metric_log_classwise:
            log_dict["val_metric_classwise/miou"] = self.val_classwise_iou

        return TaskStepResult(loss=loss, log_dict=log_dict)

    def get_optimizer(self, total_steps: int) -> tuple[Optimizer, LRScheduler]:
        params_wd, params_no_wd = get_weight_decay_parameters([self])
        params_wd = [p for p in params_wd if p.requires_grad]
        params_no_wd = [p for p in params_no_wd if p.requires_grad]
        params: list[dict[str, Any]] = [
            {"name": "params", "params": params_wd},
            {
                "name": "no_weight_decay",
                "params": params_no_wd,
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            params=params,
            lr=self.model_args.lr,
            weight_decay=self.model_args.weight_decay,
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=0,
            max_epochs=total_steps,
        )
        return optimizer, scheduler

    def set_train_mode(self) -> None:
        self.train()
        if self.model_args.backbone_freeze:
            self.model.freeze_backbone()

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        fabric.clip_gradients(
            module=self,
            optimizer=optimizer,
            max_norm=self.model_args.gradient_clip_val,
        )
