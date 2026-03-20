#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import torch
from lightly.utils.scheduler import CosineWarmupScheduler
from lightning_fabric import Fabric
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from lightly_train._configs.validate import no_auto
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
)
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._metrics.semantic_segmentation.task_metric import (
    SemanticSegmentationTaskMetric,
    SemanticSegmentationTaskMetricArgs,
)
from lightly_train._optim import optimizer_helpers
from lightly_train._task_models.dinov2_linear_semantic_segmentation.task_model import (
    DINOv2LinearSemanticSegmentation,
)
from lightly_train._task_models.dinov2_linear_semantic_segmentation.transforms import (
    DINOv2LinearSemanticSegmentationTrainTransform,
    DINOv2LinearSemanticSegmentationTrainTransformArgs,
    DINOv2LinearSemanticSegmentationValTransform,
    DINOv2LinearSemanticSegmentationValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train._torch_compile import TorchCompileArgs
from lightly_train.types import MaskSemanticSegmentationBatch, PathLike


class DINOv2LinearSemanticSegmentationTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    # Default comes from PVOC12
    default_steps: ClassVar[int] = 80_000

    # Model args
    backbone_freeze: bool = True
    backbone_weights: PathLike | None = None
    drop_path_rate: float | Literal["auto"] = "auto"

    # Gradient clipping. Same value as DINOv2.
    gradient_clip_val: float = 3.0

    # Optim
    lr: float = 0.001
    weight_decay: float = 0.01

    def resolve_auto(
        self,
        total_steps: int,
        model_name: str,
        model_init_args: dict[str, Any],
        data_args: TaskDataArgs,
    ) -> None:
        if self.drop_path_rate == "auto":
            backbone_args = model_init_args.get("backbone_args", {})
            assert isinstance(backbone_args, dict)  # for mypy

            drop_path_rate = backbone_args.get("drop_path_rate", 0.0)
            assert isinstance(drop_path_rate, float)  # for mypy
            self.drop_path_rate = drop_path_rate


class DINOv2LinearSemanticSegmentationTrain(TrainModel):
    task = "semantic_segmentation"
    train_model_args_cls = DINOv2LinearSemanticSegmentationTrainArgs
    task_metric_args_cls = SemanticSegmentationTaskMetricArgs
    task_model_cls = DINOv2LinearSemanticSegmentation
    train_transform_cls = DINOv2LinearSemanticSegmentationTrainTransform
    val_transform_cls = DINOv2LinearSemanticSegmentationValTransform
    torch_compile_args_cls = TorchCompileArgs

    def __init__(
        self,
        *,
        model_name: str,
        model_args: DINOv2LinearSemanticSegmentationTrainArgs,
        data_args: MaskSemanticSegmentationDataArgs,
        train_transform_args: DINOv2LinearSemanticSegmentationTrainTransformArgs,
        val_transform_args: DINOv2LinearSemanticSegmentationValTransformArgs,
        load_weights: bool,
        metric_args: SemanticSegmentationTaskMetricArgs,
    ) -> None:
        super().__init__()
        image_size = no_auto(val_transform_args.image_size)
        normalize = no_auto(val_transform_args.normalize)

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
            image_size=image_size,
            image_normalize=normalize.model_dump(),
            load_weights=load_weights,
        )
        self.criterion = CrossEntropyLoss(ignore_index=data_args.ignore_index)

        # Metrics
        class_names = list(data_args.included_classes.values())
        self.metric_args = metric_args
        self.train_metrics = SemanticSegmentationTaskMetric(
            task_metric_args=metric_args,
            split="train",
            class_names=class_names,
            ignore_index=data_args.ignore_index,
            loss_names=["loss"],
        )
        self.val_metrics = SemanticSegmentationTaskMetric(
            task_metric_args=metric_args,
            split="val",
            class_names=class_names,
            ignore_index=data_args.ignore_index,
            loss_names=["loss"],
        )

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

        self.train_metrics.update_with_losses(
            {"loss": loss.detach()}, weight=images.shape[0]
        )
        if self.metric_args.train:
            self.train_metrics.update_with_predictions(logits.argmax(dim=1), masks)

        return TaskStepResult(loss=loss, log_dict={}, metrics=self.train_metrics)

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
            self.val_metrics.update_with_predictions(
                image_logits.argmax(dim=1), image_mask
            )
        loss /= len(images)

        self.val_metrics.update_with_losses({"loss": loss.detach()}, weight=len(images))

        return TaskStepResult(loss=loss, log_dict={}, metrics=self.val_metrics)

    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
    ) -> tuple[Optimizer, LRScheduler]:
        params_wd, params_no_wd = optimizer_helpers.get_weight_decay_parameters([self])
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
        lr = self.model_args.lr * math.sqrt(
            global_batch_size / self.model_args.default_batch_size
        )
        optimizer = AdamW(
            params=params,
            lr=lr,
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
        if self.model_args.gradient_clip_val > 0:
            fabric.clip_gradients(
                module=self,
                optimizer=optimizer,
                max_norm=self.model_args.gradient_clip_val,
                error_if_nonfinite=False,
            )
