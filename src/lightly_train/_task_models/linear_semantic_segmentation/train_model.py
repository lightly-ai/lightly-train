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
from lightly_train._task_models.linear_semantic_segmentation.task_model import (
    LinearSemanticSegmentation,
)
from lightly_train._task_models.linear_semantic_segmentation.transforms import (
    LinearSemanticSegmentationTrainTransform,
    LinearSemanticSegmentationTrainTransformArgs,
    LinearSemanticSegmentationValTransform,
    LinearSemanticSegmentationValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train._torch_compile import TorchCompileArgs
from lightly_train._torch_helpers import total_gradient_norm
from lightly_train._visualize import semantic_segmentation
from lightly_train.types import MaskSemanticSegmentationBatch, PathLike


class LinearSemanticSegmentationTrainArgs(TrainModelArgs):
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
        gradient_accumulation_steps: int,
        train_num_batches: int,
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


class LinearSemanticSegmentationTrain(TrainModel):
    task = "semantic_segmentation"
    train_model_args_cls = LinearSemanticSegmentationTrainArgs
    task_metric_args_cls = SemanticSegmentationTaskMetricArgs
    task_model_cls = LinearSemanticSegmentation
    train_transform_cls = LinearSemanticSegmentationTrainTransform
    val_transform_cls = LinearSemanticSegmentationValTransform
    torch_compile_args_cls = TorchCompileArgs

    def __init__(
        self,
        *,
        model_name: str,
        model_args: LinearSemanticSegmentationTrainArgs,
        data_args: MaskSemanticSegmentationDataArgs,
        train_transform_args: LinearSemanticSegmentationTrainTransformArgs,
        val_transform_args: LinearSemanticSegmentationValTransformArgs,
        load_weights: bool,
        metric_args: SemanticSegmentationTaskMetricArgs,
        gradient_accumulation_steps: int,
    ) -> None:
        super().__init__()
        image_size = no_auto(val_transform_args.image_size)
        normalize = no_auto(val_transform_args.normalize)

        self.model_args = model_args
        self.model = LinearSemanticSegmentation(
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
            train_loss_running_mean_window=gradient_accumulation_steps,
        )
        self.val_metrics = SemanticSegmentationTaskMetric(
            task_metric_args=metric_args,
            split="val",
            class_names=class_names,
            ignore_index=data_args.ignore_index,
            loss_names=["loss"],
        )

        # TODO(Nauryz, 04/2026): These visualization thresholds are currently
        # hardcoded, but we may want to make them configurable in the future
        # (with logger_args).
        self.viz_max_images = 4
        self.viz_alpha = 0.6

    def get_task_model(self) -> LinearSemanticSegmentation:
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

        return TaskStepResult(
            loss=loss,
            log_dict={},
            metrics=self.train_metrics,
            visualization=(
                semantic_segmentation.SemanticSegmentationTaskStepVisualization(
                    batch=batch,
                    class_names=self.model.included_classes,
                    image_normalize=self.model.image_normalize,
                    max_images=self.viz_max_images,
                    alpha=self.viz_alpha,
                )
            ),
        )

    def validation_step(
        self,
        fabric: Fabric,
        batch: MaskSemanticSegmentationBatch,
        step: int,
    ) -> TaskStepResult:
        images = batch["image"]
        masks = batch["mask"]

        # Tile, forward, un-tile, and score one image at a time. Tiling produces
        # one crop per aspect-ratio unit, so the total number of crops (and the
        # full-resolution logits kept for them) is unbounded across the batch;
        # holding them all at once runs out of memory for wide or tall validation
        # images. Consuming each image's logits into the loss/metrics before moving
        # on keeps peak memory to a single image, independent of the validation
        # batch size. Only the first ``viz_max_images`` predictions are retained
        # (on CPU) for the visualization.
        loss = torch.tensor(0.0, device=images[0].device)
        viz_logits: list[Tensor] = []
        for image, image_mask in zip(images, masks):
            image_size = (image.shape[-2], image.shape[-1])

            crops_list, origins = self.model.tile([image])

            # Number of classes the head predicts, dropping the ignore channel to
            # match ``untile``'s inputs.
            num_classes = len(self.model.internal_class_to_class)
            if self.model.class_ignore_index is not None:
                num_classes -= 1

            # Full-resolution accumulators for a single image. Forwarding the crops
            # in chunks of at most the dataloader batch size and scattering each
            # chunk's logits into these accumulators right away keeps only the
            # running sum/count (one image) rather than every chunk's logits, so the
            # retained output memory no longer grows with the number of crops.
            logit_sum = torch.zeros((num_classes, *image_size), device=image.device)
            logit_count = torch.zeros_like(logit_sum)

            chunk_size = len(images)
            for start in range(0, len(crops_list), chunk_size):
                chunk_origins = origins[start : start + chunk_size]
                chunk_logits = self.model.forward_train(
                    torch.stack(crops_list[start : start + chunk_size])
                )
                if self.model.class_ignore_index is not None:
                    chunk_logits = chunk_logits[:, :-1]

                for crop_logits, (_, crop_start, crop_end, is_tall) in zip(
                    chunk_logits, chunk_origins
                ):
                    if is_tall:
                        logit_sum[:, crop_start:crop_end, :] += crop_logits
                        logit_count[:, crop_start:crop_end, :] += 1
                    else:
                        logit_sum[:, :, crop_start:crop_end] += crop_logits
                        logit_count[:, :, crop_start:crop_end] += 1

            # Average the logits in the regions of overlap.
            image_logits = logit_sum / logit_count

            image_logits = image_logits.unsqueeze(0)  # Add batch dimension.
            image_mask = image_mask.unsqueeze(0)  # Add batch dimension.
            loss += self.criterion(image_logits, image_mask)
            self.val_metrics.update_with_predictions(
                image_logits.argmax(dim=1), image_mask
            )

            if len(viz_logits) < self.viz_max_images:
                viz_logits.append(image_logits.squeeze(0).cpu())
        loss /= len(images)

        self.val_metrics.update_with_losses({"loss": loss.detach()}, weight=len(images))
        return TaskStepResult(
            loss=loss,
            log_dict={},
            metrics=self.val_metrics,
            visualization=(
                semantic_segmentation.SemanticSegmentationTaskStepVisualization(
                    batch=batch,
                    logits=viz_logits,
                    class_names=self.model.included_classes,
                    image_normalize=self.model.image_normalize,
                    max_images=self.viz_max_images,
                    alpha=self.viz_alpha,
                )
            ),
        )

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

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> Tensor | None:
        gradient_clip_val = self.model_args.gradient_clip_val
        if gradient_clip_val > 0:
            return fabric.clip_gradients(
                module=self,
                optimizer=optimizer,
                max_norm=gradient_clip_val,
                error_if_nonfinite=False,
            )
        # Clipping disabled: return the total norm for logging without mutating grads.
        return total_gradient_norm(self.parameters())
