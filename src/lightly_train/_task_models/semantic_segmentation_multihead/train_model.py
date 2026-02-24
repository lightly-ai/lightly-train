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
from pydantic import Field, model_validator
from torch import Tensor
from torch.nn import CrossEntropyLoss, ModuleDict
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD

from lightly_train._configs.validate import no_auto
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
)
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._optim import optimizer_helpers
from lightly_train._task_checkpoint import TaskSaveCheckpointArgs
from lightly_train._task_models.semantic_segmentation_multihead.task_model import (
    SemanticSegmentationMultihead,
)
from lightly_train._task_models.semantic_segmentation_multihead.transforms import (
    SemanticSegmentationMultiheadTrainTransform,
    SemanticSegmentationMultiheadTrainTransformArgs,
    SemanticSegmentationMultiheadValTransform,
    SemanticSegmentationMultiheadValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train.types import MaskSemanticSegmentationBatch, PathLike


class SemanticSegmentationMultiheadSaveCheckpointArgs(TaskSaveCheckpointArgs):
    watch_metric: str = "val_metric/miou"
    mode: Literal["min", "max"] = "max"


class SemanticSegmentationMultiheadTrainArgs(TrainModelArgs):
    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = 80_000

    save_checkpoint_args_cls: ClassVar[type[TaskSaveCheckpointArgs]] = (
        SemanticSegmentationMultiheadSaveCheckpointArgs
    )

    # Backbone args
    backbone_weights: PathLike | None = None
    backbone_args: dict[str, Any] = Field(default_factory=dict)

    gradient_clip_val: float = 0.0

    # Optim
    lr: list[float] | float = [
        0.00001,
        0.00003,
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
    ]
    weight_decay: float = 0.0
    lr_warmup_steps: int = 0

    # Metrics
    metrics: dict[str, dict[str, Any]] | Literal["auto"] = "auto"
    metrics_classwise: dict[str, dict[str, Any]] | None = None
    metric_log_classwise: bool = True

    @model_validator(mode="after")
    def _convert_lr_to_list(self) -> SemanticSegmentationMultiheadTrainArgs:
        """Convert float lr to single-element list."""
        if isinstance(self.lr, float):
            self.lr = [self.lr]
        return self

    def resolve_auto(
        self,
        total_steps: int,
        model_name: str,
        model_init_args: dict[str, Any],
        data_args: TaskDataArgs,
    ) -> None:
        if self.metrics == "auto":
            self.metrics = {
                "miou": {},
            }


class SemanticSegmentationMultiheadTrain(TrainModel):
    task = "semantic_segmentation_multihead"
    train_model_args_cls = SemanticSegmentationMultiheadTrainArgs
    task_model_cls = SemanticSegmentationMultihead
    train_transform_cls = SemanticSegmentationMultiheadTrainTransform
    val_transform_cls = SemanticSegmentationMultiheadValTransform

    def __init__(
        self,
        *,
        model_name: str,
        model_args: SemanticSegmentationMultiheadTrainArgs,
        data_args: MaskSemanticSegmentationDataArgs,
        train_transform_args: SemanticSegmentationMultiheadTrainTransformArgs,
        val_transform_args: SemanticSegmentationMultiheadValTransformArgs,
        load_weights: bool,
    ) -> None:
        super().__init__()
        # Lazy import because torchmetrics is an optional dependency.
        from torchmetrics import ClasswiseWrapper, MeanMetric, MetricCollection

        image_size = no_auto(val_transform_args.image_size)
        normalize = no_auto(val_transform_args.normalize)

        self.model_args = model_args

        # Convert lr to list and store for optimizer creation.
        self.lrs = model_args.lr if isinstance(model_args.lr, list) else [model_args.lr]

        # Generate head names from learning rates.
        head_names = [_format_head_name(lr) for lr in self.lrs]

        self.model = SemanticSegmentationMultihead(
            model=model_name,
            classes=data_args.included_classes,
            head_names=head_names,
            class_ignore_index=(
                data_args.ignore_index if data_args.ignore_classes else None
            ),
            image_size=image_size,
            image_normalize=normalize.model_dump(),
            backbone_weights=model_args.backbone_weights,
            backbone_args=model_args.backbone_args,
            load_weights=load_weights,
        )

        self.loss_fn = CrossEntropyLoss(ignore_index=data_args.ignore_index)

        # Create per-head training and validation loss metrics.
        val_loss_metrics: dict[str, MeanMetric] = {}
        for lr in self.lrs:
            head_name = _format_head_name(lr)
            val_loss_metrics[head_name] = MeanMetric()
        self.val_loss_metrics = ModuleDict(val_loss_metrics)

        # Create metrics from configuration.
        base_metrics = {}
        for metric_name, metric_config in no_auto(model_args.metrics).items():
            base_metrics.update(
                _create_metric(
                    metric_name=metric_name,
                    metric_config=metric_config,
                    num_classes=data_args.num_included_classes,
                    ignore_index=data_args.ignore_index,
                )
            )

        # Create per-head metric collections for training and validation.
        train_metrics: dict[str, MetricCollection] = {}
        val_metrics: dict[str, MetricCollection] = {}
        for lr in self.lrs:
            head_name = _format_head_name(lr)
            # Clone metrics for this head.
            head_train_metrics = {}
            head_val_metrics = {}
            for key, metric in base_metrics.items():
                head_train_metrics[key] = metric.clone()
                head_val_metrics[key] = metric.clone()

            # Type ignore because old torchmetrics versions (0.8) don't have the
            # correct type annotations for MetricCollection.
            train_metrics[head_name] = MetricCollection(
                head_train_metrics,  # type: ignore[arg-type]
                prefix="train_metric_head/",
                postfix=f"_{head_name}",
            )
            val_metrics[head_name] = MetricCollection(
                head_val_metrics,  # type: ignore[arg-type]
                prefix="val_metric_head/",
                postfix=f"_{head_name}",
            )
        self.train_metrics = ModuleDict(train_metrics)
        self.val_metrics = ModuleDict(val_metrics)

        # Create classwise metrics if enabled.
        train_metrics_classwise: dict[str, MetricCollection] = {}
        val_metrics_classwise: dict[str, MetricCollection] = {}
        self.train_metrics_classwise: ModuleDict | None
        self.val_metrics_classwise: ModuleDict | None
        if model_args.metric_log_classwise:
            # If metrics_classwise is None, use metrics from main metrics config.
            if model_args.metrics_classwise is None:
                metrics_classwise_config = no_auto(model_args.metrics)
            else:
                metrics_classwise_config = model_args.metrics_classwise

            class_labels = list(data_args.included_classes.values())

            # Create classwise base metrics.
            classwise_base_metrics = {}
            for metric_name, metric_config in metrics_classwise_config.items():
                base_metrics_classwise = _create_metric(
                    metric_name=metric_name,
                    metric_config=metric_config,
                    num_classes=data_args.num_included_classes,
                    ignore_index=data_args.ignore_index,
                    classwise=True,
                )
                for key, base_metric in base_metrics_classwise.items():
                    # Type ignore because old torchmetrics versions (0.8) don't support
                    # the `prefix` argument.
                    classwise_base_metrics[key] = ClasswiseWrapper(  # type: ignore[call-arg]
                        base_metric,
                        prefix="_",
                        labels=class_labels,
                    )

            # Create per-head classwise metrics.
            for lr in self.lrs:
                head_name = _format_head_name(lr)
                head_train_classwise_metrics = {}
                head_val_classwise_metrics = {}
                for key, metric in classwise_base_metrics.items():
                    head_train_classwise_metrics[key] = metric.clone()
                    head_val_classwise_metrics[key] = metric.clone()

                # Type ignore because old torchmetrics versions (0.8) don't have the
                # correct type annotations for MetricCollection.
                train_metrics_classwise[head_name] = MetricCollection(
                    head_train_classwise_metrics,  # type: ignore[arg-type]
                    prefix="train_metric_head_classwise/",
                    postfix=f"_{head_name}",
                )
                val_metrics_classwise[head_name] = MetricCollection(
                    head_val_classwise_metrics,  # type: ignore[arg-type]
                    prefix="val_metric_head_classwise/",
                    postfix=f"_{head_name}",
                )
            self.train_metrics_classwise = ModuleDict(train_metrics_classwise)
            self.val_metrics_classwise = ModuleDict(val_metrics_classwise)
        else:
            self.train_metrics_classwise = None
            self.val_metrics_classwise = None

    def get_task_model(self) -> SemanticSegmentationMultihead:
        return self.model

    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
    ) -> tuple[Optimizer, LRScheduler]:
        # Create parameter groups for each head with independent learning rates.
        params: list[dict[str, Any]] = []

        for lr in self.lrs:
            head_name = _format_head_name(lr)
            head_module = self.model.class_heads[head_name]

            # Get parameters for this head, separated by weight decay.
            params_wd, params_no_wd = optimizer_helpers.get_weight_decay_parameters(
                [head_module]
            )

            # Filter out parameters with requires_grad=False.
            params_wd = [p for p in params_wd if p.requires_grad]
            params_no_wd = [p for p in params_no_wd if p.requires_grad]

            # Scale learning rate for this head.
            lr_scaled = lr * math.sqrt(
                global_batch_size / self.model_args.default_batch_size
            )

            # Create parameter groups for this head.
            if params_wd:
                params.append(
                    {
                        "name": f"{head_name}_params",
                        "params": params_wd,
                        "lr": lr_scaled,
                    }
                )
            if params_no_wd:
                params.append(
                    {
                        "name": f"{head_name}_no_weight_decay",
                        "params": params_no_wd,
                        "lr": lr_scaled,
                        "weight_decay": 0.0,
                    }
                )

        # Create optimizer with all parameter groups.
        optimizer = SGD(
            params=params,
            weight_decay=self.model_args.weight_decay,
        )

        # Create learning rate scheduler.
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=self.model_args.lr_warmup_steps,
            max_epochs=total_steps,
        )

        return optimizer, scheduler

    def training_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch, step: int
    ) -> TaskStepResult:
        images = batch["image"]
        assert isinstance(images, Tensor), "Images must be a single tensor for training"
        masks = batch["mask"]
        assert isinstance(masks, Tensor), "Masks must be a single tensor for training"

        # Forward through all heads.
        logits_dict = self.model.forward_train(images)

        # Compute loss and update metrics for each head.
        losses: list[Tensor] = []
        log_dict: dict[str, Any] = {}
        for head_name, logits in logits_dict.items():
            # Handle ignore class.
            if self.model.class_ignore_index is not None:
                logits = logits[:, :-1]  # Drop logits for the ignored class.

            # Compute loss.
            loss = self.loss_fn(logits, masks)
            losses.append(loss)

            # Update per-head loss tracker.
            log_dict[f"train_loss_head/{head_name}"] = loss.detach()

            # Update per-head metrics.
            self.train_metrics[head_name].update(logits, masks)  # type: ignore[operator]

            # Update per-head classwise metrics if enabled.
            if self.train_metrics_classwise is not None:
                self.train_metrics_classwise[head_name].update(logits, masks)  # type: ignore[operator]

        # Sum losses for backprop.
        loss_sum = torch.stack(losses).sum()
        # Mean loss for logging.
        loss_mean = torch.stack(losses).mean()
        log_dict["train_loss"] = loss_mean.detach()

        # Add per-head metrics to log dict.
        for head_name in logits_dict.keys():
            log_dict.update(dict(self.train_metrics[head_name].items()))  # type: ignore[operator]

        # Add classwise metrics if enabled.
        if self.train_metrics_classwise is not None:
            for head_name in logits_dict.keys():
                log_dict.update(dict(self.train_metrics_classwise[head_name].items()))  # type: ignore[operator]

        return TaskStepResult(loss=loss_sum, log_dict=log_dict)

    def validation_step(
        self, fabric: Fabric, batch: MaskSemanticSegmentationBatch
    ) -> TaskStepResult:
        images = batch["image"]
        masks = batch["mask"]
        image_sizes = [(image.shape[-2], image.shape[-1]) for image in images]

        # Tile the images.
        crops_list, origins = self.model.tile(images)
        crops = torch.stack(crops_list)

        # Forward through all heads.
        crop_logits_dict = self.model.forward_train(crops)

        # Process each head separately.
        losses: list[Tensor] = []
        log_dict: dict[str, Any] = {}

        for head_name, crop_logits in crop_logits_dict.items():
            # Handle ignore class.
            if self.model.class_ignore_index is not None:
                crop_logits = crop_logits[:, :-1]

            # Un-tile the predictions.
            logits = self.model.untile(
                crop_logits=crop_logits, origins=origins, image_sizes=image_sizes
            )

            # Compute loss and update metrics per image.
            head_loss = torch.tensor(
                0.0, device=crop_logits.device, dtype=crop_logits.dtype
            )
            for image_logits, image_mask in zip(logits, masks):
                image_logits = image_logits.unsqueeze(0)  # Add batch dimension.
                image_mask = image_mask.unsqueeze(0)  # Add batch dimension.
                loss = self.loss_fn(image_logits, image_mask)
                head_loss += loss
                self.val_metrics[head_name].update(image_logits, image_mask)  # type: ignore[operator]
                if self.val_metrics_classwise is not None:
                    self.val_metrics_classwise[head_name].update(  # type: ignore[operator]
                        image_logits, image_mask
                    )
            head_loss /= len(images)
            losses.append(head_loss)

            # Update per-head loss tracker.
            self.val_loss_metrics[head_name].update(head_loss, weight=len(images))  # type: ignore[operator]
            log_dict[f"val_loss_head/{head_name}"] = head_loss.detach()

        # Mean loss across all heads.
        loss_mean = torch.stack(losses).mean()
        log_dict["val_loss"] = loss_mean.detach()

        # Add per-head metrics to log dict.
        for head_name in crop_logits_dict.keys():
            log_dict.update(dict(self.val_metrics[head_name].items()))  # type: ignore[operator]

        # Add classwise metrics if enabled.
        if self.val_metrics_classwise is not None:
            for head_name in crop_logits_dict.keys():
                log_dict.update(dict(self.val_metrics_classwise[head_name].items()))  # type: ignore[operator]

        # Use sum of losses for consistency with training_step.
        loss_sum = torch.stack(losses).sum()
        return TaskStepResult(loss=loss_sum, log_dict=log_dict)

    def set_train_mode(self) -> None:
        self.train()
        self.model.freeze_backbone()

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        if self.model_args.gradient_clip_val > 0:
            fabric.clip_gradients(
                module=self,
                optimizer=optimizer,
                max_norm=self.model_args.gradient_clip_val,
            )


def _create_metric(
    metric_name: str,
    metric_config: dict[str, Any],
    num_classes: int,
    ignore_index: int,
    classwise: bool = False,
) -> dict[str, Any]:
    """Create metrics from configuration.

    Args:
        metric_name: Name of the metric (e.g., "miou").
        metric_config: Configuration dictionary for the metric.
        num_classes: Number of classes.
        ignore_index: Index to ignore during metric calculation.
        classwise: Whether to create classwise metrics.

    Returns:
        Dictionary mapping metric names to metric instances.
    """
    from torchmetrics import Metric
    from torchmetrics.classification import (  # type: ignore[attr-defined]
        MulticlassJaccardIndex,
    )

    metrics: dict[str, Metric] = {}

    if metric_name == "miou":
        key = "miou"
        metrics[key] = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=ignore_index,
            validate_args=False,
            average=None if classwise else "macro",
        )
    else:
        raise ValueError(f"Unsupported metric: {metric_name}")

    return metrics


def _format_head_name(lr: float) -> str:
    """Format learning rate into a head name.

    Args:
        lr: Learning rate value.

    Returns:
        Formatted head name, e.g., "lr0_001" for lr=0.001.
    """
    # Convert to string and replace dot with underscore.
    lr_str = f"{lr:.10f}".rstrip("0").rstrip(".")
    lr_str = lr_str.replace(".", "_")
    return f"lr{lr_str}"
