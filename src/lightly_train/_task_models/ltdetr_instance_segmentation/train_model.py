#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
import logging
import math
from typing import Any, ClassVar, Literal

import torch
from lightning_fabric import Fabric
from pydantic import AliasChoices, Field
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import AdamW, Optimizer  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import (  # type: ignore[attr-defined]
    LinearLR,
    LRScheduler,
)

from lightly_train._configs.validate import no_auto
from lightly_train._data.instance_segmentation_dataset import (
    COCOInstanceSegmentationDataArgs,
    YOLOInstanceSegmentationDataArgs,
)
from lightly_train._distributed import reduce_dict
from lightly_train._metrics.instance_segmentation.task_metric import (
    InstanceSegmentationTaskMetric,
    InstanceSegmentationTaskMetricArgs,
)
from lightly_train._optim import optimizer_helpers
from lightly_train._task_models.instance_segmentation_components.edgecrafter_criterion import (
    EdgeCrafterInstanceSegmentationCriterion,
)
from lightly_train._task_models.instance_segmentation_components.matcher import (
    MaskAwareHungarianMatcher,
)
from lightly_train._task_models.ltdetr_instance_segmentation.task_model import (
    LTDETRInstanceSegmentation,
)
from lightly_train._task_models.ltdetr_instance_segmentation.transforms import (
    LTDETRInstanceSegmentationTrainTransform,
    LTDETRInstanceSegmentationTrainTransformArgs,
    LTDETRInstanceSegmentationValTransform,
    LTDETRInstanceSegmentationValTransformArgs,
)
from lightly_train._task_models.ltdetr_object_detection.ecvit_vit_wrapper import (
    ECViTBackboneWrapper,
)
from lightly_train._task_models.object_detection_components.ema import ModelEMA
from lightly_train._task_models.object_detection_components.flat_cosine import (
    FlatCosineLRScheduler,
)
from lightly_train._task_models.object_detection_components.ltdetr_schedule import (
    resolve_ltdetr_step_schedule,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train._torch_compile import TorchCompileArgs
from lightly_train._torch_helpers import total_gradient_norm
from lightly_train._visualize import instance_segmentation
from lightly_train.types import InstanceSegmentationBatch, PathLike

logger = logging.getLogger(__name__)

_LTDETR_SEG_LOSS_WEIGHT_DICT: dict[str, float] = {
    "loss_vfl": 1.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
    "loss_mask": 5.0,
    "loss_dice": 5.0,
    "loss_fgl": 0.15,
    "loss_ddf": 1.5,
}
_LTDETR_SEG_LOSSES: list[str] = ["vfl", "boxes", "masks", "local"]

# Loss names logged during validation (no FGL/DDF, matching detection's val logging).
_LTDETR_SEG_VAL_LOSS_NAMES: list[str] = [
    "loss",
    "loss_vfl",
    "loss_bbox",
    "loss_giou",
    "loss_mask",
    "loss_dice",
]
_LTDETR_SEG_TRAIN_LOSS_NAMES: list[str] = ["loss", *_LTDETR_SEG_LOSS_WEIGHT_DICT]

# LT-DETR training-codepath guard (TRN-2312): enforce a minimum bbox size in
# pixels on the LTDETR training code path so that degenerate targets /
# predictions do not destabilize the matcher and losses.
_LTDETR_MIN_BBOX_SIZE_PX: float = 4.0


class LTDETRInstanceSegmentationTrainArgs(TrainModelArgs):
    """Training args for LTDETR (EdgeCrafter/ECViT) instance segmentation.

    Standalone: mirrors the LTDETR object-detection recipe but does not inherit from
    it. ECSeg is D-FINE based, so the effective losses always include the D-FINE FGL/DDF
    losses on top of the vfl / box / mask / dice losses.
    """

    default_batch_size: ClassVar[int] = 32
    default_steps: ClassVar[int] = (
        266_112  # 6x ECDet-S schedule (72 epochs at batch 32)
    )

    # ECViT (EdgeCrafter) backbones all use a fixed patch size of 16.
    patch_size: int | Literal["auto"] | None = "auto"

    # Backbone configuration. The task model rejects non-None backbone_args /
    # backbone_weights for ECViT instance segmentation, so these stay at their defaults.
    backbone_weights: PathLike | None = None
    backbone_url: str = ""
    backbone_args: dict[str, Any] = {}
    backbone_freeze: bool = False

    # EMA configuration.
    use_ema_model: bool = True
    ema_momentum: float = 0.9999
    ema_warmup_steps: int = 2000

    # Matcher: detection costs + mask/dice costs.
    matcher_weight_dict: dict[str, float] = Field(
        default_factory=lambda: {
            "cost_class": 2.0,
            "cost_bbox": 5.0,
            "cost_giou": 2.0,
            "cost_mask": 2.0,
            "cost_dice": 2.0,
        }
    )
    matcher_alpha: float = 0.25
    matcher_gamma: float = 2.0

    # Criterion: detection losses + mask/dice losses.
    loss_weight_dict: dict[str, float] = Field(
        default_factory=lambda: dict(_LTDETR_SEG_LOSS_WEIGHT_DICT)
    )
    losses: list[str] = Field(default_factory=lambda: list(_LTDETR_SEG_LOSSES))
    loss_alpha: float = 0.75
    loss_gamma: float = 2.0

    # Point sampling for mask matching/loss.
    # ECSeg derives the point budget from the mask resolution with ratio 16.
    mask_point_sample_ratio: int | None = 16
    # Fallback point budget when mask_point_sample_ratio is disabled.
    mask_num_points: int = 12544
    mask_oversample_ratio: float = 3.0
    mask_importance_sample_ratio: float = 0.75

    # Miscellaneous.
    gradient_clip_val: float = 0.1

    # Optimizer configuration.
    lr: float = Field(
        default=5e-4,
        validation_alias=AliasChoices("lr", "optimizer_lr"),
    )
    backbone_lr_factor: float = 0.05
    weight_decay: float = Field(
        default=1e-4,
        validation_alias=AliasChoices("weight_decay", "optimizer_weight_decay"),
    )
    optimizer_betas: tuple[float, float] = (0.9, 0.999)

    # Scheduler configuration.
    scheduler_name: Literal["linear", "flat-cosine"] = "flat-cosine"
    scheduler_start_factor: float = 0.01
    scheduler_flat_steps: int | Literal["auto"] = "auto"
    scheduler_no_aug_steps: int | Literal["auto"] = "auto"
    lr_warmup_steps: int | Literal["auto"] = Field(
        default="auto",
        validation_alias=AliasChoices("lr_warmup_steps", "scheduler_warmup_steps"),
    )

    def resolve_auto(
        self,
        total_steps: int,
        gradient_accumulation_steps: int,
        train_num_batches: int,
        model_name: str,
        model_init_args: dict[str, Any],
        data_args: Any,
    ) -> None:
        if self.patch_size == "auto":
            patch_size = model_init_args.get("patch_size", None)
            # EdgeCrafter (ECViT) backbones all use a fixed patch size of 16.
            self.patch_size = int(patch_size) if patch_size is not None else 16

        if (
            self.lr_warmup_steps == "auto"
            or self.scheduler_flat_steps == "auto"
            or self.scheduler_no_aug_steps == "auto"
        ):
            scheduler_step_schedule = resolve_ltdetr_step_schedule(
                total_steps=total_steps,
                train_num_batches=train_num_batches,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            if self.lr_warmup_steps == "auto":
                self.lr_warmup_steps = scheduler_step_schedule.step_start
            if self.scheduler_flat_steps == "auto":
                self.scheduler_flat_steps = scheduler_step_schedule.step_flat
            if self.scheduler_no_aug_steps == "auto":
                self.scheduler_no_aug_steps = (
                    total_steps - scheduler_step_schedule.step_stop
                )


class LTDETRInstanceSegmentationTrain(TrainModel):
    task = "instance_segmentation"
    train_model_args_cls = LTDETRInstanceSegmentationTrainArgs
    task_metric_args_cls = InstanceSegmentationTaskMetricArgs
    task_model_cls = LTDETRInstanceSegmentation
    train_transform_cls = LTDETRInstanceSegmentationTrainTransform
    val_transform_cls = LTDETRInstanceSegmentationValTransform
    torch_compile_args_cls = TorchCompileArgs

    def __init__(
        self,
        *,
        model_name: str,
        model_args: LTDETRInstanceSegmentationTrainArgs,
        data_args: YOLOInstanceSegmentationDataArgs | COCOInstanceSegmentationDataArgs,
        train_transform_args: LTDETRInstanceSegmentationTrainTransformArgs,
        val_transform_args: LTDETRInstanceSegmentationValTransformArgs,
        load_weights: bool,
        metric_args: InstanceSegmentationTaskMetricArgs,
        gradient_accumulation_steps: int,
    ) -> None:
        super().__init__()

        self.model_args = model_args
        self.data_args = data_args

        # Get the normalization.
        normalize = no_auto(val_transform_args.normalize)
        self._normalize = normalize

        if normalize is None:
            normalize_dict = None
        else:
            normalize_dict = normalize.model_dump()

        backbone_args: dict[str, Any] | None = model_args.backbone_args
        if not backbone_args:
            backbone_args = None
        self.model: LTDETRInstanceSegmentation = LTDETRInstanceSegmentation(
            model_name=model_name,
            image_size=no_auto(val_transform_args.image_size),
            classes=data_args.included_classes,
            image_normalize=normalize_dict,
            backbone_freeze=model_args.backbone_freeze,
            backbone_args=backbone_args,
            patch_size=no_auto(model_args.patch_size),
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

        matcher = MaskAwareHungarianMatcher(
            weight_dict=model_args.matcher_weight_dict,
            alpha=model_args.matcher_alpha,
            gamma=model_args.matcher_gamma,
            num_points=model_args.mask_num_points,
            mask_point_sample_ratio=model_args.mask_point_sample_ratio,
        )
        self.train_loss_names = _LTDETR_SEG_TRAIN_LOSS_NAMES
        self.val_loss_names = _LTDETR_SEG_VAL_LOSS_NAMES
        # EdgeCrafterInstanceSegmentationCriterion subclasses DFINECriterion, so it needs
        # reg_max for the FGL/DDF losses (the ECSeg decoder subclasses DFINETransformer).
        self.criterion = EdgeCrafterInstanceSegmentationCriterion(  # type: ignore[no-untyped-call]
            matcher=matcher,
            weight_dict=model_args.loss_weight_dict,
            losses=model_args.losses,
            alpha=model_args.loss_alpha,
            gamma=model_args.loss_gamma,
            num_classes=len(data_args.included_classes),
            reg_max=self.model.decoder.reg_max,
            num_points=model_args.mask_num_points,
            oversample_ratio=model_args.mask_oversample_ratio,
            importance_sample_ratio=model_args.mask_importance_sample_ratio,
        )

        class_names = list(data_args.included_classes.values())
        self.metric_args = metric_args
        self.train_metrics = InstanceSegmentationTaskMetric(
            task_metric_args=metric_args,
            split="train",
            class_names=class_names,
            loss_names=self.train_loss_names,
            train_loss_running_mean_window=gradient_accumulation_steps,
        )
        self.val_metrics = InstanceSegmentationTaskMetric(
            task_metric_args=metric_args,
            split="val",
            class_names=class_names,
            loss_names=self.val_loss_names,
        )

        # TODO(Nauryz, 04/2026): These visualization thresholds are currently
        # hardcoded, but we may want to make them configurable in the future
        # (with logger_args).
        self.viz_score_threshold = 0.1
        self.viz_max_images = 4
        self.viz_alpha = 0.5

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
        self.criterion.train()
        if self.model_args.backbone_freeze:
            self.model.freeze_backbone()

    def training_step(
        self, fabric: Fabric, batch: InstanceSegmentationBatch, step: int
    ) -> TaskStepResult:
        samples, boxes, classes, binary_masks = (
            batch["image"],
            batch["bboxes"],
            batch["classes"],
            batch["binary_masks"],
        )
        assert isinstance(samples, Tensor), (
            "Images must be a single tensor for training"
        )
        # Targets carry masks for the mask-aware matcher + criterion. Boxes are already
        # normalized cxcywh (yolo format), as expected by the criterion.
        targets = [
            {"boxes": boxes, "labels": classes, "masks": binary_masks["masks"]}
            for boxes, classes, binary_masks in zip(boxes, classes, binary_masks)
        ]
        outputs = self.model._forward_train(x=samples, targets=targets)

        # ``image_size`` matches the model's input resolution so the criterion can
        # enforce the minimum bbox size guard on the LTDETR training code path.
        loss_dict = self.criterion(
            outputs=outputs,
            targets=targets,
            epoch=None,
            step=None,
            global_step=None,
            world_size=fabric.world_size,
            image_size=self.model.image_size,
            min_bbox_size_px=_LTDETR_MIN_BBOX_SIZE_PX,
        )
        total_loss = sum(loss_dict.values())

        # Average loss dict across devices.
        loss_dict = reduce_dict(loss_dict)

        self.train_metrics.update_with_losses(
            loss_dict=_get_loss_log_dict(
                total_loss=total_loss,
                loss_dict=loss_dict,
                loss_names=self.train_loss_names,
            ),
            weight=samples.shape[0],
        )
        if self.metric_args.train:
            # The postprocessor resizes masks to the given (W, H) per image. We use the
            # model-input resolution so predicted masks match the target masks carried
            # in ``binary_masks``.
            orig_target_sizes = _orig_target_sizes(samples)
            results = self.model.postprocessor(
                outputs, orig_target_sizes=orig_target_sizes
            )
            self.train_metrics.update_with_predictions(results, batch["binary_masks"])

        return TaskStepResult(
            loss=total_loss,
            log_dict={},
            metrics=self.train_metrics,
            visualization=instance_segmentation.InstanceSegmentationTaskStepVisualization(
                batch=batch,
                class_names=self.model.included_classes,
                image_normalize=self.model.image_normalize,
                max_images=self.viz_max_images,
                alpha=self.viz_alpha,
                score_threshold=self.viz_score_threshold,
            ),
        )

    def on_train_batch_end(self) -> None:
        if self.ema_model is not None:
            self.ema_model.update(self.model)

    def validation_step(
        self, fabric: Fabric, batch: InstanceSegmentationBatch, step: int
    ) -> TaskStepResult:
        images, boxes, classes, binary_masks = (
            batch["image"],
            batch["bboxes"],
            batch["classes"],
            batch["binary_masks"],
        )
        # Val images are resized to a fixed size by the val transform; stack into a
        # single batch tensor for the forward pass.
        samples = images if isinstance(images, Tensor) else torch.stack(list(images))
        targets = [
            {"boxes": boxes, "labels": classes, "masks": binary_masks["masks"]}
            for boxes, classes, binary_masks in zip(boxes, classes, binary_masks)
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
                image_size=self.model.image_size,
                min_bbox_size_px=_LTDETR_MIN_BBOX_SIZE_PX,
            )

        total_loss = sum(loss_dict.values())

        # Average loss dict across devices.
        loss_dict = reduce_dict(loss_dict)

        orig_target_sizes = _orig_target_sizes(images)
        results: list[dict[str, Tensor]] = model_to_use.postprocessor(  # type: ignore[operator]
            outputs, orig_target_sizes=orig_target_sizes
        )

        self.val_metrics.update_with_losses(
            loss_dict=_get_loss_log_dict(
                total_loss=total_loss,
                loss_dict=loss_dict,
                loss_names=self.val_loss_names,
            ),
            weight=samples.shape[0],
        )
        self.val_metrics.update_with_predictions(results, batch["binary_masks"])

        return TaskStepResult(
            loss=total_loss,
            log_dict={},
            metrics=self.val_metrics,
            visualization=instance_segmentation.InstanceSegmentationTaskStepVisualization(
                batch=batch,
                predictions=results,
                class_names=self.model.included_classes,
                image_normalize=self.model.image_normalize,
                max_images=self.viz_max_images,
                alpha=self.viz_alpha,
                score_threshold=self.viz_score_threshold,
            ),
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

        # ECViTModelWrapper has two parts:
        #   - self.backbone  (VisionTransformer) - loaded with pretrained weights, so it
        #     gets the low backbone_lr_factor.
        #   - self.projector (nn.ModuleList of ConvNormLayer) - freshly initialized, so
        #     it is merged into the detector group to train at the full LR.
        backbone = self.model.backbone
        if isinstance(backbone, ECViTBackboneWrapper):
            ecvit_wrapper = backbone._model_wrapper
            backbone_params = list(ecvit_wrapper.backbone.parameters())
            connector_params = list(ecvit_wrapper.projector.parameters())
        else:
            backbone_params = list(backbone.parameters())
            connector_params = []

        backbone_params_wd = [p for p in backbone_params if p not in params_no_wd]
        backbone_params_no_wd = [p for p in backbone_params if p in params_no_wd]
        if backbone_params_wd:
            param_groups.append(
                {
                    "name": "backbone",
                    "params": backbone_params_wd,
                    "lr": backbone_lr,
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

        detector_params = (
            connector_params
            + list(self.model.encoder.parameters())
            + list(self.model.decoder.parameters())
        )
        detector_params_wd = [p for p in detector_params if p not in params_no_wd]
        detector_params_no_wd = [p for p in detector_params if p in params_no_wd]
        if detector_params_wd:
            param_groups.append(
                {
                    "name": "detector",
                    "params": detector_params_wd,
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

        optim = AdamW(
            param_groups,
            lr=lr,
            betas=self.model_args.optimizer_betas,
            weight_decay=self.model_args.weight_decay,
        )
        scheduler: LRScheduler
        if self.model_args.scheduler_name == "linear":
            warmup_steps = no_auto(self.model_args.lr_warmup_steps)
            if warmup_steps > total_steps:
                logger.warning(
                    f"{self.model_args.scheduler_name} scheduler has "
                    f"lr_warmup_steps={warmup_steps} "
                    f"and total_steps={total_steps}; the schedule will not complete "
                    "as intended."
                )
            scheduler = LinearLR(
                optimizer=optim,
                total_iters=warmup_steps,
                start_factor=self.model_args.scheduler_start_factor,
            )
        elif self.model_args.scheduler_name == "flat-cosine":
            scheduler = FlatCosineLRScheduler(
                optimizer=optim,
                total_steps=total_steps,
                warmup_steps=no_auto(self.model_args.lr_warmup_steps),
                flat_steps=no_auto(self.model_args.scheduler_flat_steps),
                no_aug_steps=no_auto(self.model_args.scheduler_no_aug_steps),
            )
        else:
            raise ValueError(
                f"Unknown scheduler: {self.model_args.scheduler_name!r}. "
                "Expected 'linear' or 'flat-cosine'."
            )

        return optim, scheduler

    def get_task_model(self) -> TaskModel:
        return self.model

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


def _orig_target_sizes(images: Tensor | list[Tensor]) -> Tensor:
    """Returns per-image ``(W, H)`` sizes for the postprocessor.

    The ECSeg postprocessor resizes predicted masks to these sizes; using the
    model-input resolution keeps predicted masks aligned with the target masks
    carried in the batch.
    """
    if isinstance(images, Tensor):
        height, width = images.shape[-2:]
        sizes = [[int(width), int(height)]] * images.shape[0]
        device = images.device
    else:
        sizes = [[int(image.shape[-1]), int(image.shape[-2])] for image in images]
        device = images[0].device
    return torch.tensor(sizes, device=device)


def _get_loss_log_dict(
    *,
    total_loss: Tensor,
    loss_dict: dict[str, Tensor],
    loss_names: list[str],
) -> dict[str, Tensor]:
    zero = total_loss.new_zeros(())
    log_dict = {"loss": total_loss.detach()}
    for loss_name in loss_names:
        if loss_name == "loss":
            continue
        if loss_name == "loss_ddf":
            loss_values = [
                v.detach()
                for k, v in loss_dict.items()
                if k == "loss_ddf" or k.startswith("loss_ddf_")
            ]
            if not loss_values:
                raise KeyError(
                    "No loss entries found for 'loss_ddf'. Available losses: "
                    f"{sorted(loss_dict.keys())}"
                )
            log_dict[loss_name] = sum(loss_values, start=zero)
        else:
            log_dict[loss_name] = loss_dict[loss_name].detach()
    return log_dict
