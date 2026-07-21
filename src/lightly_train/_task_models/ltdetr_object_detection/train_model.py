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
import re
from typing import Any, ClassVar, Literal, cast

import torch
from lightning_fabric import Fabric
from pydantic import AliasChoices, Field, computed_field
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import AdamW, Optimizer  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import (  # type: ignore[attr-defined]
    LinearLR,
    LRScheduler,
)
from typing_extensions import override

from lightly_train._configs.validate import no_auto
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._distributed import reduce_dict
from lightly_train._metrics.detection.task_metric import (
    ObjectDetectionTaskMetric,
    ObjectDetectionTaskMetricArgs,
)
from lightly_train._optim import optimizer_helpers
from lightly_train._pre_post_processing.object_detection import ObjectDetectionOutput
from lightly_train._task_models.ltdetr_object_detection.config import (
    LTDETR_MODEL_REGISTRY,
)
from lightly_train._task_models.ltdetr_object_detection.dino_vit_wrapper import (
    DINOSTAs,
)
from lightly_train._task_models.ltdetr_object_detection.ecvit_vit_wrapper import (
    ECViTBackboneWrapper,
)
from lightly_train._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)
from lightly_train._task_models.ltdetr_object_detection.transforms import (
    DINOv2LTDETRObjectDetectionTrainTransformV2,
    DINOv2LTDETRObjectDetectionValTransformV2,
    LTDETRObjectDetectionTrainTransform,
    LTDETRObjectDetectionTrainTransformArgs,
    LTDETRObjectDetectionValTransform,
    LTDETRObjectDetectionValTransformArgs,
)
from lightly_train._task_models.object_detection_components.dfine_criterion import (
    DFINECriterion,
)
from lightly_train._task_models.object_detection_components.dfine_decoder import (
    DFINETransformer,
)
from lightly_train._task_models.object_detection_components.ema import ModelEMA
from lightly_train._task_models.object_detection_components.flat_cosine import (
    FlatCosineLRScheduler,
)
from lightly_train._task_models.object_detection_components.ltdetr_schedule import (
    resolve_ltdetr_step_schedule,
)
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
from lightly_train._torch_compile import TorchCompileArgs
from lightly_train._torch_helpers import total_gradient_norm
from lightly_train._visualize import object_detection
from lightly_train.types import ObjectDetectionBatch, PathLike

_RTDETRV2_LOSS_WEIGHT_DICT: dict[str, float] = {
    "loss_vfl": 1.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
}
_RTDETRV2_LOSSES: list[str] = ["vfl", "boxes"]
_RTDETRV2_LOSS_NAMES: list[str] = ["loss", *_RTDETRV2_LOSS_WEIGHT_DICT]

_DFINE_EXTRA_LOSS_WEIGHT_DICT: dict[str, float] = {"loss_fgl": 0.15, "loss_ddf": 1.5}
_DFINE_EXTRA_LOSSES: list[str] = ["local"]
_DFINE_LOSS_NAMES: list[str] = [*_RTDETRV2_LOSS_NAMES, *_DFINE_EXTRA_LOSS_WEIGHT_DICT]
logger = logging.getLogger(__name__)

_DINOV2_PREFIX = "dinov2/"


def _decode_predictions_for_metrics(
    model: LTDETRObjectDetection,
    outputs: dict[str, Tensor],
    orig_target_sizes: Tensor,
) -> list[dict[str, Tensor]]:
    labels, boxes, scores = model.postprocessor.decode(
        ObjectDetectionOutput(
            logits=outputs["pred_logits"], boxes=outputs["pred_boxes"]
        ),
        orig_target_sizes,
    )
    return [
        {"labels": labels_i, "boxes": boxes_i, "scores": scores_i}
        for labels_i, boxes_i, scores_i in zip(labels, boxes, scores)
    ]


class BaseLTDETRObjectDetectionTrainArgs(TrainModelArgs):
    """Shared defaults for LTDETRObjectDetectionTrainArgs and
    DINOv2LTDETRObjectDetectionTrainArgsV2.

    Only holds fields whose defaults are identical across both subclasses.
    Fields that differ (lr, backbone_lr_factor, scheduler_name, lr_warmup_steps,
    patch_size) as well as default_batch_size/default_steps and resolve_auto are
    defined individually on each subclass.
    """

    backbone_weights: PathLike | None = None
    backbone_url: str = ""
    backbone_args: dict[str, Any] = {}
    backbone_freeze: bool = False
    decoder_name: Literal["auto", "rtdetrv2", "dfine"] = "auto"

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
        default_factory=lambda: dict(_RTDETRV2_LOSS_WEIGHT_DICT)
    )
    losses: list[str] = Field(default_factory=lambda: list(_RTDETRV2_LOSSES))
    loss_alpha: float = 0.75
    loss_gamma: float = 2.0

    # Miscellaneous
    gradient_clip_val: float = 0.1

    # Optimizer configuration
    weight_decay: float = Field(
        default=1e-4,
        validation_alias=AliasChoices("weight_decay", "optimizer_weight_decay"),
    )
    optimizer_betas: tuple[float, float] = (0.9, 0.999)

    # Scheduler configuration
    scheduler_start_factor: float = 0.01
    scheduler_flat_steps: int | Literal["auto"] = "auto"
    scheduler_no_aug_steps: int | Literal["auto"] = "auto"

    def _resolve_decoder_name(
        self, model_name: str, model_init_args: dict[str, Any]
    ) -> None:
        """Resolve the decoder from user args, checkpoint metadata, or architecture."""
        checkpoint_decoder_name = model_init_args.get("decoder_name")
        if self.decoder_name != "auto":
            if (
                checkpoint_decoder_name is not None
                and self.decoder_name != checkpoint_decoder_name
            ):
                logger.warning(
                    f"Requested decoder_name={self.decoder_name!r} differs from "
                    f"the checkpoint's decoder_name={checkpoint_decoder_name!r}; "
                    "the checkpoint decoder weights will not load cleanly."
                )
            return

        if checkpoint_decoder_name is not None:
            self.decoder_name = checkpoint_decoder_name
        else:
            config = LTDETR_MODEL_REGISTRY.get(alias=model_name)()
            self.decoder_name = config.transformer.decoder_name

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_loss_weight_dict(self) -> dict[str, float]:
        if no_auto(self.decoder_name) == "dfine":
            return {**_DFINE_EXTRA_LOSS_WEIGHT_DICT, **self.loss_weight_dict}
        return dict(self.loss_weight_dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_losses(self) -> list[str]:
        if no_auto(self.decoder_name) == "dfine":
            return [
                *self.losses,
                *(name for name in _DFINE_EXTRA_LOSSES if name not in self.losses),
            ]
        return list(self.losses)


class LTDETRObjectDetectionTrainArgs(BaseLTDETRObjectDetectionTrainArgs):
    default_batch_size: ClassVar[int] = 32
    default_steps: ClassVar[int] = (
        266_112  # 6x ECDet-S schedule (72 epochs at batch 32)
    )

    patch_size: int | Literal["auto"] | None = "auto"

    # Optimizer configuration
    lr: float = Field(
        default=5e-4,
        validation_alias=AliasChoices("lr", "optimizer_lr"),
    )

    # Per-parameter-group overrides
    backbone_lr_factor: float = 0.05

    # Scheduler configuration
    scheduler_name: Literal["linear", "flat-cosine"] = "flat-cosine"
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
        data_args: TaskDataArgs,
    ) -> None:
        if self.patch_size == "auto":
            patch_size = model_init_args.get("patch_size", None)
            if patch_size is not None:
                self.patch_size = int(patch_size)
            else:
                # EdgeCrafter (ECViT) backbones all use a fixed patch size of 16
                # (the ECViT-NN uses a ConvPyramidPatchEmbed that only supports
                # patch_size=16). Resolve that here so the train/val transforms
                # can pick the right image-size divisor and scale-jitter base.
                #
                # Use the task model's ``parse_model_name`` (not the lower-level
                # ``package_helpers.parse_model_name``) because it also expands
                # short LTDETRv2 names (e.g. ``ltdetrv2-s``) into the
                # EdgeCrafter package/backbone shape used by the parser. This
                # runs before the task-model constructor, so without it the
                # aliases would raise
                # ``Unable to resolve patch_size='auto'`` here.
                try:
                    package_name = LTDETRObjectDetection.parse_model_name(
                        model_name=model_name
                    )["package_name"]
                except ValueError:
                    package_name = ""
                if package_name == "edgecrafter":
                    self.patch_size = 16
                else:
                    match = re.match(
                        r"dinov3/(?P<model_size>vit(t|s|l|b|g|h|7b))(?P<patch_size>\d+).*",
                        model_name,
                    )

                    if match is not None:
                        self.patch_size = int(match.group("patch_size"))
                    elif re.match(r"dinov3/convnext.*", model_name) is not None:
                        self.patch_size = None
                    else:
                        raise ValueError(
                            "Unable to resolve patch_size='auto' for model "
                            f"{model_name!r}. Please provide a concrete patch_size."
                        )
        self._resolve_decoder_name(
            model_name=model_name, model_init_args=model_init_args
        )

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


class DINOv2LTDETRObjectDetectionTrainArgsV2(BaseLTDETRObjectDetectionTrainArgs):
    default_batch_size: ClassVar[int] = 16
    default_steps: ClassVar[int] = (
        100_000 // 16 * 72
    )  # TODO (Lionel, 10/25): Adjust default steps.

    # DINOv2 ViT backbones are always patch-14 (vits14/vitb14/vitl14/vitg14), matching
    # LTDETR_MODEL_REGISTRY's dinov2 backbone_args.
    patch_size: int = 14

    # Optimizer configuration
    lr: float = Field(
        default=1e-4,
        validation_alias=AliasChoices("lr", "optimizer_lr"),
    )

    # Per-parameter-group overrides
    backbone_lr_factor: float = 1e-2

    # Scheduler configuration
    scheduler_name: Literal["linear", "flat-cosine"] = "linear"
    lr_warmup_steps: int = Field(
        default=2000,
        validation_alias=AliasChoices("lr_warmup_steps", "scheduler_warmup_steps"),
    )

    def resolve_auto(
        self,
        total_steps: int,
        gradient_accumulation_steps: int,
        train_num_batches: int,
        model_name: str,
        model_init_args: dict[str, Any],
        data_args: TaskDataArgs,
    ) -> None:
        self._resolve_decoder_name(
            model_name=model_name, model_init_args=model_init_args
        )
        if self.scheduler_flat_steps == "auto" or self.scheduler_no_aug_steps == "auto":
            scheduler_step_schedule = resolve_ltdetr_step_schedule(
                total_steps=total_steps,
                train_num_batches=train_num_batches,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            if self.scheduler_flat_steps == "auto":
                self.scheduler_flat_steps = scheduler_step_schedule.step_flat
            if self.scheduler_no_aug_steps == "auto":
                self.scheduler_no_aug_steps = (
                    total_steps - scheduler_step_schedule.step_stop
                )


class LTDETRObjectDetectionTrain(TrainModel):
    task = "object_detection"
    train_model_args_cls = LTDETRObjectDetectionTrainArgs
    task_metric_args_cls = ObjectDetectionTaskMetricArgs
    task_model_cls = LTDETRObjectDetection
    train_transform_cls = LTDETRObjectDetectionTrainTransform
    val_transform_cls = LTDETRObjectDetectionValTransform
    torch_compile_args_cls = TorchCompileArgs

    @override
    @classmethod
    def get_train_model_args_cls(
        cls, model_name: str
    ) -> type[LTDETRObjectDetectionTrainArgs | DINOv2LTDETRObjectDetectionTrainArgsV2]:
        if model_name.startswith(_DINOV2_PREFIX):
            return DINOv2LTDETRObjectDetectionTrainArgsV2
        return LTDETRObjectDetectionTrainArgs

    @override
    @classmethod
    def get_train_transform_cls(
        cls, model_name: str
    ) -> type[
        LTDETRObjectDetectionTrainTransform
        | DINOv2LTDETRObjectDetectionTrainTransformV2
    ]:
        if model_name.startswith(_DINOV2_PREFIX):
            return DINOv2LTDETRObjectDetectionTrainTransformV2
        return LTDETRObjectDetectionTrainTransform

    @override
    @classmethod
    def get_val_transform_cls(
        cls, model_name: str
    ) -> type[
        LTDETRObjectDetectionValTransform | DINOv2LTDETRObjectDetectionValTransformV2
    ]:
        if model_name.startswith(_DINOV2_PREFIX):
            return DINOv2LTDETRObjectDetectionValTransformV2
        return LTDETRObjectDetectionValTransform

    def __init__(
        self,
        *,
        model_name: str,
        model_args: LTDETRObjectDetectionTrainArgs
        | DINOv2LTDETRObjectDetectionTrainArgsV2,
        data_args: YOLOObjectDetectionDataArgs,
        train_transform_args: LTDETRObjectDetectionTrainTransformArgs,
        val_transform_args: LTDETRObjectDetectionValTransformArgs,
        load_weights: bool,
        metric_args: ObjectDetectionTaskMetricArgs,
        gradient_accumulation_steps: int,
    ) -> None:
        super().__init__()

        self.model_args = model_args
        self.data_args = data_args

        # Get the normalization.
        normalize = no_auto(val_transform_args.normalize)
        normalize_dict: dict[str, Any] | None
        self._normalize = normalize

        if normalize is None:
            normalize_dict = None
        else:
            normalize_dict = normalize.model_dump()

        backbone_args: dict[str, Any] | None = model_args.backbone_args
        if not backbone_args:
            backbone_args = None
        decoder_name = cast(
            Literal["rtdetrv2", "dfine"], no_auto(model_args.decoder_name)
        )
        self.model: LTDETRObjectDetection = LTDETRObjectDetection(
            model_name=model_name,
            image_size=no_auto(val_transform_args.image_size),
            classes=data_args.included_classes,
            image_normalize=normalize_dict,
            backbone_freeze=model_args.backbone_freeze,
            backbone_args=backbone_args,
            patch_size=no_auto(model_args.patch_size),
            backbone_weights=model_args.backbone_weights,
            decoder_name=decoder_name,
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

        criterion: DFINECriterion | RTDETRCriterionv2
        if decoder_name == "dfine":
            self.train_loss_names = _DFINE_LOSS_NAMES
            self.val_loss_names = _RTDETRV2_LOSS_NAMES
            if not isinstance(self.model.decoder, DFINETransformer):
                raise TypeError("decoder='dfine' requires a DFINETransformer decoder.")
            criterion = DFINECriterion(  # type: ignore[no-untyped-call]
                matcher=matcher,
                weight_dict=model_args.effective_loss_weight_dict,
                losses=model_args.effective_losses,
                alpha=model_args.loss_alpha,
                gamma=model_args.loss_gamma,
                num_classes=len(data_args.included_classes),
                reg_max=self.model.decoder.reg_max,
            )
        else:
            self.train_loss_names = _RTDETRV2_LOSS_NAMES
            self.val_loss_names = _RTDETRV2_LOSS_NAMES
            criterion = RTDETRCriterionv2(  # type: ignore[no-untyped-call]
                matcher=matcher,
                weight_dict=model_args.effective_loss_weight_dict,
                losses=model_args.effective_losses,
                alpha=model_args.loss_alpha,
                gamma=model_args.loss_gamma,
                num_classes=len(data_args.included_classes),
            )
        self.criterion = criterion

        class_names = list(data_args.included_classes.values())
        self.metric_args = metric_args
        self.train_metrics = ObjectDetectionTaskMetric(
            task_metric_args=metric_args,
            split="train",
            class_names=class_names,
            box_format="xyxy",
            loss_names=self.train_loss_names,
            train_loss_running_mean_window=gradient_accumulation_steps,
        )
        self.val_metrics = ObjectDetectionTaskMetric(
            task_metric_args=metric_args,
            split="val",
            class_names=class_names,
            box_format="xyxy",
            loss_names=self.val_loss_names,
        )

        # TODO(Nauryz, 04/2026): These visualization thresholds are currently
        # hardcoded, but we may want to make them configurable in the future
        # (with logger_args).

        self.viz_score_threshold = 0.1
        self.viz_max_images = 4

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
            loss_dict=_get_loss_log_dict(
                total_loss=total_loss,
                loss_dict=loss_dict,
                loss_names=self.train_loss_names,
            ),
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
            results = _decode_predictions_for_metrics(
                model=self.model,
                outputs=outputs,
                orig_target_sizes=orig_target_sizes_tensor,
            )
            self.train_metrics.update_with_predictions(results, targets)

        return TaskStepResult(
            loss=total_loss,
            log_dict={},
            metrics=self.train_metrics,
            visualization=object_detection.ObjectDetectionTaskStepVisualization(
                batch=batch,
                class_names=self.model.included_classes,
                image_normalize=self.model.image_normalize,
                max_images=self.viz_max_images,
                score_threshold=self.viz_score_threshold,
            ),
        )

    def on_train_batch_end(self) -> None:
        if self.ema_model is not None:
            self.ema_model.update(self.model)

    def validation_step(
        self, fabric: Fabric, batch: ObjectDetectionBatch, step: int
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
        results = _decode_predictions_for_metrics(
            model=self.model,
            outputs=outputs,
            orig_target_sizes=orig_target_sizes_tensor,
        )

        # Metrics
        self.val_metrics.update_with_losses(
            loss_dict=_get_loss_log_dict(
                total_loss=total_loss,
                loss_dict=loss_dict,
                loss_names=self.val_loss_names,
            ),
            weight=samples.shape[0],
        )
        self.val_metrics.update_with_predictions(results, targets)

        return TaskStepResult(
            loss=total_loss,
            log_dict={},
            metrics=self.val_metrics,
            visualization=object_detection.ObjectDetectionTaskStepVisualization(
                batch=batch,
                results=results,
                class_names=self.model.included_classes,
                image_normalize=self.model.image_normalize,
                score_threshold=self.viz_score_threshold,
                max_images=self.viz_max_images,
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

        backbone = self.model.backbone
        if isinstance(backbone, DINOSTAs):
            # Only the pretrained ViT gets the low backbone LR.
            backbone_params = list(backbone.backbone_model.parameters())
            # The connector modules (sta, convs, norms) are randomly initialized and
            # are merged into the detector group to train at the full LR.
            vit_params_ids = {id(p) for p in backbone_params}
            connector_params = [
                p for p in backbone.parameters() if id(p) not in vit_params_ids
            ]
        elif isinstance(backbone, ECViTBackboneWrapper):
            # ECViTModelWrapper has two parts:
            #   - self.backbone  (VisionTransformer) - loaded with pretrained
            #     weights, so it gets the low backbone_lr_factor.
            #   - self.projector (nn.ModuleList of ConvNormLayer) - freshly
            #     initialized, so it is merged into the detector group to
            #     train at the full LR (same split as the DINOv3 ViT branch).
            ecvit_wrapper = backbone._model_wrapper  # type: ignore[attr-defined]
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
