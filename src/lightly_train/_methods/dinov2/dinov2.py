#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import torch
from lightly.loss import (
    KoLeoLoss,
)  # we use LightlySSL's KoLeoLoss for better numerical stability
from lightly.models.modules.heads import DINOProjectionHead
from lightly.models.utils import update_momentum
from lightly.utils import optim
from lightly.utils.scheduler import cosine_schedule
from torch import Tensor
from torch.nn import Flatten
from torch.optim.optimizer import Optimizer

from lightly_train import _scaling
from lightly_train._configs.validate import no_auto
from lightly_train._methods.dinov2.dinov2_loss import (
    DINOLoss,
    iBOTPatchLoss,
)  # we use the original DINOLoss and iBOTPatchLoss
from lightly_train._methods.dinov2.dinov2_transform import (
    DINOv2Transform,
)
from lightly_train._methods.method import Method, TrainingStepResult
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.sgd_args import SGDArgs
from lightly_train._optim.trainable_modules import TrainableModules
from lightly_train._scaling import IMAGENET_SIZE, ScalingInfo
from lightly_train._transforms.transform import (
    MethodTransform,
)
from lightly_train.types import Batch


@dataclass
class DINOv2TrainingStepResult:
    dino_global_loss: Tensor
    dino_local_loss: Tensor | None
    ibot_loss: Tensor
    koleo_loss: Tensor


class DINOv2Args(MethodArgs):
    """Args for DINOv2 method for ImageNet dataset."""

    # crops
    global_crops_number: int = 2
    local_crops_number: int = 8

    # projection head
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    output_dim: int | Literal["auto"] = "auto"
    batch_norm: bool = False
    student_freeze_last_layer_epochs: int = 1
    norm_last_layer: bool = False
    ibot_separate_head: bool = False

    # loss
    dino_loss_weight: float = 1.0
    ibot_loss_weight: float = 1.0
    koleo_loss_weight: float = 0.1
    teacher_temp: float | Literal["auto"] = "auto"
    warmup_teacher_temp: float | Literal["auto"] = "auto"
    warmup_teacher_temp_epochs: int | Literal["auto"] = "auto"
    student_temp: float = 0.1
    center_momentum: float = 0.9

    # momentum
    momentum_start: float | Literal["auto"] = "auto"
    momentum_end: float = 1.0

    # weight decay
    weight_decay_start: float | Literal["auto"] = "auto"
    weight_decay_end: float | Literal["auto"] = "auto"

    def resolve_auto(
        self, scaling_info: ScalingInfo, optimizer_args: OptimizerArgs
    ) -> None:
        dataset_size = scaling_info.dataset_size

        if self.output_dim == "auto":
            # Default output dim of 65536 is too large for small datasets.
            self.output_dim = _scaling.get_bucket_value(
                input=dataset_size,
                buckets=[
                    (20_000, 1024),
                    (50_000, 2048),
                    (100_000, 4096),
                    (200_000, 16384),
                    (500_000, 32768),
                    (float("inf"), 65536),
                ],
            )

        if self.teacher_temp == "auto":
            # Default teacher temperature of 0.07 is too high for small datasets. Lower
            # temperature results in stronger sharpening which avoids collapse to uniform
            # distribution.
            self.teacher_temp = _scaling.interpolate(
                dataset_size,
                input_start=20_000,
                input_end=IMAGENET_SIZE,
                value_start=0.02,
                value_end=0.07,
                round_ndigits=2,
            )

        if self.warmup_teacher_temp == "auto":
            self.warmup_teacher_temp = min(
                self.teacher_temp,
                _scaling.interpolate(
                    input=self.teacher_temp,
                    input_start=0.02,
                    input_end=0.07,
                    value_start=0.02,
                    value_end=0.04,
                    round_ndigits=2,
                ),
            )

        if self.warmup_teacher_temp_epochs == "auto":
            # Default warmup teacher temperature epochs of 30 is too high when training
            # for only a few total epochs. Have the warmup period be 30% of all epochs,
            # but with a maximum of 30 epochs.
            self.warmup_teacher_temp_epochs = int(
                _scaling.interpolate(
                    scaling_info.epochs,
                    input_start=0,
                    input_end=100,
                    value_start=0,
                    value_end=30,
                )
            )

        if self.momentum_start == "auto":
            # Default momentum start of 0.996 is too high for small datasets. Lower momentum
            # results in slower updates of the teacher model. This is important because with
            # high momentum (fast changing teacher) and a small dataset, the initial
            # training epochs become unstable.
            self.momentum_start = _scaling.interpolate(
                dataset_size,
                input_start=20_000,
                input_end=IMAGENET_SIZE,
                value_start=0.99,
                value_end=0.996,
                round_ndigits=3,
            )

        if isinstance(optimizer_args, (AdamWArgs, SGDArgs)):
            weight_decay = optimizer_args.weight_decay
        else:
            raise ValueError(f"Unsupported optimizer_args type: {type(optimizer_args)}")
        if self.weight_decay_start == "auto":
            self.weight_decay_start = weight_decay
        if self.weight_decay_end == "auto":
            self.weight_decay_end = weight_decay


class DINOv2AdamWArgs(AdamWArgs):
    lr: float = 0.0005
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4


class DINOv2(Method):
    def __init__(
        self,
        method_args: DINOv2Args,
        optimizer_args: OptimizerArgs,
        embedding_model: EmbeddingModel,
        global_batch_size: int,
    ):
        super().__init__(
            method_args=method_args,
            optimizer_args=optimizer_args,
            embedding_model=embedding_model,
            global_batch_size=global_batch_size,
        )
        self.method_args = method_args

        # Calculate the number of crops
        self.n_local_crops = method_args.local_crops_number
        self.n_global_crops = method_args.global_crops_number
        self.n_global_crops_loss_terms = (self.n_global_crops - 1) * self.n_global_crops
        self.n_local_crops_loss_terms = max(self.n_local_crops * self.n_global_crops, 1)

        # Create teacher models
        self.teacher_embedding_model = embedding_model

        head_input_dim: int = self.teacher_embedding_model.embed_dim
        ibot_separate_head: bool = self.method_args.ibot_separate_head
        self.teacher_dino_head = DINOProjectionHead(
            input_dim=head_input_dim,
            hidden_dim=self.method_args.hidden_dim,
            bottleneck_dim=self.method_args.bottleneck_dim,
            output_dim=self.method_args.output_dim,
            batch_norm=self.method_args.batch_norm,
            norm_last_layer=self.method_args.norm_last_layer,
        )
        if ibot_separate_head:
            self.teacher_ibot_head = DINOProjectionHead(
                input_dim=head_input_dim,
                hidden_dim=self.method_args.hidden_dim,
                bottleneck_dim=self.method_args.bottleneck_dim,
                output_dim=self.method_args.output_dim,
                norm_last_layer=self.method_args.norm_last_layer,
            )
        else:
            self.teacher_ibot_head = self.teacher_dino_head

        # Create student models
        self.student_embedding_model = copy.deepcopy(self.teacher_embedding_model)
        self.student_dino_head = DINOProjectionHead(
            input_dim=head_input_dim,
            hidden_dim=self.method_args.hidden_dim,
            bottleneck_dim=self.method_args.bottleneck_dim,
            output_dim=self.method_args.output_dim,
            batch_norm=self.method_args.batch_norm,
            freeze_last_layer=self.method_args.student_freeze_last_layer_epochs,
            norm_last_layer=self.method_args.norm_last_layer,
        )
        if ibot_separate_head:
            self.student_ibot_head = DINOProjectionHead(
                input_dim=head_input_dim,
                hidden_dim=self.method_args.hidden_dim,
                bottleneck_dim=self.method_args.bottleneck_dim,
                output_dim=self.method_args.output_dim,
                freeze_last_layer=self.method_args.student_freeze_last_layer_epochs,
                norm_last_layer=self.method_args.norm_last_layer,
            )
        else:
            self.student_ibot_head = self.student_dino_head

        self.flatten = Flatten()

        # Losses
        self.dino_loss = DINOLoss()
        self.ibot_loss = iBOTPatchLoss()
        self.koleo_loss = KoLeoLoss()

        self.dino_loss_weight = self.method_args.dino_loss_weight
        self.ibot_loss_weight = self.method_args.ibot_loss_weight
        self.koleo_loss_weight = self.method_args.koleo_loss_weight

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        training_step_log: DINOv2TrainingStepResult = self.training_step_impl(
            batch, batch_idx
        )

        dino_global_loss = training_step_log.dino_global_loss
        dino_local_loss = training_step_log.dino_local_loss
        ibot_loss = training_step_log.ibot_loss
        koleo_loss = training_step_log.koleo_loss

        log_dict = {
            "dino_global_loss": dino_global_loss,
            "ibot_loss": ibot_loss,
            "koleo_loss": koleo_loss,
        }

        if dino_local_loss is not None:
            log_dict["dino_local_loss"] = dino_local_loss

        views = batch["views"]
        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(views[0]),
        )

        if self.global_step == 0:
            # Show example views of the images in the first batch only.
            self._log_example_views(train_batch=batch)

        loss = (
            self.dino_loss_weight * dino_global_loss
            + self.dino_loss_weight * dino_local_loss
            + self.ibot_loss_weight * ibot_loss
            + self.koleo_loss_weight * koleo_loss
        )

        return loss

    def training_step_impl(self, batch: Batch, batch_idx: int) -> TrainingStepResult:
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=no_auto(self.method_args.momentum_start),
            end_value=self.method_args.momentum_end,
        )
        update_momentum(
            self.student_embedding_model, self.teacher_embedding_model, m=momentum
        )
        update_momentum(self.student_dino_head, self.teacher_dino_head, m=momentum)

        views = batch["views"]
        global_views = torch.cat(views[:2])

        # Process global views through teacher and student networks
        x_teacher = self._forward_teacher(global_views)

        # Check if we have local views
        if (len_views := len(views)) > 2:
            local_views = torch.cat(views[2:])
            x_student = torch.cat(
                [
                    self._forward_student(global_views),
                    self._forward_student(local_views),
                ]
            )
        else:
            # Process only global views
            x_student = self._forward_student(global_views)

        # Compute the losses
        dino_global_loss = (
            self.dino_loss(
                teacher_out_softmaxed_centered_list=x_teacher.chunk(2),
                student_output_list=x_student.chunk(len_views),
            )
            * self.n_global_crops
            / (self.n_global_crops_loss_terms + self.n_local_crops_loss_terms)
        )
        if self.n_local_crops > 0:
            dino_local_loss = self.dino_loss(
                teacher_out_softmaxed_centered_list=x_teacher.chunk(2),
                student_output_list=x_student.chunk(len_views),
            ) / (self.n_global_crops_loss_terms + self.n_local_crops_loss_terms)

        ibot_loss = self.ibot_loss(
            teacher_patch_tokens=x_teacher.chunk(2),
            student_patch_tokens=x_student.chunk(len_views),
        )

        koleo_loss = sum(self.koleo_loss(token) for token in x_student.chunk(2))

        return TrainingStepResult(
            dino_global_loss=dino_global_loss,
            dino_local_loss=dino_local_loss if self.n_local_crops > 0 else None,
            ibot_loss=ibot_loss,
            koleo_loss=koleo_loss,
        )

    @torch.no_grad()
    def _forward_teacher(self, x: Tensor) -> Tensor:
        x = self.teacher_embedding_model(x)
        x = self.flatten(x)
        x = self.teacher_dino_head(x)
        return x

    def _forward_student(self, x: Tensor) -> Tensor:
        x = self.student_embedding_model(x)
        x = self.flatten(x)
        x = self.student_dino_head(x)
        return x

    @staticmethod
    def method_args_cls() -> type[DINOv2Args]:
        return DINOv2Args

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[OptimizerArgs]:
        classes: dict[OptimizerType | Literal["auto"], type[OptimizerArgs]] = {
            OptimizerType.ADAMW: DINOv2AdamWArgs,
        }
        return classes.get(optim_type, Method.optimizer_args_cls(optim_type=optim_type))

    def trainable_modules(self) -> TrainableModules:
        return TrainableModules(
            modules=[self.student_embedding_model, self.student_dino_head]
        )

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        self.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=3.0,
            gradient_clip_algorithm="norm",
        )
        self.student_dino_head.cancel_last_layer_gradients(self.current_epoch)

    def on_before_optimizer_step(self, optimizer: Optimizer, *args: Any) -> None:
        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.method_args.weight_decay_start,
            end_value=self.method_args.weight_decay_end,
        )
        optim.update_param_groups(
            optimizer, updates=[{"name": "params", "weight_decay": weight_decay}]
        )

    @staticmethod
    def transform_cls() -> type[MethodTransform]:
        return DINOv2Transform
