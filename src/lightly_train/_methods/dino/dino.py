#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
from typing import Any, Literal

import torch
from lightly.loss import DINOLoss
from lightly.models.modules.heads import DINOProjectionHead
from lightly.models.utils import update_momentum
from lightly.utils import optim
from lightly.utils.scheduler import cosine_schedule
from torch import Tensor
from torch.nn import Flatten
from torch.optim.optimizer import Optimizer

from lightly_train import _scaling
from lightly_train._configs.validate import no_auto
from lightly_train._methods.dino.dino_transform import (
    DINOTransform,
)
from lightly_train._methods.method import Method, TrainingStepResult
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._models.model_wrapper import ModelWrapper
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


class DINOArgs(MethodArgs):
    """Args for DINO method for ImageNet dataset."""

    # projection head
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    output_dim: int | Literal["auto"] = "auto"
    student_freeze_last_layer_epochs: int = 1
    batch_norm: bool = False
    norm_last_layer: bool = True
    # loss
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
        self,
        scaling_info: ScalingInfo,
        optimizer_args: OptimizerArgs,
        wrapped_model: ModelWrapper,
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


class DINOAdamWArgs(AdamWArgs):
    lr: float = 0.0005
    weight_decay: float = 0.04


class DINOSGDArgs(SGDArgs):
    lr: float = 0.03
    weight_decay: float = 0.0001


class DINO(Method):
    def __init__(
        self,
        method_args: DINOArgs,
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
        self.teacher_embedding_model = embedding_model
        self.teacher_projection_head = DINOProjectionHead(
            input_dim=self.teacher_embedding_model.embed_dim,
            hidden_dim=method_args.hidden_dim,
            bottleneck_dim=method_args.bottleneck_dim,
            output_dim=no_auto(method_args.output_dim),
            batch_norm=method_args.batch_norm,
            freeze_last_layer=0,
            norm_last_layer=method_args.norm_last_layer,
        )
        self.student_embedding_model = copy.deepcopy(self.teacher_embedding_model)
        self.student_projection_head = DINOProjectionHead(
            input_dim=self.student_embedding_model.embed_dim,
            hidden_dim=method_args.hidden_dim,
            bottleneck_dim=method_args.bottleneck_dim,
            output_dim=no_auto(method_args.output_dim),
            batch_norm=method_args.batch_norm,
            freeze_last_layer=method_args.student_freeze_last_layer_epochs,
            norm_last_layer=method_args.norm_last_layer,
        )
        self.flatten = Flatten(start_dim=1)
        self.criterion = DINOLoss(
            output_dim=no_auto(method_args.output_dim),
            teacher_temp=no_auto(method_args.teacher_temp),
            warmup_teacher_temp=no_auto(method_args.warmup_teacher_temp),
            warmup_teacher_temp_epochs=no_auto(method_args.warmup_teacher_temp_epochs),
            student_temp=method_args.student_temp,
            center_momentum=method_args.center_momentum,
        )

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
        update_momentum(
            self.student_projection_head, self.teacher_projection_head, m=momentum
        )

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

        loss = self.criterion(
            teacher_out=x_teacher.chunk(2),
            student_out=x_student.chunk(len_views),
            epoch=self.current_epoch,
        )

        return TrainingStepResult(loss=loss)

    @torch.no_grad()
    def _forward_teacher(self, x: Tensor) -> Tensor:
        x = self.teacher_embedding_model(x)
        x = self.flatten(x)
        x = self.teacher_projection_head(x)
        return x

    def _forward_student(self, x: Tensor) -> Tensor:
        x = self.student_embedding_model(x)
        x = self.flatten(x)
        x = self.student_projection_head(x)
        return x

    @staticmethod
    def method_args_cls() -> type[DINOArgs]:
        return DINOArgs

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[OptimizerArgs]:
        classes: dict[OptimizerType | Literal["auto"], type[OptimizerArgs]] = {
            "auto": DINOSGDArgs,
            OptimizerType.ADAMW: DINOAdamWArgs,
            OptimizerType.SGD: DINOSGDArgs,
        }
        return classes.get(optim_type, Method.optimizer_args_cls(optim_type=optim_type))

    def trainable_modules(self) -> TrainableModules:
        return TrainableModules(
            modules=[self.student_embedding_model, self.student_projection_head]
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
        self.student_projection_head.cancel_last_layer_gradients(self.current_epoch)

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
        return DINOTransform
