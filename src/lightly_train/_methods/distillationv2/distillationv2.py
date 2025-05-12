#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, init
from torch.optim.optimizer import Optimizer

from lightly_train._methods.distillationv2.distillationv2_loss import DistillationV2Loss
from lightly_train._methods.distillationv2.distillationv2_transform import (
    DistillationV2Transform,
)
from lightly_train._methods.method import Method, TrainingStepResult
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._modules.teachers.build_teacher import get_teacher
from lightly_train._optim.lars_args import LARSArgs
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.trainable_modules import TrainableModules
from lightly_train._transforms.transform import (
    MethodTransform,
)
from lightly_train.types import Batch

logger = logging.getLogger(__name__)


class DistillationV2Args(MethodArgs):
    """Args for DistillationV2 method for dataset."""

    # Number of teacher blocks from the teacher model to use.
    n_teacher_blocks: int = 2

    # Default teacher
    teacher: str = "dinov2_vitb14"


class DistillationV2LARSArgs(LARSArgs):
    lr: float = 1.5
    momentum: float = 0.9
    dampening: float = 0
    weight_decay: float = 1e-6
    nesterov: bool = False
    trust_coefficient: float = 0.001
    eps: float = 1e-8


class DistillationV2(Method):
    def __init__(
        self,
        method_args: DistillationV2Args,
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
        # Get the teacher model.
        self.teacher_embedding_model = get_teacher(teacher_name=method_args.teacher)
        self.teacher_embedding_dim = (
            method_args.n_teacher_blocks * self.teacher_embedding_model.embed_dim
        )

        # Store the student model.
        self.student_embedding_model = embedding_model

        # Instantiate a linear projection head that performs the mapping
        # from the student embedding space to the teacher embedding space.
        self.student_projection_head = Linear(
            embedding_model.embed_dim, self.teacher_embedding_dim
        )

        # Initialize the weights of the linear projection head with a
        # truncated normal.
        init.trunc_normal_(self.student_projection_head.weight, std=0.02)

        # Instantiate the criterion.
        self.criterion = DistillationV2Loss()
        self.method_args = method_args

    def training_step_impl(self, batch: Batch, batch_idx: int) -> TrainingStepResult:
        # Get the images. In distillation, we only use one view.
        views = batch["views"][0]

        # Mixup the data.
        views = self._mixup_data(views)

        # Get the [B, D] teacher features.
        x_teacher = self._forward_teacher(views)

        # Get the [B, D] student features.
        x_student = self._forward_student(views)

        # Compute the loss.
        loss = self.criterion(
            teacher_features=x_teacher,
            student_features=x_student,
        )

        return TrainingStepResult(loss=loss)

    @torch.no_grad()
    def _forward_teacher(self, x: Tensor) -> Tensor:
        # Forward the images through the teacher model.
        x_list = self.teacher_embedding_model.get_intermediate_layers(
            x, n=self.method_args.n_teacher_blocks
        )
        x = torch.cat(x_list, dim=-1)
        return x

    def _forward_student(self, x: Tensor) -> Tensor:
        # Store the image size.
        b, _, image_h, image_w = x.shape

        # Infer the spatial size of the teacher features.
        teacher_features_h = image_h // self.teacher_embedding_model.patch_size
        teacher_features_w = image_w // self.teacher_embedding_model.patch_size

        # Forward the images through the student model.
        x = self.student_embedding_model(x, pool=False)

        # The projection head expects tensors with channel last format.
        x = x.permute(0, 2, 3, 1)

        # Forward the student features through the projection head to
        # match the dimension of the teacher: (B, H, W, C) -> (B, H, W, D).
        x = self.student_projection_head(x)

        # Resize the student spatial features to have the same resolution
        # as the teacher spatial features.
        x = x.permute(0, 3, 1, 2)  # (B, H, W, D) -> (B, D, H, W)
        x = F.interpolate(
            x,
            size=(teacher_features_h, teacher_features_w),
            mode="bilinear",
            align_corners=False,
        )

        # Flatten the spatial dimensions to match the teacher features:
        # (B, D, H, W) -> (B, H * W, D).
        x = x.permute(0, 2, 3, 1).view(b, -1, self.teacher_embedding_dim)

        return x

    @staticmethod
    def _mixup_data(x: Tensor) -> Tensor:
        # Sample lambda from a uniform distribution U(0, 1).
        lambda_ = torch.empty(1).uniform_(0.0, 1.0).item()

        # Obtain a random permutation of the image indices.
        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        # Perform a convex combination of the images and shuffled images.
        mixed_x = lambda_ * x + (1.0 - lambda_) * x[index, :]
        return mixed_x

    @staticmethod
    def method_args_cls() -> type[DistillationV2Args]:
        return DistillationV2Args

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[OptimizerArgs]:
        classes: dict[OptimizerType | Literal["auto"], type[OptimizerArgs]] = {
            "auto": DistillationV2LARSArgs,
            OptimizerType.LARS: DistillationV2LARSArgs,
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
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )

    @staticmethod
    def transform_cls() -> type[MethodTransform]:
        return DistillationV2Transform

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Remove the teacher model from the checkpoint before saving."""
        # Iterate over the state dict and filter out the teacher model.
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("teacher_embedding_model.")
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Ensure the teacher statedict is in the checkpoint before resuming."""
        # Add the teacher model to the checkpoint.
        checkpoint["state_dict"].update(
            {
                f"teacher_embedding_model.{k}": v
                for k, v in self.teacher_embedding_model.state_dict().items()
            }
        )

    @torch.no_grad()
    def _broadcast_teacher_weights(self) -> None:
        """Broadcast the teacher weights from rank 0 to all ranks.
        This is necessary to ensure that all ranks have the same teacher weights.
        Only global rank 0 downloads the teacher weights.
        """
        for param in self.teacher_embedding_model.state_dict().values():
            if isinstance(param, torch.Tensor):
                torch.distributed.broadcast(param, src=0)

    def on_fit_start(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            logger.info("Broadcasting teacher weights from rank 0 to all ranks.")
            self._broadcast_teacher_weights()
