#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Flatten, Linear, Module, init
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim.optimizer import Optimizer

from lightly_train import _scaling
from lightly_train._configs.validate import no_auto
from lightly_train._methods.distillationv3.distillationv3_loss import DistillationV3Loss
from lightly_train._methods.distillationv3.distillationv3_transform import (
    DistillationV3Transform,
)
from lightly_train._methods.method import Method, TrainingStepResult
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper
from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._optim.lars_args import LARSArgs
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.trainable_modules import TrainableModules
from lightly_train._scaling import ScalingInfo
from lightly_train._transforms.transform import (
    MethodTransform,
)
from lightly_train.types import Batch

logger = logging.getLogger(__name__)


def get_teacher(
    teacher_name: str,
    num_input_channels: int,
    teacher_weights: str | Path | None = None,
) -> Module:
    wrapped_model = package_helpers.get_wrapped_model(
        model=teacher_name,
        num_input_channels=num_input_channels,
    )
    assert isinstance(
        wrapped_model,
        (DINOv2ViTModelWrapper, DINOv3ViTModelWrapper, DINOv3VConvNeXtModelWrapper),
    )
    wrapped_model.make_teacher()
    teacher_embedding_model = wrapped_model.get_model()

    # If a path to the teacher weights is provided, load them.
    if teacher_weights is not None:
        if not Path(teacher_weights).exists():
            raise FileNotFoundError(
                f"Teacher weights file {teacher_weights} does not exist."
            )
        device = next(teacher_embedding_model.parameters()).device
        state_dict = torch.load(teacher_weights, weights_only=True, map_location=device)
        teacher_embedding_model.load_state_dict(state_dict)
        logger.info(f"Loaded teacher weights from {teacher_weights}.")

    teacher_embedding_model.eval()

    # Freeze the teacher parameters.
    for p in teacher_embedding_model.parameters():
        p.requires_grad_(False)

    return teacher_embedding_model


class DistillationV3Args(MethodArgs):
    """Args for DistillationV3 method for dataset."""

    # Number of teacher blocks from the teacher model to use.
    n_teacher_blocks: int = 1

    # Default number of teacher embeddings to store in the queue to serve as pseudo classification weights.
    queue_size: int | Literal["auto"] = "auto"

    # Default temperature parameter to regulate the sharpness of the distributions in the loss.
    temperature_global: float = 0.07
    temperature_local: float = 0.07

    # Default teacher.
    teacher: str = "dinov3/vitb16"

    # Optional teacher weight path.
    teacher_weights: str | Path | None = None

    # Deprecated. Does not have any effect.
    teacher_url: str | None = None

    # Scaling method for the learning rate.
    lr_scale_method: Literal["linear", "sqrt"] = "sqrt"
    reference_batch_size: int = 1536

    # Loss weighting.
    local_loss_weight: float = 1.0

    def resolve_auto(
        self,
        scaling_info: ScalingInfo,
        optimizer_args: OptimizerArgs,
        wrapped_model: ModelWrapper,
    ) -> None:
        if self.queue_size == "auto":
            # Reduce the queue size for smaller datasets.
            self.queue_size = _scaling.get_bucket_value(
                input=scaling_info.dataset_size,
                buckets=[
                    # (dataset_size, queue_size)
                    # Memory bank size is roughly 50% of the minimal dataset size and
                    # 25% of the maximal dataset size for the given bucket. For example,
                    # a bucket with 100-250 images has a queue size of 64.
                    (50, 16),
                    (100, 32),
                    (250, 64),
                    (500, 128),
                    (1_000, 256),
                    (2_000, 512),
                    (4_000, 1024),
                    (10_000, 2048),
                    (20_000, 4096),
                    (float("inf"), 8192),
                ],
            )
        elif self.queue_size >= scaling_info.dataset_size:
            raise ValueError(
                f"The specified queue size ({self.queue_size}) cannot be larger than the dataset size ({scaling_info.dataset_size})."
            )


class DistillationV3LARSArgs(LARSArgs):
    lr: float = 1.8  # 1.8 = 0.3 * 1536 / 256
    momentum: float = 0.9
    dampening: float = 0
    weight_decay: float = 1e-6
    nesterov: bool = False
    trust_coefficient: float = 0.001
    eps: float = 1e-8


class DistillationV3AdamWArgs(AdamWArgs):
    lr: float = 0.0005
    weight_decay: float = 0.04


class DistillationV3(Method):
    def __init__(
        self,
        method_args: DistillationV3Args,
        optimizer_args: OptimizerArgs,
        embedding_model: EmbeddingModel,
        global_batch_size: int,
        num_input_channels: int,
    ):
        super().__init__(
            method_args=method_args,
            optimizer_args=optimizer_args,
            embedding_model=embedding_model,
            global_batch_size=global_batch_size,
            num_input_channels=num_input_channels,
        )
        # Get the teacher model.
        self.teacher_embedding_model = get_teacher(
            method_args.teacher,
            num_input_channels=num_input_channels,
            teacher_weights=method_args.teacher_weights,
        )

        # Store the student model.
        self.student_embedding_model = embedding_model
        self.flatten = Flatten(start_dim=1)

        # Instantiate linear projection heads that performs the mapping from the student embedding space to the teacher embedding space.
        self.teacher_embedding_dim: int = (
            method_args.n_teacher_blocks * self.teacher_embedding_model.embed_dim  # type: ignore
        )
        self.student_projection_head_global = Linear(
            embedding_model.embed_dim,
            self.teacher_embedding_dim,  # type: ignore
        )
        self.student_projection_head_local = Linear(
            embedding_model.embed_dim,
            self.teacher_embedding_dim,  # type: ignore
        )

        # Initialize the weights of the linear projection heads with a truncated normal.
        init.trunc_normal_(self.student_projection_head_global.weight, std=0.02)
        init.trunc_normal_(self.student_projection_head_local.weight, std=0.02)

        # Instantiate the criterion.
        self.criterion = DistillationV3Loss(
            temperature_global=method_args.temperature_global,
            temperature_local=method_args.temperature_local,
        )
        self.method_args = method_args

        # Initialize a buffer to store the teacher representations from previous batches. The queue is independent to each gpu.
        self.teacher_queue: Tensor
        self.register_buffer(
            "teacher_queue",
            torch.zeros(
                [
                    no_auto(method_args.queue_size),
                    self.teacher_embedding_dim,  # type: ignore
                ]
            ),
        )

    def training_step_impl(self, batch: Batch, batch_idx: int) -> TrainingStepResult:
        # Get the images. In distillation, we only use one view.
        views = batch["views"][0]

        # Mixup the data.
        views = self._mixup_data(views)

        # Get the [B, D] teacher features.
        x_teacher_global, x_teacher_local, (teacher_features_h, teacher_features_w) = (
            self._forward_teacher(views)
        )

        # Get the [B, D] student features.
        x_student_global, x_student_local = self._forward_student(
            views, teacher_features_h, teacher_features_w
        )

        # Update the queue
        self._update_queue(x_teacher=x_teacher_global)

        # Compute the loss.
        global_loss, local_loss = self.criterion(
            teacher_features_global=x_teacher_global,
            teacher_features_local=x_teacher_local,
            student_features_global=x_student_global,
            student_features_local=x_student_local,
            queue=self.teacher_queue,
        )

        # Combine the losses.
        loss = global_loss + self.method_args.local_loss_weight * local_loss

        return TrainingStepResult(
            loss=loss,
            log_dict={
                "train_loss/local_loss": local_loss.item(),
                "train_loss/global_loss": global_loss.item(),
            },
        )

    @torch.no_grad()
    def _update_queue(self, x_teacher: Tensor) -> None:
        # Get the batch and queue size.
        B = x_teacher.size(0)
        queue_size = self.teacher_queue.size(0)

        # Handle cases where the queue size smaller than the batch size.
        if B >= queue_size:
            # When the queue is smaller than the batch, the queue is filled with a subset of the batch.
            self.teacher_queue = x_teacher[:queue_size].clone()
        else:
            # Shift the queue by B to the right.
            self.teacher_queue[B:] = self.teacher_queue[:-B].clone()

            # Replace the first B elements in the queue.
            self.teacher_queue[:B] = x_teacher

    @torch.no_grad()
    def _forward_teacher(self, x: Tensor) -> tuple[Tensor, Tensor, tuple[int, int]]:
        """Forward the images through the teacher model and return them in the
        (B, H * W, n_teacher_blocks * D) format.
        """
        # List with n_teacher_blocks tensors with shape (B, D, H, W)
        x_list = list(
            self.teacher_embedding_model.get_intermediate_layers(  # type: ignore[operator]
                x,
                n=self.method_args.n_teacher_blocks,
                reshape=True,
                return_class_token=True,
            )
        )

        # Make sure all feature maps have the same spatial size as the last layer.
        # For ViTs this is always the case. But ConvNeXts return feature maps of
        # different sizes. E.g. 14x14 and 7x7.
        teacher_features_h, teacher_features_w = x_list[-1][0].shape[-2:]
        x_list_global = []
        x_list_local = []
        for x_local, x_global in x_list:
            h, w = x_local.shape[-2:]
            if (h != teacher_features_h) or (w != teacher_features_w):
                x_local = F.interpolate(
                    x_local,
                    size=(teacher_features_h, teacher_features_w),
                    mode="bilinear",
                    align_corners=False,
                )
            x_list_global.append(x_global)
            x_list_local.append(x_local)

        # Concat along the feature dimension.
        # (B, n_teacher_blocks * D, H, W)
        x_local = torch.cat(x_list_local, dim=1)
        # (B, n_teacher_blocks * D)
        x_global = torch.cat(x_list_global, dim=1)

        # (B, n_teacher_blocks * D, H, W) -> (B, H * W, n_teacher_blocks * D)
        x_local = x_local.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        # L2-normalize the features.
        x_local = F.normalize(x_local, dim=-1, p=2)
        x_global = F.normalize(x_global, dim=-1, p=2)
        return (x_global, x_local, (teacher_features_h, teacher_features_w))

    def _forward_student(
        self, x: Tensor, teacher_features_h: int, teacher_features_w: int
    ) -> tuple[Tensor, Tensor]:
        """Forward the images through the student model and return them in the
        (B, H*W, D) format where D = teacher_embedding_dim.
        """
        # Forward the images through the student model.
        # x = self.student_embedding_model(x)
        x_global_local = self.student_embedding_model.wrapped_model.forward_features(x)
        x_global = self.student_embedding_model.wrapped_model.forward_pool(
            x_global_local
        )["pooled_features"]
        x_local = x_global_local["features"]

        # Discard empty spatial dimensions: (B, C, 1, 1) -> (B, C).
        x_global = self.flatten(x_global)

        # (B, C, H, W) -> (B, H * W, C)
        x_local = x_local.permute(0, 2, 3, 1)
        # x_local = x_local.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        # Forward the student features through the projection head to match the dimension of the teacher: (B, C) -> (B, D).
        x_global = self.student_projection_head_global(x_global)
        x_local = self.student_projection_head_global(x_local)

        # Resize the student spatial features to have the same resolution
        # as the teacher spatial features.
        x_local = x_local.permute(0, 3, 1, 2)  # (B, H, W, D) -> (B, D, H, W)
        x_local = F.interpolate(
            x_local,
            size=(teacher_features_h, teacher_features_w),
            mode="bilinear",
            align_corners=False,
        )

        # Flatten the spatial dimensions to match the teacher features:
        # (B, D, H, W) -> (B, H * W, D).
        x_local = x_local.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        # L2-normalize the features.
        x_global = F.normalize(x_global, dim=-1, p=2)
        x_local = F.normalize(x_local, dim=-1, p=2)
        return x_global, x_local

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
    def method_args_cls() -> type[DistillationV3Args]:
        return DistillationV3Args

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[OptimizerArgs]:
        classes: dict[OptimizerType | Literal["auto"], type[OptimizerArgs]] = {
            "auto": DistillationV3LARSArgs,
            OptimizerType.LARS: DistillationV3LARSArgs,
            OptimizerType.ADAMW: DistillationV3AdamWArgs,
        }
        return classes.get(optim_type, Method.optimizer_args_cls(optim_type=optim_type))

    def trainable_modules(self) -> TrainableModules:
        return TrainableModules(
            modules=[self.student_embedding_model, self.student_projection_head_global]
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
        return DistillationV3Transform

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

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        """Ensure only teacher-related keys are missing from the statedict."""
        # Load with strict=False to capture missing/unexpected keys.
        incompatible_keys = cast(
            _IncompatibleKeys, super().load_state_dict(state_dict, strict=False)
        )

        # Filter out teacher-related keys from the list of missing keys.
        missing_keys = [
            k
            for k in incompatible_keys.missing_keys
            if not k.startswith("teacher_embedding_model.")
        ]

        # No key should be missing besides the ones related to the teacher model.
        if strict and (missing_keys or incompatible_keys.unexpected_keys):
            raise RuntimeError(
                f"Unexpected keys in state_dict: {incompatible_keys.unexpected_keys}\n"
                f"Missing keys in state_dict: {missing_keys}"
            )
        return incompatible_keys
