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
from torch.nn import GELU, LayerNorm, Linear, Module, Sequential, init
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim.optimizer import Optimizer

from lightly_train._methods.distillationv2.distillationv2_loss import DistillationV2Loss
from lightly_train._methods.distillationv2.distillationv2_transform import (
    DistillationV2Transform,
)
from lightly_train._methods.method import Method, TrainingStepResult
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models import package_helpers
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov3.dinov3_convnext import DINOv3VConvNeXtModelWrapper
from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.lars_args import LARSArgs
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.trainable_modules import TrainableModules
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
        logger.debug(f"Loaded teacher weights from {teacher_weights}.")

    teacher_embedding_model.eval()

    # Freeze the teacher parameters.
    for p in teacher_embedding_model.parameters():
        p.requires_grad_(False)

    return teacher_embedding_model


class DistillationV2Args(MethodArgs):
    """Args for DistillationV2 method for dataset."""

    # Number of teacher blocks from the teacher model to use.
    n_teacher_blocks: int = 2

    # Default teacher
    teacher: str = "dinov2/vitb14-noreg"

    # Optional teacher weight path.
    teacher_weights: str | Path | None = None

    # Deprecated. Does not have any effect.
    teacher_url: str | None = None

    # Number of projection layers in the projection head.
    n_projection_layers: int = 1

    # Hidden dimension of the projection head.
    projection_hidden_dim: int = 2048

    # Scaling method for the learning rate.
    lr_scale_method: Literal["linear", "sqrt"] = "sqrt"
    reference_batch_size: int = 1536


class DistillationV2LARSArgs(LARSArgs):
    lr: float = 9.0  # 9.0 = 1.5 * 1536 / 256
    momentum: float = 0.9
    dampening: float = 0
    weight_decay: float = 1e-6
    nesterov: bool = False
    trust_coefficient: float = 0.001
    eps: float = 1e-8


class DistillationV2Head(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_layers: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        n_layers = max(n_layers, 1)

        if n_layers == 1:
            self.mlp: Module = Linear(in_dim, out_dim)
        else:
            layers: list[Module] = [Linear(in_dim, hidden_dim)]
            layers.append(LayerNorm(hidden_dim))
            layers.append(GELU())
            for _ in range(n_layers - 2):
                layers.append(Linear(hidden_dim, hidden_dim))
                layers.append(LayerNorm(hidden_dim))
                layers.append(GELU())
            layers.append(Linear(hidden_dim, out_dim))
            self.mlp = Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, m: Module) -> None:
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        # Convert to channel last format.
        x = x.permute(0, 2, 3, 1)
        x = self.mlp(x)
        return x


class DistillationV2(Method):
    def __init__(
        self,
        method_args: DistillationV2Args,
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
            teacher_name=method_args.teacher,
            num_input_channels=num_input_channels,
            teacher_weights=method_args.teacher_weights,
        )
        self.teacher_embedding_dim: int = (
            method_args.n_teacher_blocks * self.teacher_embedding_model.embed_dim  # type: ignore
        )

        # Store the student model.
        self.student_embedding_model = embedding_model

        # Instantiate a projection head that performs the mapping
        # from the student embedding space to the teacher embedding space.
        self.student_projection_head = DistillationV2Head(
            in_dim=embedding_model.embed_dim,
            out_dim=self.teacher_embedding_dim,
            n_layers=method_args.n_projection_layers,
            hidden_dim=method_args.projection_hidden_dim,
        )

        # Instantiate the criterion.
        self.criterion = DistillationV2Loss()
        self.method_args = method_args

    def training_step_impl(self, batch: Batch, batch_idx: int) -> TrainingStepResult:
        # Get the images. In distillation, we only use one view.
        views = batch["views"][0]

        # Mixup the data.
        views = self._mixup_data(views)

        # Get the (B, H*W, D) teacher features.
        x_teacher, (teacher_features_h, teacher_features_w) = self._forward_teacher(
            views
        )

        # Get the (B, H*W, D) student features.
        x_student = self._forward_student(
            views,
            teacher_features_h=teacher_features_h,
            teacher_features_w=teacher_features_w,
        )

        # Compute the loss.
        loss = self.criterion(
            teacher_features=x_teacher,
            student_features=x_student,
        )

        return TrainingStepResult(loss=loss)

    @torch.no_grad()
    def _forward_teacher(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Forward the images through the teacher model and return them in the
        (B, H * W, n_teacher_blocks * D) format.
        """
        # List with n_teacher_blocks tensors with shape (B, D, H, W)
        x_list = list(
            self.teacher_embedding_model.get_intermediate_layers(  # type: ignore[operator]
                x, n=self.method_args.n_teacher_blocks, reshape=True
            )
        )

        # Make sure all feature maps have the same spatial size as the last layer.
        # For ViTs this is always the case. But ConvNeXts return feature maps of
        # different sizes. E.g. 14x14 and 7x7.
        teacher_features_h, teacher_features_w = x_list[-1].shape[-2:]
        for i in range(len(x_list)):
            x = x_list[i]
            h, w = x.shape[-2:]
            if (h != teacher_features_h) or (w != teacher_features_w):
                x = F.interpolate(
                    x,
                    size=(teacher_features_h, teacher_features_w),
                    mode="bilinear",
                    align_corners=False,
                )
            x_list[i] = x

        # Concat along the feature dimension.
        # (B, n_teacher_blocks * D, H, W)
        x = torch.cat(x_list, dim=1)
        # (B, n_teacher_blocks * D, H, W) -> (B, H * W, n_teacher_blocks * D)
        x = x.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        return (x, (teacher_features_h, teacher_features_w))

    def _forward_student(
        self, x: Tensor, teacher_features_h: int, teacher_features_w: int
    ) -> Tensor:
        """Forward the images through the student model and return them in the
        (B, H*W, D) format where D = teacher_embedding_dim.
        """
        # Forward the images through the student model.
        x = self.student_embedding_model(x, pool=False)

        # Forward the student features through the projection head to
        # match the dimension of the teacher: (B, C, H, W) -> (B, H, W, D).
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
        x = x.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

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
