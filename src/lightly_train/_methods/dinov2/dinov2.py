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
from torch.optim.optimizer import Optimizer

from lightly_train import _scaling
from lightly_train._configs.validate import no_auto
from lightly_train._methods.dinov2.dinov2_loss import (
    DINOLoss,
    IBOTPatchLoss,
)  # we use the original DINOLoss and IBOTPatchLoss
from lightly_train._methods.dinov2.dinov2_transform import (
    DINOv2ViTLGTransform,
    DINOv2ViTSBTransform,
)
from lightly_train._methods.method import Method, TrainingStepResult
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._modules.teachers.dinov2.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.sgd_args import SGDArgs
from lightly_train._optim.trainable_modules import TrainableModules
from lightly_train._scaling import IMAGENET_SIZE, ScalingInfo
from lightly_train.types import Batch


@dataclass
class DINOv2TrainingStepResult:
    dino_global_loss: Tensor
    dino_local_loss: Tensor | None
    ibot_loss: Tensor
    koleo_loss: Tensor


class DINOv2Args(MethodArgs):
    """Args for DINOv2 method for ImageNet dataset."""

    # projection head
    ibot_separate_head: bool = False
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    output_dim: int = 65536
    batch_norm: bool = False
    student_freeze_last_layer_epochs: int = 1
    norm_last_layer: bool = False

    # loss
    dino_loss_weight: float = 1.0
    ibot_loss_weight: float = 1.0
    koleo_loss_weight: float = 0.1

    student_temp: float = 0.1
    center_momentum: float = 0.9

    teacher_temp: float = 0.04
    warmup_teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 30

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


class DINOv2ViTSArgs(DINOv2Args):
    pass


class DINOv2ViTBArgs(DINOv2Args):
    pass


class DINOv2ViTLArgs(DINOv2Args):
    # projection head
    ibot_separate_head: bool = True
    bottleneck_dim: int = 384
    bottleneck_dim_ibot: int = 256
    output_dim: int = 131072


class DINOv2ViTGArgs(DINOv2Args):
    # projection head
    ibot_separate_head: bool = True
    bottleneck_dim: int = 384
    bottleneck_dim_ibot: int = 256
    output_dim: int = 131072


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

        # Load configs based on the model architecture
        model_wrapper: DINOv2ViTModelWrapper = embedding_model.model_wrapper
        model: DinoVisionTransformer = model_wrapper._model
        embed_dim = model_wrapper._feature_dim
        self.method_args = self.method_args_cls(
            depth=model.n_blocks,
            num_heads=model.num_heads,
            embed_dim=embed_dim,
        )()

        # Calculate the number of crops
        self.n_global_crops = 2
        self.n_local_crops = 8  # transform_cls().transform_args_cls().transform_args.local_view.num_views
        self.n_global_crops_loss_terms = (self.n_global_crops - 1) * self.n_global_crops
        self.n_local_crops_loss_terms = max(self.n_local_crops * self.n_global_crops, 1)

        # Create teacher and student embedding models
        self.teacher_embedding_model_wrapper = model_wrapper
        self.student_embedding_model_wrapper = copy.deepcopy(
            self.teacher_embedding_model_wrapper
        )

        # Create teacher and student dino heads
        self.teacher_dino_head = DINOProjectionHead(
            input_dim=embed_dim,
            hidden_dim=self.method_args.hidden_dim,
            bottleneck_dim=self.method_args.bottleneck_dim,
            output_dim=self.method_args.output_dim,
            batch_norm=self.method_args.batch_norm,
            norm_last_layer=self.method_args.norm_last_layer,
        )
        self.student_dino_head = DINOProjectionHead(
            input_dim=embed_dim,
            hidden_dim=self.method_args.hidden_dim,
            bottleneck_dim=self.method_args.bottleneck_dim,
            output_dim=self.method_args.output_dim,
            batch_norm=self.method_args.batch_norm,
            freeze_last_layer=self.method_args.student_freeze_last_layer_epochs,
            norm_last_layer=self.method_args.norm_last_layer,
        )

        # Create teacher and student iBOT head
        self.ibot_separate_head: bool = self.method_args.ibot_separate_head
        if self.ibot_separate_head:
            self.teacher_ibot_head = DINOProjectionHead(
                input_dim=embed_dim,
                hidden_dim=self.method_args.hidden_dim,
                bottleneck_dim=self.method_args.bottleneck_dim_ibot,
                output_dim=self.method_args.output_dim,
                batch_norm=self.method_args.batch_norm,
                norm_last_layer=self.method_args.norm_last_layer,
            )
            self.student_ibot_head = DINOProjectionHead(
                input_dim=embed_dim,
                hidden_dim=self.method_args.hidden_dim,
                bottleneck_dim=self.method_args.bottleneck_dim_ibot,
                output_dim=self.method_args.output_dim,
                batch_norm=self.method_args.batch_norm,
                freeze_last_layer=self.method_args.student_freeze_last_layer_epochs,
                norm_last_layer=self.method_args.norm_last_layer,
            )
        else:
            self.teacher_ibot_head = self.teacher_dino_head
            self.student_ibot_head = self.student_dino_head

        # Losses
        self.dino_loss = DINOLoss(
            out_dim=self.method_args.output_dim,
            student_temp=self.method_args.student_temp,
            center_momentum=self.method_args.center_momentum,
        )
        self.ibot_loss = IBOTPatchLoss(
            patch_out_dim=self.method_args.output_dim,
            student_temp=self.method_args.student_temp,
            center_momentum=self.method_args.center_momentum,
        )
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
            self.student_embedding_model_wrapper,
            self.teacher_embedding_model_wrapper,
            m=momentum,
        )
        update_momentum(self.student_dino_head, self.teacher_dino_head, m=momentum)

        views = batch["views"]
        global_views = views[: self.n_global_crops]

        # Process global views through teacher and student networks
        cls_tokens_teacher, masked_patch_tokens_teacher = self._forward_teacher(
            global_views
        )

        # Check if we have local views
        if self.n_local_crops > 0:
            local_views = views[self.n_global_crops :]
            x_student = [
                self._forward_student(global_views),
                self._forward_student(local_views),
            ]
        else:
            # Process only global views
            x_student = self._forward_student(global_views)

        # Compute the DINO loss
        x_teacher_softmaxed = self.dino_loss.softmax_center_teacher(
            cls_tokens_teacher, teacher_temp=self.method_args.teacher_temp
        )
        self.dino_loss.update_center(cls_tokens_teacher)
        dino_global_loss = (
            self.dino_loss.forward(
                teacher_out_softmaxed_centered_list=x_teacher_softmaxed.chunk(
                    self.n_global_crops
                ),
                student_output_list=x_student.chunk(
                    self.n_global_crops + self.n_local_crops
                ),
            )
            * self.n_global_crops
            / (self.n_global_crops_loss_terms + self.n_local_crops_loss_terms)
        )
        if self.n_local_crops > 0:
            dino_local_loss = self.dino_loss.forward(
                teacher_out_softmaxed_centered_list=x_teacher_softmaxed.chunk(
                    self.n_global_crops
                ),
                student_output_list=x_student.chunk(
                    self.n_global_crops + self.n_local_crops
                ),
            ) / (self.n_global_crops_loss_terms + self.n_local_crops_loss_terms)

        # Compute the iBOT loss
        x_teacher_masked_softmaxed = self.ibot_loss.softmax_center_teacher(
            masked_patch_tokens_teacher.unsqueeze(0),
            teacher_temp=self.method_args.teacher_temp,
        )
        self.ibot_loss.update_center(masked_patch_tokens_teacher.unsqueeze(0))
        ibot_loss = self.ibot_loss.forward_masked(
            teacher_patch_tokens=x_teacher_masked_softmaxed.chunk(self.n_global_crops),
            student_patch_tokens=x_student.chunk(
                self.n_global_crops + self.n_local_crops
            ),
        )

        koleo_loss = sum(
            self.koleo_loss(token) for token in x_student.chunk(self.n_global_crops)
        )  # only use global views

        return TrainingStepResult(
            dino_global_loss=dino_global_loss,
            dino_local_loss=dino_local_loss if self.n_local_crops > 0 else None,
            ibot_loss=ibot_loss,
            koleo_loss=koleo_loss,
        )

    @torch.no_grad()
    def _forward_teacher(self, x: Tensor) -> Tensor:
        results = self.teacher_embedding_model_wrapper.forward_features(x)
        patch_tokens = results["features"]  # [B, C, H, W]
        cls_tokens = results["cls_token"]  # [B, 1, C]

        # process the cls tokens
        cls_tokens = cls_tokens.chunk(self.n_global_crops)
        # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
        cls_tokens = torch.cat((cls_tokens[1], cls_tokens[0]))
        n_cls_tokens = cls_tokens.shape[0]

        # process the patch tokens
        patch_tokens = patch_tokens.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        n_channels = patch_tokens.shape[-1]

        # pas the cls and patch tokens through the head
        if not self.ibot_separate_head:
            upperbound = patch_tokens.shape[1]  # TODO
            mask_indices_list = torch.arange()  # TODO
            n_masked_patches = torch.full(
                (1,), fill_value=mask_indices_list.shape[0], dtype=torch.long
            )  # TODO

            buffer_tensor_teacher = patch_tokens.new_zeros(
                n_cls_tokens + upperbound, n_channels
            )
            buffer_tensor_teacher[:n_cls_tokens].copy_(cls_tokens)
            torch.index_select(
                patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor_teacher[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ],
            )

            tokens_after_head = self.teacher_dino_head(buffer_tensor_teacher)
            cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
            masked_patch_tokens_after_head = tokens_after_head[
                n_cls_tokens : n_cls_tokens + n_masked_patches
            ]
        else:
            buffer_tensor_teacher = patch_tokens.new_zeros(upperbound, n_channels)
            torch.index_select(
                patch_tokens.flatten(0, 1),
                dim=0,
                index=mask_indices_list,
                out=buffer_tensor_teacher[:n_masked_patches],
            )

            cls_tokens_after_head = self.teacher_dino_head(cls_tokens)
            masked_patch_tokens_after_head = self.teacher_ibot_head(
                buffer_tensor_teacher
            )[:n_masked_patches]

        return cls_tokens_after_head, masked_patch_tokens_after_head

    def _forward_student(self, x: Tensor) -> Tensor:
        results = self.student_embedding_model_wrapper.forward_features(
            x
        )  # [B, C, H, W]
        patch_tokens = results["features"].flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        cls_tokens = results["cls_token"]  # [B, 1, C]

        patch_tokens_after_head = self.student_dino_head(patch_tokens)
        cls_tokens_after_head = self.student_ibot_head(cls_tokens)

        return patch_tokens_after_head, cls_tokens_after_head

    @staticmethod
    def method_args_cls(
        depth, num_heads, embed_dim
    ) -> type[DINOv2ViTSArgs | DINOv2ViTBArgs | DINOv2ViTLArgs | DINOv2ViTGArgs]:
        if depth == 40 and num_heads == 24 and embed_dim == 1536:
            method_args = DINOv2ViTGArgs  # giant
        elif depth == 24 and num_heads == 16 and embed_dim == 1024:
            method_args = DINOv2ViTLArgs  # large
        elif depth == 12 and num_heads == 12 and embed_dim == 768:
            method_args = DINOv2ViTBArgs  # base
        elif depth == 12 and num_heads == 6 and embed_dim == 384:
            method_args = DINOv2ViTSArgs  # small
        else:
            raise ValueError("Unsupported model configs.")

        return method_args

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
            modules=[self.student_embedding_model_wrapper, self.student_dino_head]
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
    def transform_cls(
        depth, num_heads, embed_dim
    ) -> type[DINOv2ViTSBTransform | DINOv2ViTLGTransform]:
        if depth == 40 and num_heads == 24 and embed_dim == 1536:
            method_args = DINOv2ViTLGTransform  # giant
        elif depth == 24 and num_heads == 16 and embed_dim == 1024:
            method_args = DINOv2ViTLGTransform  # large
        elif depth == 12 and num_heads == 12 and embed_dim == 768:
            method_args = DINOv2ViTSBTransform  # base
        elif depth == 12 and num_heads == 6 and embed_dim == 384:
            method_args = DINOv2ViTSBTransform  # small
        else:
            raise ValueError("Unsupported model configs.")

        return method_args
