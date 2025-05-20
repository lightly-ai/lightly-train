#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from lightly.loss import (
    KoLeoLoss,
)  # we use LightlySSL's KoLeoLoss for better numerical stability
from lightly.models.modules.heads import DINOProjectionHead
from lightly.models.utils import update_momentum
from lightly.utils.optim import update_param_groups
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from lightly_train._configs.validate import no_auto
from lightly_train._methods.dinov2.dinov2_loss import (
    DINOLoss,
    IBOTPatchLoss,
)  # we use the original DINOLoss and IBOTPatchLoss
from lightly_train._methods.dinov2.dinov2_optim import (
    DINOv2AdamWViTLGArgs,
    DINOv2AdamWViTSBArgs,
    get_optimizer_with_decay,
)
from lightly_train._methods.dinov2.dinov2_transform import (
    DINOv2ViTSBTransform,
)
from lightly_train._methods.dinov2.utils import (
    MaskingGenerator,
    create_collated_masks,
    linear_warmup_schedule,  # TODO: import from LightlySSL after new release
)
from lightly_train._methods.method import Method, TrainingStepResult
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.trainable_modules import TrainableModules
from lightly_train._scaling import ScalingInfo
from lightly_train.types import Batch


@dataclass
class DINOv2TrainingStepResult(TrainingStepResult):
    dino_global_loss: Tensor
    dino_local_loss: Tensor | None
    ibot_loss: Tensor
    koleo_loss: Tensor


class DINOv2Args(MethodArgs):
    """Args for DINOv2 method for ImageNet dataset."""

    # crops
    n_global_crops: int = 2
    n_local_crops: int = (
        8  # transform_cls().transform_args_cls().transform_args.local_view.num_views
    )

    # projection head
    ibot_separate_head: bool = False
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    output_dim: int = 65536
    batch_norm: bool = False
    student_freeze_last_layer_epochs: int = 1
    norm_last_layer: bool = False
    # NOTE: head_n_layers is 3 for all heads so we use LightlySSL's DINO head

    # loss
    dino_loss_weight: float = 1.0
    ibot_loss_weight: float = 1.0
    koleo_loss_weight: float = 0.1

    student_temp: float = 0.1

    centering: Literal["softmax", "sinkhorn_knopp"] = "softmax"
    center_momentum: float = 0.9

    # teacher momentum
    momentum_start: float = 0.992
    momentum_end: float = 1.0

    # teacher temp scheduler
    start_teacher_temp: float = 0.04
    end_teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 30

    # masking
    mask_ratio_min: float = 0.1
    mask_ratio_max: float = 0.5
    mask_probability: float = 0.5

    # lr scheduler
    min_lr: float = 1.0e-06
    warmup_epochs: int = 10

    # lr decay
    layerwise_decay: float = 0.9
    patch_embed_lr_multiplier: float = 0.2

    # weight decay scheduler
    weight_decay_end: float = 0.4

    # gradient clipping
    gradient_clip_val: float = 3.0

    def resolve_auto(
        self,
        scaling_info: ScalingInfo,
        optimizer_args: OptimizerArgs,
        model: Module,
    ) -> None:
        # Determine the args based on the model architecture
        depth: int = model.n_blocks
        num_heads: int = model.num_heads
        self.embed_dim: int = model._embed_dim
        if depth == 40 and num_heads == 24 and self.embed_dim == 1536:  # giant
            # projection head
            self.ibot_separate_head = True
            self.bottleneck_dim = 384
            self.bottleneck_dim_ibot = 256
            self.output_dim = 131072
            # loss
            self.centering = "sinkhorn_knopp"
            # teacher momentum
            self.momentum_start = 0.994
            # lr scheduler
            self.warmup_epochs = 80
            # lr decay
            self.layerwise_decay = 1.0
            # weight decay scheduler
            self.weight_decay_end = 0.2

        elif depth == 24 and num_heads == 16 and self.embed_dim == 1024:  # large
            # projection head
            self.ibot_separate_head = True
            self.bottleneck_dim = 384
            self.bottleneck_dim_ibot = 256
            self.output_dim = 131072
            # loss
            self.centering = "sinkhorn_knopp"
            # teacher momentum
            self.momentum_start = 0.994
            # lr scheduler
            self.warmup_epochs = 80
            # lr decay
            self.layerwise_decay = 1.0
            # weight decay scheduler
            self.weight_decay_end = 0.2

        elif depth == 12 and num_heads == 12 and self.embed_dim == 768:  # base
            pass
        elif depth == 12 and num_heads == 6 and self.embed_dim == 384:  # small
            pass
        else:
            raise ValueError("Unsupported model configs.")

        # TODO: scale some params based on the scaling info


class DINOv2(Method):
    def __init__(
        self,
        method_args: DINOv2Args,
        optimizer_args: DINOv2AdamWViTSBArgs | DINOv2AdamWViTLGArgs,
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
        self.method_args = method_args
        self.optimizer_args = optimizer_args

        # Calculate the number of crops
        self.n_global_crops = self.method_args.n_global_crops
        self.n_local_crops = self.method_args.n_local_crops
        self.n_global_crops_loss_terms = (self.n_global_crops - 1) * self.n_global_crops
        self.n_local_crops_loss_terms = max(self.n_local_crops * self.n_global_crops, 1)

        # Create teacher and student embedding models
        model_wrapper: DINOv2ViTModelWrapper = embedding_model.model_wrapper
        self.teacher_embedding_model_wrapper = model_wrapper
        self.student_embedding_model_wrapper = copy.deepcopy(
            self.teacher_embedding_model_wrapper
        )

        # Create teacher and student dino heads
        self.teacher_dino_head = DINOProjectionHead(
            input_dim=self.method_args.embed_dim,
            hidden_dim=self.method_args.hidden_dim,
            bottleneck_dim=self.method_args.bottleneck_dim,
            output_dim=self.method_args.output_dim,
            batch_norm=self.method_args.batch_norm,
            norm_last_layer=self.method_args.norm_last_layer,
        )
        self.student_dino_head = DINOProjectionHead(
            input_dim=self.method_args.embed_dim,
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
                input_dim=self.method_args.embed_dim,
                hidden_dim=self.method_args.hidden_dim,
                bottleneck_dim=self.method_args.bottleneck_dim_ibot,
                output_dim=self.method_args.output_dim,
                batch_norm=self.method_args.batch_norm,
                norm_last_layer=self.method_args.norm_last_layer,
            )
            self.student_ibot_head = DINOProjectionHead(
                input_dim=self.method_args.embed_dim,
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
        self.centering = self.method_args.centering
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

        loss = training_step_log.loss
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

        return loss

    def training_step_impl(
        self, batch: Batch, batch_idx: int
    ) -> DINOv2TrainingStepResult:
        # Momentum update teacher.
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=no_auto(self.method_args.momentum_start),
            end_value=self.method_args.momentum_end,
        )
        update_momentum(
            self.student_embedding_model_wrapper._model,
            self.teacher_embedding_model_wrapper._model,
            m=momentum,
        )
        update_momentum(self.student_dino_head, self.teacher_dino_head, m=momentum)

        # Teacher temperature scheduling
        self.teacher_temp = linear_warmup_schedule(
            step=self.trainer.global_step,
            warmup_steps=int(
                self.method_args.warmup_teacher_temp_epochs  # type: ignore[operator]
                / self.trainer.max_epochs
                * self.trainer.estimated_stepping_batches
            ),
            start_value=self.method_args.start_teacher_temp,
            end_value=self.method_args.end_teacher_temp,
        )

        views = batch["views"]
        global_views = torch.cat(
            views[: self.n_global_crops]
        )  # G * [B, C, H, W] -> [G*B, C, H, W]

        # Masking
        self.n_crops = global_views.shape[0]  # G*B
        self.n_channels = global_views.shape[1]  # C
        h = global_views.shape[2]
        w = global_views.shape[3]

        mask_generator = MaskingGenerator(
            input_size=(h, w),
            max_num_patches=int(0.5 * h * w),
        )
        n_masked_crops = int(self.n_crops * self.method_args.mask_probability)
        mask_ratio_min = self.method_args.mask_ratio_min
        mask_ratio_max = self.method_args.mask_ratio_max
        masks = create_collated_masks(
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            n_masked_crops=n_masked_crops,
            n_crops=self.n_crops,
            mask_generator=mask_generator,
        )

        self.collated_masks = masks["collated_masks"].to(
            device=self.device, non_blocking=True
        )
        self.mask_indices_list = masks["mask_indices_list"].to(
            device=self.device, non_blocking=True
        )
        self.masks_weight = masks["masks_weight"].to(
            device=self.device, non_blocking=True
        )
        self.n_masked_patches = self.mask_indices_list.shape[0]
        self.n_masked_patches_max = int(
            (n_masked_crops + 1) * h * w * (mask_ratio_min + mask_ratio_max) / 2.0
        )

        # Process global views through teacher and student networks
        teacher_cls_tokens_centered, teacher_masked_patch_tokens_centered = (
            self._forward_teacher(global_views)  # [G, B, D], [M, D]
        )
        student_cls_tokens_global, student_masked_patch_tokens_global = (
            self._forward_student_global(global_views)  # [G*B, D], [M, D]
        )

        # Process local views through student network if they exist
        if self.n_local_crops > 0:
            local_views = torch.cat(
                views[self.n_global_crops :]
            )  # L * [B, C, H, W] -> [L*B, C, H, W]
            student_cls_tokens_local = self._forward_student_local(
                local_views
            )  # [L*B, D]

        # Compute the DINO loss
        dino_global_loss = (
            self.dino_loss.forward(
                student_output_list=[student_cls_tokens_global],  # [[G*B, D]]
                teacher_out_softmaxed_centered_list=[
                    teacher_cls_tokens_centered.flatten(0, 1)
                ],  # [[G*B, D]], these were chunked and stacked in reverse so A is matched to B,
            )
            * self.n_global_crops
            / (self.n_global_crops_loss_terms + self.n_local_crops_loss_terms)
        )
        if self.n_local_crops > 0:
            dino_local_loss = (
                self.dino_loss.forward(
                    student_output_list=student_cls_tokens_local.chunk(
                        self.n_local_crops
                    ),  # [L, B, D]
                    teacher_out_softmaxed_centered_list=teacher_cls_tokens_centered,  # [G, B, D]
                )
                / (self.n_global_crops_loss_terms + self.n_local_crops_loss_terms)
            )

        # Compute the iBOT loss
        ibot_loss = self.ibot_loss.forward_masked(
            student_patch_tokens_masked=student_masked_patch_tokens_global,
            teacher_patch_tokens_masked=teacher_masked_patch_tokens_centered,
            student_masks_flat=self.collated_masks,
            n_masked_patches=self.n_masked_patches,
            masks_weight=self.masks_weight,
        )

        koleo_loss = sum(
            self.koleo_loss(token)
            for token in student_cls_tokens_global.chunk(self.n_global_crops)
        )  # [G, B, D], only use global views

        loss = (
            self.dino_loss_weight * dino_global_loss
            + self.dino_loss_weight * dino_local_loss
            if dino_local_loss is not None
            else 0.0
            + self.ibot_loss_weight * ibot_loss
            + self.koleo_loss_weight * koleo_loss
        )

        return DINOv2TrainingStepResult(
            loss=loss,
            dino_global_loss=dino_global_loss,
            dino_local_loss=dino_local_loss if self.n_local_crops > 0 else None,
            ibot_loss=ibot_loss,
            koleo_loss=koleo_loss,
        )

    @torch.no_grad()
    def _forward_teacher(self, x: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.teacher_embedding_model_wrapper.forward_features(
            x
        )  # input [G*B, C, ...]
        patches = tokens["features"]  # [G*B, C, H, W]
        cls_tokens = tokens["cls_token"]  # [G*B, C]

        # process the cls tokens
        cls_tokens_chunked = cls_tokens.chunk(self.n_global_crops)  # [G, B, C]
        # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
        cls_tokens = torch.cat(
            (cls_tokens_chunked[1], cls_tokens_chunked[0])
        )  # [G*B, C]

        # process the patch tokens
        patches = patches.flatten(2).permute(0, 2, 1)  # [G*B, H*W, C]

        # forward through the teacher dino and ibot heads
        if not self.ibot_separate_head:
            buffer_tokens = patches.new_zeros(
                self.n_crops + self.n_masked_patches_max, self.n_channels
            )  # [G*B+M_max, C]
            buffer_tokens[: self.n_crops].copy_(cls_tokens)
            torch.index_select(
                patches.flatten(0, 1),  # [G*B*H*W, C]
                dim=0,
                index=self.mask_indices_list,
                out=buffer_tokens[self.n_crops : self.n_crops + self.n_masked_patches],
            )  # [M, C]

            tokens_after_head = self.teacher_dino_head.forward(
                buffer_tokens
            )  # [G*B+M_max, D]

            cls_tokens_after_dino = tokens_after_head[: self.n_crops]  # [G*B, D]
            masked_patch_tokens_after_ibot = tokens_after_head[
                self.n_crops : self.n_crops + self.n_masked_patches
            ]  # [M, D]
        else:
            masked_patch_tokens = torch.index_select(
                patches.flatten(0, 1),  # [G*B*H*W, C]
                dim=0,
                index=self.mask_indices_list,
            )  # [M, C]

            cls_tokens_after_dino = self.teacher_dino_head.forward(
                cls_tokens
            )  # [G*B, D]
            masked_patch_tokens_after_ibot = self.teacher_ibot_head.forward(
                masked_patch_tokens
            )  # [M, D]

        # centering
        if self.centering == "softmax":
            cls_tokens_centered = self.dino_loss.softmax_center_teacher(
                cls_tokens_after_dino, teacher_temp=self.teacher_temp
            ).view(
                self.n_global_crops, -1, *cls_tokens_after_dino.shape[1:]
            )  # [G, B, D]
            self.dino_loss.update_center(cls_tokens_after_dino)

            masked_patch_tokens_centered = self.ibot_loss.softmax_center_teacher(
                masked_patch_tokens_after_ibot[: self.n_masked_patches],
                teacher_temp=self.teacher_temp,
            )  # [M, D]
            self.ibot_loss.update_center(
                masked_patch_tokens_after_ibot[: self.n_masked_patches]
            )
        elif self.centering == "sinkhorn_knopp":
            cls_tokens_centered = self.dino_loss.sinkhorn_knopp_teacher(
                cls_tokens_after_dino, teacher_temp=self.teacher_temp
            ).view(
                self.n_global_crops, -1, *cls_tokens_after_dino.shape[1:]
            )  # [G, B, D]

            masked_patch_tokens_centered = self.ibot_loss.sinkhorn_knopp_teacher(
                masked_patch_tokens_after_ibot,
                teacher_temp=self.teacher_temp,
                n_masked_patches_tensor=torch.tensor(
                    [self.n_masked_patches], dtype=torch.long
                ).to(device=self.device, non_blocking=True),
            )  # [M, D]
        else:
            raise ValueError(f"Unknown centering method: {self.centering}")

        return cls_tokens_centered, masked_patch_tokens_centered

    def _forward_student_global(self, x: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.student_embedding_model_wrapper.forward_features(
            x
        )  # input [G*B, C, ...]
        patches = tokens["features"]  # [G*B, C, H, W]
        cls_tokens = tokens["cls_token"]  # [G*B, C]

        # process the patch tokens
        patches = patches.flatten(2).permute(0, 2, 1)  # [G*B, H*W, C]

        # forward through the student dino and ibot heads
        if not self.ibot_separate_head:
            buffer_tokens = patches.new_zeros(
                self.n_crops + self.n_masked_patches_max, self.n_channels
            )  # [G*B+M_max, C]
            buffer_tokens[: self.n_crops].copy_(cls_tokens)
            torch.index_select(
                patches.flatten(0, 1),  # [G*B*H*W, C]
                dim=0,
                index=self.mask_indices_list,
                out=buffer_tokens[self.n_crops : self.n_crops + self.n_masked_patches],
            )  # [M, C]

            tokens_after_head = self.student_dino_head.forward(
                buffer_tokens
            )  # [G*B+M_max, D]

            cls_tokens_after_dino = tokens_after_head[: self.n_crops]  # [G*B, D]
            masked_patch_tokens_after_ibot = tokens_after_head[
                self.n_crops : self.n_crops + self.n_masked_patches
            ]  # [M, D]
        else:
            masked_patch_tokens = torch.index_select(
                patches.flatten(0, 1),  # [G*B*H*W, C]
                dim=0,
                index=self.mask_indices_list,
            )  # [M, C]

            cls_tokens_after_dino = self.student_dino_head.forward(
                cls_tokens
            )  # [G*B, D]
            masked_patch_tokens_after_ibot = self.student_ibot_head.forward(
                masked_patch_tokens
            )  # [M, D]

        return cls_tokens_after_dino, masked_patch_tokens_after_ibot

    def _forward_student_local(self, x: Tensor) -> Tensor:
        tokens = self.student_embedding_model_wrapper.forward_features(
            x
        )  # input [L*B, C, ...]
        cls_tokens = tokens["cls_token"]  # [L*B, C]

        # forward through the teacher dino and ibot heads
        cls_tokens_after_dino: Tensor = self.student_dino_head.forward(
            cls_tokens
        )  # [L*B, D]

        return cls_tokens_after_dino

    @staticmethod
    def method_args_cls() -> type[DINOv2Args]:
        return DINOv2Args

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[DINOv2AdamWViTSBArgs | DINOv2AdamWViTLGArgs]:
        classes: dict[OptimizerType | Literal["auto"], type[OptimizerArgs]] = {
            OptimizerType.ADAMW: DINOv2AdamWViTSBArgs,
        }

        return classes.get(optim_type, Method.optimizer_args_cls(optim_type=optim_type))

    def trainable_modules(self) -> TrainableModules:
        return TrainableModules(
            modules=[
                self.student_embedding_model_wrapper._model,
                self.student_dino_head,
                self.student_ibot_head,
            ],  # decay is realized in get_optimizer_with_decay
        )

    # Ignore the return type, because pytorch-lightning types it wrongly.
    # See https://github.com/Lightning-AI/pytorch-lightning/issues/20106
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim = get_optimizer_with_decay(
            optim_args=self.optimizer_args,
            trainable_modules=self.trainable_modules(),
            lr_scale=math.sqrt(self.global_batch_size / 1024),  # square root scaling
            layerwise_decay=self.method_args.layerwise_decay,
            patch_embed_lr_multiplier=self.method_args.patch_embed_lr_multiplier,
        )

        if self.trainer.max_epochs is None:
            raise RuntimeError("Max epochs is not set.")

        max_epochs = max(1, self.trainer.max_epochs)

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optim,
                warmup_epochs=int(
                    self.trainer.estimated_stepping_batches
                    / max_epochs
                    * self.method_args.warmup_epochs
                ),
                max_epochs=int(self.trainer.estimated_stepping_batches),
                start_value=self.optimizer_args.lr,
                end_value=self.method_args.min_lr,
            ),
            "interval": "step",
        }
        return [optim], [scheduler]  # type: ignore[return-value]

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        self.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm="norm",
        )

    def on_before_optimizer_step(self, optimizer: Optimizer, *args: Any) -> None:
        self.student_dino_head.cancel_last_layer_gradients(self.current_epoch)
        self.student_ibot_head.cancel_last_layer_gradients(self.current_epoch)

        # Apply weight decay schedule
        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.optimizer_args.weight_decay,
            end_value=self.method_args.weight_decay_end,
        )

        updates = []
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                updates.append({"name": group["name"], "weight_decay": weight_decay})

        update_param_groups(optimizer, updates=updates)

    @staticmethod
    def transform_cls() -> type[DINOv2ViTSBTransform]:
        return DINOv2ViTSBTransform
