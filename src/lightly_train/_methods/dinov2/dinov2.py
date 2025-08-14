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
from functools import partial
from typing import Any, ClassVar, Literal, Mapping

import torch
from lightly.loss import (
    KoLeoLoss,
)  # we use LightlySSL's KoLeoLoss for better numerical stability
from lightly.models.utils import update_momentum
from lightly.utils.optim import update_param_groups
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from pydantic import Field
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from lightly_train._configs.validate import no_auto
from lightly_train._methods.dinov2.dinov2_head import DINOv2ProjectionHead
from lightly_train._methods.dinov2.dinov2_loss import (
    DINOLoss,
    IBOTPatchLoss,
)  # we use the original DINOLoss and IBOTPatchLoss
from lightly_train._methods.dinov2.dinov2_transform import (
    DINOv2ViTTransform,
)

# TODO(Guarin, 06/25): import linear_warmup_schedule from LightlySSL once we no longer
# support LightlySSL <= 1.5.20
from lightly_train._methods.dinov2.scheduler import linear_warmup_schedule
from lightly_train._methods.dinov2.utils import (
    MaskingGenerator,
    create_collated_masks,
    get_optimizer_with_decay,
)
from lightly_train._methods.method import Method, TrainingStepResult
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.trainable_modules import TrainableModules
from lightly_train._scaling import ScalingInfo
from lightly_train.types import Batch

logger = logging.getLogger(__name__)


def freeze_eval_module(module: Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class DINOv2Args(MethodArgs):
    """Args for DINOv2 method following the fast setup from the original DINOv2 paper.

    See: https://github.com/facebookresearch/dinov2/tree/main?tab=readme-ov-file#training
    """

    default_steps: ClassVar[int | None] = 125_000
    default_epochs: ClassVar[int | None] = None

    # projection head
    # False/True for fast/long setup in original DINOv2
    ibot_separate_head: bool = False
    hidden_dim: int = 2048
    dino_bottleneck_dim: int = 256  # 256/384 for fast/long setup in original DINOv2
    ibot_bottleneck_dim: int = 256
    output_dim: int = 65536  # 65536/131072 for fast/long setup in original DINOv2
    batch_norm: bool = False
    student_freeze_last_layer_steps: int = 1250

    # freeze student backbone
    # Useful when starting from DINOv2 pretrained weights. This allows the projection
    # head to be trained while the backbone is frozen. This is important because the
    # DINOv2 pretrained weights do not include the projection head.
    student_freeze_backbone_steps: int = 0

    # loss
    dino_loss_weight: float = 1.0
    ibot_loss_weight: float = 1.0
    koleo_loss_weight: float = 0.1

    # softmax/sinkhorn_knopp for fast/long setup in original DINOv2
    center_method: Literal["softmax", "sinkhorn_knopp"] = "softmax"
    center_momentum: float = 0.9

    # teacher momentum
    # TODO(Guarin, 06/25): Figure out good momentum start value for smaller datasets.
    momentum_start: float = 0.992  # 0.992/0.994 for fast/long setup in original DINOv2
    momentum_end: float = 1.0

    student_temp: float = 0.1
    # TODO(Guarin, 06/25): Figure out good teacher temp start/end values for smaller
    # datasets.
    teacher_temp_start: float = 0.04
    teacher_temp_end: float = 0.07
    teacher_temp_warmup_steps: int = 37500

    # masking
    mask_ratio_min: float = 0.1
    mask_ratio_max: float = 0.5
    mask_probability: float = 0.5

    # lr scheduler
    min_lr: float = 1.0e-06
    warmup_steps: int = 12500

    # lr decay
    layerwise_decay: float = 0.9  # 0.9/1.0 for fast/long setup in original DINOv2
    patch_embed_lr_multiplier: float = 0.2

    # lr scaling
    lr_scale_method: Literal["linear", "sqrt"] = "sqrt"
    reference_batch_size: int = 1024

    # weight decay scheduler
    weight_decay_start: float | Literal["auto"] = "auto"
    # TODO(Guarin, 06/25): Should we adjust weight decay depending on model
    # architecture?
    weight_decay_end: float = 0.4  # 0.4/0.2 for fast/long setup in original DINOv2

    # gradient clipping
    gradient_clip_val: float = 3.0

    def resolve_auto(
        self,
        scaling_info: ScalingInfo,
        optimizer_args: OptimizerArgs,
        wrapped_model: ModelWrapper,
    ) -> None:
        if isinstance(optimizer_args, AdamWArgs):
            weight_decay = optimizer_args.weight_decay
        else:
            raise ValueError(f"Unsupported optimizer_args type: {type(optimizer_args)}")
        if self.weight_decay_start == "auto":
            self.weight_decay_start = weight_decay


class DINOv2AdamWViTArgs(AdamWArgs):
    # 0.004/0.0002 for fast/long setup in original DINOv2
    # 0.002 works well with ViT-S/14 for ImageNet1k
    lr: float = 0.004
    # Strict is set to False because OmegaConf does not support parsing tuples from the
    # CLI. Setting strict to False allows Pydantic to convert lists to tuples.
    betas: tuple[float, float] = Field(default=(0.9, 0.999), strict=False)
    eps: float = 1e-8
    weight_decay: float = 0.04


class DINOv2Head(Module):
    def __init__(
        self, dino_head: DINOv2ProjectionHead, ibot_head: DINOv2ProjectionHead
    ) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2(Method):
    RECOMMENDED_MIN_STEPS = 125000

    def __init__(
        self,
        method_args: DINOv2Args,
        optimizer_args: DINOv2AdamWViTArgs,
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

        # Create teacher and student embedding models
        self.teacher_embedding_model = embedding_model
        self.student_embedding_model = copy.deepcopy(self.teacher_embedding_model)

        wrapped_model: DINOv2ViTModelWrapper = (
            self.teacher_embedding_model.wrapped_model  # type: ignore[assignment]
        )
        wrapped_model.make_teacher()
        freeze_eval_module(self.teacher_embedding_model)

        model = wrapped_model.get_model()
        self._patch_size = model.patch_size

        # Create teacher and student heads
        dino_head = partial(
            DINOv2ProjectionHead,
            in_dim=model.embed_dim,
            hidden_dim=method_args.hidden_dim,
            bottleneck_dim=method_args.dino_bottleneck_dim,
            out_dim=method_args.output_dim,
            use_bn=method_args.batch_norm,
        )
        teacher_dino_head = dino_head()
        student_dino_head = dino_head()

        ibot_head = partial(
            DINOv2ProjectionHead,
            in_dim=model.embed_dim,
            hidden_dim=method_args.hidden_dim,
            bottleneck_dim=method_args.dino_bottleneck_dim,
            out_dim=method_args.output_dim,
            use_bn=method_args.batch_norm,
        )
        teacher_ibot_head = teacher_dino_head
        student_ibot_head = student_dino_head
        if method_args.ibot_separate_head:
            teacher_ibot_head = ibot_head()
            student_ibot_head = ibot_head()

        self.teacher_head = DINOv2Head(
            dino_head=teacher_dino_head, ibot_head=teacher_ibot_head
        )
        self.student_head = DINOv2Head(
            dino_head=student_dino_head, ibot_head=student_ibot_head
        )
        freeze_eval_module(self.teacher_head)

        # Losses
        # TODO(Jonas 06/25): make two loss versions one for centering softmax and one for sinkhorn knopp,
        # so we could instantiate the corresponding one and remove logic form the train loop
        # LightlySSL solution: https://github.com/lightly-ai/lightly/blob/90ca6abf4cbd34df6e0b58f675d92dc194883602/lightly/models/modules/center.py#L1
        self.dino_loss = DINOLoss(
            out_dim=method_args.output_dim,
            student_temp=method_args.student_temp,
            center_momentum=method_args.center_momentum,
        )
        self.ibot_loss = IBOTPatchLoss(
            patch_out_dim=method_args.output_dim,
            student_temp=method_args.student_temp,
            center_momentum=method_args.center_momentum,
        )
        self.koleo_loss = KoLeoLoss()

    def training_step_impl(self, batch: Batch, batch_idx: int) -> TrainingStepResult:
        # Teacher temperature scheduling
        teacher_temp = linear_warmup_schedule(
            step=self.trainer.global_step,
            warmup_steps=self.method_args.teacher_temp_warmup_steps,
            start_value=self.method_args.teacher_temp_start,
            end_value=self.method_args.teacher_temp_end,
        )

        # Get the views
        views = batch["views"]
        # Calculate the number of crops
        n_global_crops = 2
        n_local_crops = len(views) - n_global_crops
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)

        global_views = torch.cat(
            views[:n_global_crops]
        )  # G * [B, C, H, W] -> [G*B, C, H, W]

        # Masking
        # TODO(Jonas 06/25): put the masking into a separate method
        n_crops = global_views.shape[0]  # G*B
        batch_size = n_crops // n_global_crops
        h = global_views.shape[2] // self._patch_size
        w = global_views.shape[3] // self._patch_size

        mask_generator = MaskingGenerator(
            input_size=(h, w),
            max_num_patches=int(
                0.5 * h * w
            ),  # NOTE: max patch ratio 0.5 is carried over from the original DINOv2 code, can be tuned
        )
        n_masked_crops = int(n_crops * self.method_args.mask_probability)
        masks = create_collated_masks(
            mask_ratio_min=self.method_args.mask_ratio_min,
            mask_ratio_max=self.method_args.mask_ratio_max,
            n_masked_crops=n_masked_crops,
            n_crops=n_crops,
            mask_generator=mask_generator,
        )

        collated_masks = masks["collated_masks"].to(
            device=self.device, non_blocking=True
        )
        mask_indices_list = masks["mask_indices_list"].to(
            device=self.device, non_blocking=True
        )
        masks_weight = masks["masks_weight"].to(device=self.device, non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]

        # Process global views through teacher and student networks
        # TODO(Jonas 06/25): kwargs
        # TODO(Jonas 06/25): consider to move all the forwards into a single forward
        teacher_cls_tokens_centered, teacher_masked_patch_tokens_centered = (
            self._forward_teacher(
                global_views,
                batch_size,
                mask_indices_list,
                n_masked_patches,
                teacher_temp,
            )  # [G, B, D], [M, D]
        )
        (
            student_cls_tokens_global,
            student_cls_tokens_global_before_head,
            student_masked_patch_tokens_global,
        ) = self._forward_student_global(
            x=global_views,
            masks=collated_masks,
            mask_indices_list=mask_indices_list,
        )  # [G*B, D], [M, D]

        # TODO(Jonas 06/25): clarify if we actually need this list variant --> simplify interface
        # Compute the DINO loss
        dino_global_loss = (
            self.dino_loss.forward(
                student_output_list=[student_cls_tokens_global],  # [[G*B, D]]
                teacher_out_softmaxed_centered_list=[
                    teacher_cls_tokens_centered.flatten(0, 1)
                ],  # [[G*B, D]], these were chunked and stacked in reverse so A is matched to B,
            )
            * 2
            / (n_global_crops_loss_terms + n_local_crops_loss_terms)
        )

        # Process local views through student network if they exist
        dino_local_loss = torch.zeros_like(dino_global_loss)
        # TODO(Jonas 06/25): since n_local_crops is known on instantiation, we could avoid the check and instead instantiate a get_local_views depending on the attribute, similar the forward local could be instantiated like that
        if n_local_crops > 0:
            local_views = torch.cat(
                views[n_global_crops:]
            )  # L * [B, C, H, W] -> [L*B, C, H, W]
            student_cls_tokens_local = self._forward_student_local(
                local_views
            )  # [L*B, D]

            # TODO(Jonas 06/25): ideally move everything to tensor only no list
            dino_local_loss = (
                self.dino_loss.forward(
                    student_output_list=student_cls_tokens_local.chunk(
                        n_local_crops
                    ),  # [L, B, D]
                    teacher_out_softmaxed_centered_list=teacher_cls_tokens_centered,  # [G, B, D]
                )
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

        # Compute the iBOT loss
        ibot_loss = self.ibot_loss.forward_masked(
            student_patch_tokens_masked=student_masked_patch_tokens_global,
            teacher_patch_tokens_masked=teacher_masked_patch_tokens_centered,
            student_masks_flat=collated_masks,
            n_masked_patches=n_masked_patches,
            masks_weight=masks_weight,
        )

        koleo_loss = sum(
            self.koleo_loss(token)
            for token in student_cls_tokens_global_before_head.chunk(2)
        )  # [G, B, D], only use global views

        loss = (
            self.method_args.dino_loss_weight * dino_global_loss
            + self.method_args.dino_loss_weight * dino_local_loss
            + self.method_args.ibot_loss_weight * ibot_loss
            + self.method_args.koleo_loss_weight * koleo_loss
        )

        return TrainingStepResult(
            loss=loss,
            log_dict={
                "train_loss/dino_global_loss": dino_global_loss,
                "train_loss/dino_local_loss": dino_local_loss,
                "train_loss/ibot_loss": ibot_loss,
                "train_loss/koleo_loss": koleo_loss,
            },
        )

    @torch.no_grad()
    def _forward_teacher(
        self,
        x: Tensor,
        batch_size: int,
        mask_indices_list: Tensor,
        n_masked_patches: int,
        teacher_temp: float,
    ) -> tuple[Tensor, Tensor]:
        tokens = self.teacher_embedding_model.wrapped_model.forward_features(
            x
        )  # input [G*B, C, ...]

        # process the cls tokens
        # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
        cls_tokens = tokens["cls_token"]  # [G*B, C]
        cls_tokens = torch.cat(
            (cls_tokens[batch_size:], cls_tokens[:batch_size])
        )  # [G*B, C]
        cls_tokens_after_dino = self.teacher_head.dino_head.forward(
            cls_tokens
        )  # [G*B, D]

        # process the masked patch tokens
        patch_tokens = tokens["features"]  # [G*B, C, H/p, W/p]
        # TODO(Jonas 06/25): why not flattening the patch tokens here all in one go?
        patch_tokens = patch_tokens.flatten(2).permute(0, 2, 1)  # [G*B, H/p*W/p, C]

        masked_patch_tokens = torch.index_select(
            patch_tokens.flatten(0, 1),  # [G*B*H/p*W/p, C]
            dim=0,
            index=mask_indices_list,
        )  # [M, C]
        masked_patch_tokens_after_ibot = self.teacher_head.ibot_head.forward(
            masked_patch_tokens
        )  # [M, D]

        # centering
        # TODO(Jonas 06/25): instantiate the centering method in the loss and remove the logic from here
        if self.method_args.center_method == "softmax":
            # TODO(Jonas 06/25): reshape the return inside the loss
            cls_tokens_centered = self.dino_loss.softmax_center_teacher(
                cls_tokens_after_dino, teacher_temp=teacher_temp
            ).view(2, -1, *cls_tokens_after_dino.shape[1:])  # [G, B, D]
            self.dino_loss.update_center(cls_tokens_after_dino)

            # TODO(Jonas 06/25): change the code inside the loss to avoid the unsqueeze
            masked_patch_tokens_after_ibot = masked_patch_tokens_after_ibot.unsqueeze(0)
            masked_patch_tokens_centered = self.ibot_loss.softmax_center_teacher(
                masked_patch_tokens_after_ibot,
                teacher_temp=teacher_temp,
            )  # [M, D]
            masked_patch_tokens_centered = masked_patch_tokens_centered.squeeze(0)
            self.ibot_loss.update_center(masked_patch_tokens_after_ibot)
        elif self.method_args.center_method == "sinkhorn_knopp":
            # TODO(Jonas 06/25): reshape the return inside the loss
            cls_tokens_centered = self.dino_loss.sinkhorn_knopp_teacher(
                cls_tokens_after_dino, teacher_temp=teacher_temp
            ).view(2, -1, *cls_tokens_after_dino.shape[1:])  # [G, B, D]

            masked_patch_tokens_centered = self.ibot_loss.sinkhorn_knopp_teacher(
                masked_patch_tokens_after_ibot,
                teacher_temp=teacher_temp,
                # TODO(Jonas 06/25): move this into the loss if required
                n_masked_patches_tensor=torch.tensor(
                    [n_masked_patches], dtype=torch.long
                ).to(device=self.device, non_blocking=True),
            )  # [M, D]
        else:
            raise ValueError(
                f"Unknown centering method: {self.method_args.center_method}"
            )

        return cls_tokens_centered, masked_patch_tokens_centered

    def _forward_student_global(
        self,
        x: Tensor,
        masks: Tensor,
        mask_indices_list: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        wrapped_model: DINOv2ViTModelWrapper = (
            self.student_embedding_model.wrapped_model  # type: ignore[assignment]
        )
        tokens = wrapped_model.forward_features(x=x, masks=masks)  # input [G*B, C, ...]

        # process the cls tokens
        cls_tokens = tokens["cls_token"]  # [G*B, C]
        cls_tokens_after_dino = self.student_head.dino_head.forward(
            cls_tokens
        )  # [G*B, D]

        # process the patch tokens
        patch_tokens = tokens["features"]  # [G*B, C, H/p, W/p]
        # TODO(Jonas 06/25): why not flattening the patch tokens here all in one go?
        patch_tokens = patch_tokens.flatten(2).permute(0, 2, 1)  # [G*B, H/p*W/p, C]

        masked_patch_tokens = torch.index_select(
            patch_tokens.flatten(0, 1),  # [G*B*H/p*W/p, C]
            dim=0,
            index=mask_indices_list,
        )  # [M, C]
        masked_patch_tokens_after_ibot = self.student_head.ibot_head.forward(
            masked_patch_tokens
        )  # [M, D]

        return cls_tokens_after_dino, cls_tokens, masked_patch_tokens_after_ibot

    def _forward_student_local(self, x: Tensor) -> Tensor:
        tokens = self.student_embedding_model.wrapped_model.forward_features(
            x
        )  # input [L*B, C, ...]

        # process the cls tokens
        # TODO(Jonas 06/25): unnecessary assignment, can be removed
        cls_tokens = tokens["cls_token"]  # [L*B, C]
        cls_tokens_after_dino: Tensor = self.student_head.dino_head.forward(
            cls_tokens
        )  # [L*B, D]

        return cls_tokens_after_dino

    @staticmethod
    def method_args_cls() -> type[DINOv2Args]:
        return DINOv2Args

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[OptimizerArgs]:
        classes: dict[OptimizerType | Literal["auto"], type[OptimizerArgs]] = {
            "auto": DINOv2AdamWViTArgs,
            OptimizerType.ADAMW: DINOv2AdamWViTArgs,
        }

        return classes.get(optim_type, Method.optimizer_args_cls(optim_type=optim_type))

    def trainable_modules(self) -> TrainableModules:
        # decay is realized in get_optimizer_with_decay
        return TrainableModules(
            modules=[
                # TODO(Guarin, 06/25): We should pass here the embedding model instead
                # of the wrapped model for clarity. But this requires changing
                # get_optimizer_with_decay.
                self.student_embedding_model.wrapped_model.get_model(),
                self.student_head,
            ],
        )

    # Ignore the return type, because pytorch-lightning types it wrongly.
    # See https://github.com/Lightning-AI/pytorch-lightning/issues/20106
    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Scale the learning rate based on the global batch size.
        lr_scale: float = self.global_batch_size / self.method_args.reference_batch_size
        if self.method_args.lr_scale_method == "sqrt":
            lr_scale = math.sqrt(lr_scale)

        self.optimizer_args.lr *= lr_scale  # type: ignore[attr-defined]
        optim = get_optimizer_with_decay(
            optim_args=self.optimizer_args,
            trainable_modules=self.trainable_modules(),
            layerwise_decay=self.method_args.layerwise_decay,
            patch_embed_lr_multiplier=self.method_args.patch_embed_lr_multiplier,
        )

        if self.trainer.max_epochs is None:
            raise RuntimeError("Max epochs is not set.")

        warmup_steps = min(
            # warmup_steps has to be smaller than the total number of steps because
            # of: https://github.com/lightly-ai/lightly/pull/1842
            # TODO(Guarin, 06/25): Remove this once we no longer support
            # LightlySSL <= 1.5.21.
            self.trainer.estimated_stepping_batches - 1,
            self.method_args.warmup_steps,
        )

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optim,
                # The arguments are called "epochs" but they can also be steps.
                warmup_epochs=int(warmup_steps),
                max_epochs=int(self.trainer.estimated_stepping_batches),
                end_value=self.method_args.min_lr / self.optimizer_args.lr,  # type: ignore[attr-defined]
            ),  # TODO: ignore to be removed after improving optimizer args
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
            gradient_clip_val=self.method_args.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )

    def on_before_optimizer_step(self, optimizer: Optimizer, *args: Any) -> None:
        # Apply weight decay schedule
        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=no_auto(self.method_args.weight_decay_start),
            end_value=self.method_args.weight_decay_end,
        )

        updates = []
        for group in optimizer.param_groups:
            update = {}
            # NOTE: If you change behavior of parameters here then make sure to also
            # double check dinov2/utils.py:get_fused_param_groups whether it needs
            # any updates.

            # Apply weight decay schedule
            if group["weight_decay"] != 0.0:
                update["weight_decay"] = weight_decay

            # Optionally freeze student backbone
            if (
                self.trainer.global_step
                < self.method_args.student_freeze_backbone_steps
                and "head" not in group["name"]
            ):
                update["lr"] = 0.0

            # Optionally freeze student last layer
            if (
                self.trainer.global_step
                < self.method_args.student_freeze_last_layer_steps
                and "last_layer" in group["name"]
            ):
                update["lr"] = 0.0

            if update:
                update["name"] = group["name"]
                updates.append(update)
        update_param_groups(optimizer, updates=updates)

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        # Momentum update teacher.
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.method_args.momentum_start,
            end_value=self.method_args.momentum_end,
        )
        update_momentum(
            self.student_embedding_model,
            self.teacher_embedding_model,
            m=momentum,
        )
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        super().on_train_batch_end(outputs=outputs, batch=batch, batch_idx=batch_idx)

    @staticmethod
    def transform_cls() -> type[DINOv2ViTTransform]:
        return DINOv2ViTTransform

    @rank_zero_only  # type: ignore[misc]
    def warn_if_steps_too_low(self) -> None:
        # Check if the dataloader is set (for mypy).
        assert self.trainer.train_dataloader is not None, (
            "dataloader must be set before training."
        )

        # Compute dataset-specific epoch recommendation.
        dataset_size = len(self.trainer.train_dataloader.dataset)
        steps_per_epoch = dataset_size // self.global_batch_size
        recommended_epochs = math.ceil(self.RECOMMENDED_MIN_STEPS / steps_per_epoch)
        total_num_steps = self.trainer.estimated_stepping_batches

        # Display recommendation.
        logger.warning(
            f"Configured epochs ({self.trainer.max_epochs}) will result in "
            f"{total_num_steps} steps, which is fewer "
            f"than {self.RECOMMENDED_MIN_STEPS} steps. "
            f"Recommended at least {recommended_epochs} epochs "
            f"for dataset size {dataset_size} and "
            f"batch size {self.global_batch_size}."
        )

    def on_fit_start(self) -> None:
        # Warn if total steps < 125k.
        if self.trainer.estimated_stepping_batches < self.RECOMMENDED_MIN_STEPS:
            self.warn_if_steps_too_low()
