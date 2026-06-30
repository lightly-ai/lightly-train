#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import torch
from lightly.utils.scheduler import CosineWarmupScheduler
from lightning_fabric import Fabric
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from lightly_train._configs.validate import no_auto
from lightly_train._data.depth_estimation_dataset import DepthEstimationDataArgs
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._metrics.depth_estimation.task_metric import (
    DepthEstimationTaskMetric,
    DepthEstimationTaskMetricArgs,
)
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOV2_VIT_PACKAGE
from lightly_train._optim import optimizer_helpers
from lightly_train._task_models.depth_estimation.criterion import (
    GlobalLocalLoss,
    GradientMatchingLoss,
    SILogLoss,
    SkyDistillLoss,
)
from lightly_train._task_models.depth_estimation.task_model import (
    DepthAnythingDepthEstimation,
)
from lightly_train._task_models.depth_estimation.transforms import (
    DepthEstimationTrainTransform,
    DepthEstimationTrainTransformArgs,
    DepthEstimationValTransform,
    DepthEstimationValTransformArgs,
)
from lightly_train._task_models.train_model import (
    TaskStepResult,
    TrainModel,
    TrainModelArgs,
)
from lightly_train._torch_compile import TorchCompileArgs
from lightly_train._visualize.depth_estimation import (
    DepthEstimationTaskStepVisualization,
)
from lightly_train.types import DepthEstimationBatch, PathLike

_LOSS_NAMES = ["loss", "gl_loss", "silog_loss", "grad_loss", "sky_loss"]


class DepthEstimationTrainArgs(TrainModelArgs):
    # Training uses a single fixed resolution (the DepthAnything V3 base 504x504,
    # resolved from the model config) with a fixed batch size. The paper's
    # multi-resolution-per-step sampling and its constant-token-count dynamic batch
    # sizing are intentionally not implemented.
    default_batch_size: ClassVar[int] = 8
    default_steps: ClassVar[int] = 40_000

    # Backbone args.
    backbone_freeze: bool = False
    backbone_weights: PathLike | None = None
    # Multiplier on the backbone learning rate relative to the decoder. The backbone is
    # pretrained while the DPT/sky heads are random, so the backbone is fine-tuned at a
    # lower rate by default. Ignored when ``backbone_freeze`` is set.
    backbone_lr_factor: float = 0.1

    gradient_clip_val: float | Literal["auto"] = "auto"

    # Optim.
    lr: float = 5e-5
    weight_decay: float | Literal["auto"] = "auto"
    lr_warmup_steps: int | Literal["auto"] = "auto"

    # Loss. The objective follows DepthAnything V3 (arXiv:2511.10647, Eq. 7):
    # ``L = L_gl + α·L_grad + L_sky`` with ``α = 0.5``. The global-local loss ``L_gl`` is
    # the depth term; the scale-invariant log loss is kept wired but disabled
    # (``silog_loss_weight = 0.0``) so it can be re-enabled for ablations.
    gl_loss_weight: float = 1.0
    gl_loss_trunc: float = 1.0
    silog_loss_weight: float = 0.0
    silog_lambda: float = 0.5
    grad_loss_weight: float = 0.5
    grad_loss_scales: int = 4
    sky_loss_weight: float = 1.0

    def resolve_auto(
        self,
        total_steps: int,
        gradient_accumulation_steps: int,
        train_num_batches: int,
        model_name: str,
        model_init_args: dict[str, Any],
        data_args: TaskDataArgs,
    ) -> None:
        if self.weight_decay == "auto":
            self.weight_decay = 0.0 if self.backbone_freeze else 0.01
        if self.lr_warmup_steps == "auto":
            self.lr_warmup_steps = 0 if self.backbone_freeze else min(500, total_steps)
        if self.gradient_clip_val == "auto":
            self.gradient_clip_val = 0.0 if self.backbone_freeze else 3.0


class DepthEstimationTrain(TrainModel):
    task = "depth_estimation"
    train_model_args_cls = DepthEstimationTrainArgs
    task_metric_args_cls = DepthEstimationTaskMetricArgs
    task_model_cls = DepthAnythingDepthEstimation
    train_transform_cls = DepthEstimationTrainTransform
    val_transform_cls = DepthEstimationValTransform
    torch_compile_args_cls = TorchCompileArgs

    def __init__(
        self,
        *,
        model_name: str,
        model_args: DepthEstimationTrainArgs,
        data_args: DepthEstimationDataArgs,
        train_transform_args: DepthEstimationTrainTransformArgs,
        val_transform_args: DepthEstimationValTransformArgs,
        load_weights: bool,
        metric_args: DepthEstimationTaskMetricArgs,
        gradient_accumulation_steps: int,
    ) -> None:
        super().__init__()
        self.model_args = model_args
        self.metric_args = metric_args

        # Build the student without hosted weights; there is no small V3 checkpoint. When
        # starting a fresh run we load the DINOv2-pretrained backbone and keep the DPT and
        # sky heads randomly initialized.
        self.model = DepthAnythingDepthEstimation(
            model_name=model_name,
            load_weights=False,
        )
        if load_weights:
            _load_pretrained_backbone(
                model=self.model, backbone_weights=model_args.backbone_weights
            )

        self.gl_criterion = GlobalLocalLoss(trunc=model_args.gl_loss_trunc)
        self.silog_criterion = SILogLoss(lambd=model_args.silog_lambda)
        self.grad_criterion = GradientMatchingLoss(scales=model_args.grad_loss_scales)
        self.sky_criterion = SkyDistillLoss()

        self.val_metrics = DepthEstimationTaskMetric(
            task_metric_args=metric_args,
            split="val",
            loss_names=_LOSS_NAMES,
        )
        self.train_metrics = DepthEstimationTaskMetric(
            task_metric_args=metric_args,
            split="train",
            loss_names=_LOSS_NAMES,
            train_loss_running_mean_window=gradient_accumulation_steps,
        )

        # Visualization: denormalize the logged images with the same mean/std as the
        # transform so the RGB tiles look natural next to the colorized depth maps.
        normalize = no_auto(train_transform_args.normalize)
        self.image_normalize = {"mean": normalize.mean, "std": normalize.std}
        # TODO(Nauryz, 06/2026): This visualization limit is hardcoded, matching the
        # other tasks; it may become configurable via logger_args in the future.
        self.viz_max_images = 4

    def get_task_model(self) -> DepthAnythingDepthEstimation:
        return self.model

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        feats = self.model._extract_features(images)
        out: dict[str, Tensor] = self.model.decoder(
            feats=feats, H=images.shape[-2], W=images.shape[-1]
        )
        return out

    def training_step(
        self, fabric: Fabric, batch: DepthEstimationBatch, step: int
    ) -> TaskStepResult:
        return self._step(batch=batch, metrics=self.train_metrics, compute_metrics=True)

    def validation_step(
        self, fabric: Fabric, batch: DepthEstimationBatch, step: int
    ) -> TaskStepResult:
        return self._step(batch=batch, metrics=self.val_metrics, compute_metrics=True)

    def get_optimizer(
        self,
        total_steps: int,
        global_batch_size: int,
    ) -> tuple[Optimizer, LRScheduler]:
        _, params_no_wd_list = optimizer_helpers.get_weight_decay_parameters([self])
        params_no_wd = set(params_no_wd_list)

        lr = self.model_args.lr * math.sqrt(
            global_batch_size / self.model_args.default_batch_size
        )
        backbone_lr = lr * self.model_args.backbone_lr_factor

        # The backbone is pretrained and fine-tuned at the reduced backbone_lr; the DPT
        # decoder (depth + sky heads) is randomly initialized and trains at the full lr.
        backbone_params = set(self.model.backbone.parameters())
        param_groups: list[dict[str, Any]] = []
        for name, module_params, group_lr in (
            ("backbone", backbone_params, backbone_lr),
            (
                "decoder",
                set(self.model.decoder.parameters()),
                lr,
            ),
        ):
            params_wd = [
                p for p in module_params if p.requires_grad and p not in params_no_wd
            ]
            params_no_wd_group = [
                p for p in module_params if p.requires_grad and p in params_no_wd
            ]
            if params_wd:
                param_groups.append({"name": name, "params": params_wd, "lr": group_lr})
            if params_no_wd_group:
                param_groups.append(
                    {
                        "name": f"{name}_no_weight_decay",
                        "params": params_no_wd_group,
                        "lr": group_lr,
                        "weight_decay": 0.0,
                    }
                )

        optimizer = AdamW(
            params=param_groups,
            lr=lr,
            weight_decay=no_auto(self.model_args.weight_decay),
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=no_auto(self.model_args.lr_warmup_steps),
            max_epochs=total_steps,
        )
        return optimizer, scheduler

    def set_train_mode(self) -> None:
        self.train()
        if self.model_args.backbone_freeze:
            self.model.backbone.eval()
            for param in self.model.backbone.parameters():
                param.requires_grad_(False)

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        if no_auto(self.model_args.gradient_clip_val) > 0:
            fabric.clip_gradients(
                module=self,
                optimizer=optimizer,
                max_norm=no_auto(self.model_args.gradient_clip_val),
                error_if_nonfinite=False,
            )

    def _step(
        self,
        batch: DepthEstimationBatch,
        metrics: DepthEstimationTaskMetric,
        compute_metrics: bool,
    ) -> TaskStepResult:
        images = batch["image"]
        depth = batch["depth"]
        sky = batch["sky"]
        assert isinstance(images, Tensor)
        assert isinstance(depth, Tensor)
        assert isinstance(sky, Tensor)

        out = self(images)
        # Sky has no valid depth: the teacher's depth in sky regions is garbage, so
        # exclude it from the depth losses. The sky head still trains on the full sky
        # map below.
        depth_mask = (depth > 0) & (sky < 0.5)
        gl_loss = self.gl_criterion(out["depth"], depth, depth_mask)
        silog_loss = self.silog_criterion(out["depth"], depth, depth_mask)
        grad_loss = self.grad_criterion(out["depth"], depth, depth_mask)
        sky_loss = self.sky_criterion(out["sky"], sky)
        loss = (
            self.model_args.gl_loss_weight * gl_loss
            + self.model_args.silog_loss_weight * silog_loss
            + self.model_args.grad_loss_weight * grad_loss
            + self.model_args.sky_loss_weight * sky_loss
        )

        metrics.update_with_losses(
            {
                "loss": loss.detach(),
                "gl_loss": gl_loss.detach(),
                "silog_loss": silog_loss.detach(),
                "grad_loss": grad_loss.detach(),
                "sky_loss": sky_loss.detach(),
            },
            weight=images.shape[0],
        )
        if compute_metrics:
            # Zero out sky pixels so update_with_predictions' `target > 0` filter ignores
            # them, matching the sky exclusion applied to the loss.
            metric_target = torch.where(sky < 0.5, depth, torch.zeros_like(depth))
            metrics.update_with_predictions(out["depth"].detach(), metric_target)

        return TaskStepResult(
            loss=loss,
            log_dict={},
            metrics=metrics,
            visualization=DepthEstimationTaskStepVisualization(
                batch=batch,
                image_normalize=self.image_normalize,
                max_images=self.viz_max_images,
                pred_depth=out["depth"],
            ),
        )


def _load_pretrained_backbone(
    model: DepthAnythingDepthEstimation, *, backbone_weights: PathLike | None
) -> None:
    """Loads DINOv2-pretrained backbone weights into the student backbone in place.

    A local ``backbone_weights`` checkpoint takes precedence. Otherwise the public
    DINOv2 weights for the student's backbone are fetched and copied in. For backbones
    without hosted weights (e.g. the ``_vittest14`` test backbone) this is a no-op and
    the backbone keeps its random initialization.
    """
    if backbone_weights is not None:
        state_dict = torch.load(backbone_weights, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(state_dict, strict=True)
        return

    reference = DINOV2_VIT_PACKAGE.get_model(
        model_name=model.backbone_name,
        model_args=model.backbone_model_args,
        load_weights=True,
    )
    model.backbone.load_state_dict(reference.state_dict(), strict=True)
