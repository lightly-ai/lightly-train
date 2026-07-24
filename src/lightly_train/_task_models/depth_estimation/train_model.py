#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from typing import Any, ClassVar, Literal, cast

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
from lightly_train._optim import optimizer_helpers
from lightly_train._task_models.depth_estimation.criterion import (
    FeatureAlignmentLoss,
    GradientMatchingLoss,
    RelativeL1Loss,
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
from lightly_train._task_models.object_detection_components.ema import ModelEMA
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

_LOSS_NAMES = ["loss", "silog_loss", "grad_loss", "sky_loss", "abs_l1_loss"]
# Extra loss logged only when feature-space distillation is enabled.
_FEATURE_LOSS_NAME = "feat_loss"


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

    # Loss.
    silog_lambda: float = 0.5
    grad_loss_weight: float = 0.5
    grad_loss_scales: int = 4
    sky_loss_weight: float = 1.0
    # Weight of the relative-L1 (AbsRel) depth term (raw scale-aware RelativeL1Loss). Off by
    # default and metric-depth only: its gradient does not vanish with a global scale error
    # and so pins the student's absolute scale to the teacher (the scale-invariant SILog and
    # gradient terms give no such signal). It has no effect on relative-depth models, whose
    # global scale is intentionally left free.
    abs_l1_loss_weight: float = 0.0

    # Feature-space distillation of the teacher's DPT-input patch tokens. Disabled by
    # default (``feature_align_weight == 0``): when enabled the frozen ``feature_align_teacher``
    # model is loaded online and its intermediate features are aligned with the student's
    # via a learnable per-stage projection (see ``FeatureAlignmentLoss``). The teacher must
    # share the student's patch size and processing resolution and expose the same number
    # of intermediate layers (e.g. the default ViT-L teacher for a ViT-S student, both at
    # 14px/504px); only the backbone widths may differ, which the loss handles.
    feature_align_weight: float = 0.0
    feature_align_teacher: str = "dinov2/dav3-metric-large"
    feature_align_teacher_weights: PathLike | None = None

    # Exponential moving average of the weights, used for validation and export. Off by
    # default; when enabled, validation and the exported checkpoint use the EMA weights
    # (see the object-detection tasks for the same `ModelEMA` pattern). EMA smooths the
    # noisy tail of a not-yet-converged run and is a cheap variance reduction.
    use_ema: bool = False
    ema_decay: float = 0.9998
    ema_warmup_steps: int = 2000

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

        # Build the student without hosted depth weights; trainable V3 students load the
        # pretrained backbone separately and keep the DPT and sky heads randomly
        # initialized.
        self.model = DepthAnythingDepthEstimation(
            model_name=model_name,
            load_weights=False,
        )
        if load_weights:
            _load_pretrained_backbone(
                model=self.model, backbone_weights=model_args.backbone_weights
            )

        # Optional EMA of the task-model weights (backbone + DPT/sky heads), used for
        # validation and export. Built after the (optional) pretrained backbone load so
        # the shadow starts from the same weights as the live model. Reuses the shared
        # `ModelEMA` from the object-detection tasks. `None` when disabled so all
        # export/load paths behave exactly as before. EMA is an internal training trick
        # and is never surfaced to users: the exported checkpoint stores the EMA weights
        # under the normal task-model keys (see `get_export_state_dict`), so it is
        # indistinguishable from a non-EMA checkpoint.
        self.ema_model: ModelEMA | None = None
        if model_args.use_ema:
            self.ema_model = ModelEMA(
                model=self.model,
                decay=model_args.ema_decay,
                warmups=model_args.ema_warmup_steps,
            )

        # Metric-depth models predict depth with an absolute scale (`scale_mode != "none"`,
        # e.g. the DA3 canonical-camera output that inference scales by `focal / 300`).
        # Relative-depth models leave the global scale unconstrained. This flag switches
        # the depth loss from scale-invariant to scale-aware and disables the
        # scale-and-shift alignment in the validation metrics.
        self._is_metric = self.model._scale_mode != "none"

        # For metric models the depth term must be scale-aware, so the scale-invariant
        # mean-subtraction is turned off (`lambd=0` reduces SILog to a log-space L2, see
        # SILogLoss). Relative models keep the configured `silog_lambda`.
        silog_lambda = 0.0 if self._is_metric else model_args.silog_lambda
        self.silog_criterion = SILogLoss(lambd=silog_lambda)
        self.grad_criterion = GradientMatchingLoss(scales=model_args.grad_loss_scales)
        self.sky_criterion = SkyDistillLoss()
        # Relative-L1 (AbsRel) term (see abs_l1_loss_weight). Metric-depth only: it pins the
        # student's absolute scale to the teacher. Relative-depth models do not use it (their
        # global scale is intentionally free).
        self.abs_l1_criterion = RelativeL1Loss()

        # Feature-space distillation is opt-in: only when its weight is positive do we
        # load the (expensive) frozen teacher and add the alignment loss and its metric.
        self.feature_align_weight = model_args.feature_align_weight
        self.teacher: DepthAnythingDepthEstimation | None = None
        self.feature_align_criterion: FeatureAlignmentLoss | None = None
        loss_names = list(_LOSS_NAMES)
        if self.feature_align_weight > 0:
            teacher = _build_teacher(
                model_name=model_args.feature_align_teacher,
                weights=model_args.feature_align_teacher_weights,
            )
            if len(teacher.out_layers) != len(self.model.out_layers):
                raise ValueError(
                    "The teacher and student must expose the same number of "
                    f"intermediate layers for feature alignment, got "
                    f"{len(teacher.out_layers)} (teacher) and "
                    f"{len(self.model.out_layers)} (student)."
                )
            if teacher.patch_size != self.model.patch_size:
                raise ValueError(
                    "The teacher and student must share the patch size for feature "
                    f"alignment (the token grids must match), got "
                    f"{teacher.patch_size} (teacher) and "
                    f"{self.model.patch_size} (student)."
                )
            self.teacher = teacher
            self.feature_align_criterion = FeatureAlignmentLoss(
                student_dim=int(self.model.backbone.embed_dim),
                teacher_dim=int(teacher.backbone.embed_dim),
                num_stages=len(self.model.out_layers),
            )
            loss_names.append(_FEATURE_LOSS_NAME)

        self.val_metrics = DepthEstimationTaskMetric(
            task_metric_args=metric_args,
            split="val",
            loss_names=loss_names,
            align=not self._is_metric,
        )
        self.train_metrics = DepthEstimationTaskMetric(
            task_metric_args=metric_args,
            split="train",
            loss_names=loss_names,
            train_loss_running_mean_window=gradient_accumulation_steps,
            align=not self._is_metric,
        )

        # Visualization: denormalize the logged images with the same mean/std as the
        # transform so the RGB tiles look natural next to the colorized depth maps.
        normalize = no_auto(train_transform_args.normalize)
        self.image_normalize = {"mean": normalize.mean, "std": normalize.std}
        # TODO(Nauryz, 06/2026): This visualization limit is hardcoded, matching the
        # other tasks; it may become configurable via logger_args in the future.
        self.viz_max_images = 4

    def get_task_model(self) -> DepthAnythingDepthEstimation:
        # Inference and export use the EMA weights when EMA is enabled (the smoothed
        # weights validate better), otherwise the live model.
        if self.ema_model is not None:
            return cast(DepthAnythingDepthEstimation, self.ema_model.model)
        return self.model

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        return self._forward(model=self.model, images=images)

    def training_step(
        self, fabric: Fabric, batch: DepthEstimationBatch, step: int
    ) -> TaskStepResult:
        return self._step(
            model=self.model,
            batch=batch,
            metrics=self.train_metrics,
            compute_metrics=True,
        )

    def validation_step(
        self, fabric: Fabric, batch: DepthEstimationBatch, step: int
    ) -> TaskStepResult:
        # Validate with the EMA weights when enabled, otherwise the live model.
        model = (
            self.model
            if self.ema_model is None
            else cast(DepthAnythingDepthEstimation, self.ema_model.model)
        )
        return self._step(
            model=model,
            batch=batch,
            metrics=self.val_metrics,
            compute_metrics=True,
        )

    def on_train_batch_end(self) -> None:
        # Update the EMA shadow after each optimizer step (framework hook, called from
        # the training loop). No-op when EMA is disabled.
        if self.ema_model is not None:
            self.ema_model.update(self.model)

    def get_export_state_dict(self) -> dict[str, Any]:
        """Returns the state dict for exporting.

        EMA is an internal training trick and is not surfaced to users: when it is
        enabled the exported checkpoint stores the *EMA* weights (which validate better)
        but under the normal live-model (``model.``) keys, and the ``ema_model.`` keys
        are dropped. The result is byte-for-byte structurally identical to a non-EMA
        export, so downstream loading (into the task model or a fresh train model) is
        unchanged and callers cannot tell EMA was used.
        """
        state_dict = super().get_export_state_dict()
        if self.ema_model is None:
            return state_dict

        prefix = "ema_model.model."
        # Overwrite each live-model weight with its EMA counterpart, then drop all
        # `ema_model.` entries so only the standard train-model keys remain.
        ema_weights = {
            f"model.{name[len(prefix) :]}": value
            for name, value in state_dict.items()
            if name.startswith(prefix)
        }
        result = {
            name: value
            for name, value in state_dict.items()
            if not name.startswith("ema_model.")
        }
        result.update(ema_weights)
        return result

    def load_train_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Loads a training/export state dict, initializing the EMA shadow if enabled.

        Exported and converted checkpoints never contain ``ema_model.`` keys (EMA is not
        surfaced; see ``get_export_state_dict``), so when EMA is enabled the shadow is
        seeded from the loaded live-model weights rather than from the state dict. This
        is the metric-fine-tune-from-a-relative-checkpoint path: the relative checkpoint
        has no EMA keys, so the metric run's EMA starts from the loaded relative weights.
        Loading is delegated to the base method with the ``ema_model`` submodule
        temporarily detached so its (absent) keys are not reported as missing.
        """
        if self.ema_model is None:
            return super().load_train_state_dict(
                state_dict, strict=strict, assign=assign
            )

        # Detach the EMA submodule so its parameters are not expected in `state_dict`.
        ema_model = self.ema_model
        self.ema_model = None
        try:
            incompatible = super().load_train_state_dict(
                state_dict, strict=strict, assign=assign
            )
        finally:
            self.ema_model = ema_model
        # Re-seed the EMA shadow from the freshly loaded live weights.
        self.ema_model.model.load_state_dict(self.model.state_dict())
        return incompatible

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
        # The feature-alignment projections (when enabled) are also randomly initialized,
        # so they join the decoder group at the full lr; the frozen teacher has no
        # trainable parameters and is not part of any group.
        backbone_params = set(self.model.backbone.parameters())
        decoder_params = set(self.model.decoder.parameters())
        if self.feature_align_criterion is not None:
            decoder_params |= set(self.feature_align_criterion.parameters())
        param_groups: list[dict[str, Any]] = []
        for name, module_params, group_lr in (
            ("backbone", backbone_params, backbone_lr),
            ("decoder", decoder_params, lr),
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
        # The teacher is frozen for feature distillation; ``self.train()`` above flips it
        # back to train mode, so restore eval to keep its features deterministic.
        if self.teacher is not None:
            self.teacher.eval()

    def clip_gradients(self, fabric: Fabric, optimizer: Optimizer) -> None:
        if no_auto(self.model_args.gradient_clip_val) > 0:
            fabric.clip_gradients(
                module=self,
                optimizer=optimizer,
                max_norm=no_auto(self.model_args.gradient_clip_val),
                error_if_nonfinite=False,
            )

    def _forward(
        self, model: DepthAnythingDepthEstimation, images: Tensor
    ) -> dict[str, Tensor]:
        feats = model._extract_features(images)
        out: dict[str, Tensor] = model.decoder(
            feats=feats, H=images.shape[-2], W=images.shape[-1]
        )
        return out

    def _step(
        self,
        model: DepthAnythingDepthEstimation,
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

        # Extract the student's DPT-input features once and reuse them for the decoder
        # output and (when enabled) the feature-alignment loss, avoiding a second
        # backbone forward pass.
        student_feats = model._extract_features(images)
        out: dict[str, Tensor] = model.decoder(
            feats=student_feats, H=images.shape[-2], W=images.shape[-1]
        )
        # Sky has no valid depth: the teacher's depth in sky regions is garbage, so
        # exclude it from the depth losses. The sky head still trains on the full sky
        # map below.
        depth_mask = (depth > 0) & (sky < 0.5)
        silog_loss = self.silog_criterion(out["depth"], depth, depth_mask)
        grad_loss = self.grad_criterion(out["depth"], depth, depth_mask)
        sky_loss = self.sky_criterion(out["sky"], sky)
        loss = (
            silog_loss
            + self.model_args.grad_loss_weight * grad_loss
            + self.model_args.sky_loss_weight * sky_loss
        )
        # Relative-L1 (AbsRel) term. Metric-depth only: the raw scale-aware AbsRel pins the
        # student's absolute scale to the teacher (the scale-invariant SILog and gradient
        # terms give no such signal). It is not applied to relative-depth models, whose
        # global scale is intentionally free; there it stays a logged constant 0 so the loss
        # dict shape matches the metric collection.
        if self._is_metric:
            abs_l1_loss = self.abs_l1_criterion(out["depth"], depth, depth_mask)
            loss = loss + self.model_args.abs_l1_loss_weight * abs_l1_loss
        else:
            abs_l1_loss = loss.new_zeros(())

        loss_log = {
            "silog_loss": silog_loss.detach(),
            "grad_loss": grad_loss.detach(),
            "sky_loss": sky_loss.detach(),
            "abs_l1_loss": abs_l1_loss.detach(),
        }
        # Feature-space distillation of the teacher's DPT-input patch tokens. The teacher
        # shares the student's patch size and input resolution, so both backbones produce
        # the same token grid and the features align token-for-token (only the widths
        # differ, which the loss projects away). Enabled only when a positive weight has
        # loaded the teacher.
        if self.feature_align_criterion is not None and self.teacher is not None:
            with torch.no_grad():
                teacher_feats = self.teacher._extract_features(images)
            feat_loss = self.feature_align_criterion(student_feats, teacher_feats)
            loss = loss + self.feature_align_weight * feat_loss
            loss_log[_FEATURE_LOSS_NAME] = feat_loss.detach()

        metrics.update_with_losses(
            {"loss": loss.detach(), **loss_log},
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
    """Loads pretrained backbone weights into the student backbone in place.

    A local ``backbone_weights`` checkpoint takes precedence. Otherwise the public
    pretrained weights for the student's backbone are fetched and copied in. For backbones
    without hosted weights (e.g. the ``_vittest14`` test backbone) this is a no-op and
    the backbone keeps its random initialization.
    """
    if backbone_weights is not None:
        state_dict = torch.load(backbone_weights, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(state_dict, strict=True)
        return

    reference = model.backbone_package.get_model(
        model_name=model.backbone_name,
        model_args=model.backbone_model_args,
        load_weights=True,
    )
    model.backbone.load_state_dict(reference.state_dict(), strict=True)


def _build_teacher(
    *, model_name: str, weights: PathLike | None
) -> DepthAnythingDepthEstimation:
    """Builds the frozen teacher used for feature-space distillation.

    The full converted depth checkpoint is loaded so the teacher backbone matches the one
    that produced the depth and sky pseudo-labels. Only the backbone is needed for feature
    alignment, so the DPT head is dropped to save memory; features are read at the
    teacher's own ``out_layers`` via ``_extract_features``. The teacher is frozen and set
    to eval mode.
    """
    teacher = DepthAnythingDepthEstimation(
        model_name=model_name,
        load_weights=True,
        weights=weights,
    )
    # The DPT head is unused for feature alignment; drop it to save memory. Replaced with
    # an Identity so any accidental attribute access fails loudly rather than silently.
    teacher.decoder = torch.nn.Identity()  # type: ignore[assignment]
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher
