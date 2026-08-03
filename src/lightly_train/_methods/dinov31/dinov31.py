#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""DINOv31: a DINOv2 post-train method with a PaKA dense-relational loss.

- [0]: 2025, PaKA: https://arxiv.org/abs/2509.05606

DINOv31 continues ("post-trains") a DINOv2-pretrained backbone with the full
DINOv2 objective (DINO + iBOT + KoLeo) PLUS an auxiliary Patch Kernel Alignment
(PaKA / CKA) loss that aligns the relational structure of student and teacher
dense patch tokens. It is a thin subclass of :class:`DINOv2` and does not modify
it; the PaKA term is added on top of the DINOv2 loss.

The method expects the DINOv31 transform's view layout::

    [global0, global1, dino_local0..L-1,
     clean_global0, clean_global1, paka_local0..K-1]

The two trailing clean globals (augmentation-free renders of the globals) feed
an aug-free EMA teacher (PaKA only; DINO/iBOT keep the augmented teacher); the K
trailing high-overlap locals feed the PaKA student. Each student local is paired
with its parent global only and ROI-aligned onto the local grid before CKA.

Example::

    # Post-train a DINOv2 ViT-S backbone with PaKA on COCO.
    lightly_train.pretrain(
        method="dinov31",
        model="dinov2/vits14",
        checkpoint="dinov2_vits14.ckpt",  # full DINOv2 checkpoint (incl. optimizer); only model weights are loaded
        method_args={
            "transform_args": {"image_size": [518, 518]},
            "paka_weight": 1.0,
            "paka_num_local": 8,
        },
        optim_args={
            "weight_decay_start": 0.04,         # constant (not the DINOv2 0.04->0.4 ramp)
            "weight_decay_end": 0.04,
            "warmup_steps": 1000,               # short, post-train sized
            "teacher_temp_warmup_steps": 1000,  # ~20% of a short post-train
        },
    )
"""

from __future__ import annotations

import copy
from typing import Any, Literal, Mapping, cast

import torch
from lightly.loss import PatchKernelAlignmentLoss, roi_resample_to_grid
from lightly.utils.scheduler import cosine_schedule
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.module import _IncompatibleKeys

from lightly_train._methods.dinov2.dinov2 import (
    DINOv2,
    DINOv2AdamWViTArgs,
    DINOv2Args,
    freeze_eval_module,
)
from lightly_train._methods.dinov2.dinov2_head import _build_mlp
from lightly_train._methods.dinov31.dinov31_transform import DINOv31Transform
from lightly_train._methods.method import TrainingStepResult
from lightly_train._optim.adamw8bit_args import AdamW8bitArgs
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.trainable_modules import TrainableModules
from lightly_train._torch_helpers import update_momentum
from lightly_train.types import Batch


class DINOv31AdamW8bitArgs(DINOv2AdamWViTArgs, AdamW8bitArgs):
    """8-bit AdamW args for DINOv31 (bitsandbytes).

    Keeps the DINOv2 ViT layerwise-decay defaults (lr/wd/eps/betas) via
    :class:`DINOv2AdamWViTArgs` and the 8-bit optimizer construction via
    :class:`AdamW8bitArgs`. ``DINOv2`` itself is unchanged.
    """


class DINOv31Args(DINOv2Args):
    """Arguments for DINOv31.

    Only the PaKA-specific fields are listed here; optimizer/schedule fields
    (weight decay, warmup, teacher-temperature warmup, ...) are inherited
    unchanged from :class:`DINOv2Args` and set per the recipe in the module
    docstring.
    """

    # PaKA auxiliary loss weight.
    paka_weight: float = 1.0
    # PaKA is skipped until this global step (0 = active for the whole run).
    paka_start_step: int = 0
    # Number of dedicated high-overlap student local crops (must match the
    # transform's paka_num_local).
    paka_num_local: int = 8
    # Per-image token subsample before the O(N^2) CKA kernel.
    paka_max_tokens: int = 512


class DINOv31(DINOv2):
    """DINOv2 post-train method with a PaKA dense-relational auxiliary loss."""

    def __init__(
        self,
        method_args: DINOv31Args,
        optimizer_args: Any,
        embedding_model: Any,
        global_batch_size: int,
        num_input_channels: int,
    ) -> None:
        super().__init__(
            method_args=method_args,
            optimizer_args=optimizer_args,
            embedding_model=embedding_model,
            global_batch_size=global_batch_size,
            num_input_channels=num_input_channels,
        )
        self.paka_loss = PatchKernelAlignmentLoss(
            max_tokens=method_args.paka_max_tokens
        )
        # 3-layer projection head (paper App. D.1) applied to the ROI-aligned
        # patch tokens before CKA. Student trained, teacher frozen + EMA-updated.
        # Reuses the DINOv2 MLP builder (embed->2048->2048->256, GELU). Lives on
        # the Method, so it is excluded from the exported backbone automatically.
        embed_dim = self.teacher_embedding_model.wrapped_model.get_model().embed_dim
        paka_head: Module = _build_mlp(
            nlayers=3,
            in_dim=embed_dim,
            hidden_dim=2048,
            bottleneck_dim=256,
            use_bn=False,
        )
        self.student_paka_head: Module = paka_head
        # Deepcopy so student and teacher start identical and parameter order
        # matches for the EMA update.
        self.teacher_paka_head: Module = copy.deepcopy(paka_head)
        freeze_eval_module(self.teacher_paka_head)

    @staticmethod
    def method_args_cls() -> type[DINOv31Args]:
        return DINOv31Args

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[OptimizerArgs]:
        # Adds the optional 8-bit AdamW optimizer on top of DINOv2's mapping.
        # Everything else (auto / ADAMW / SGD / LARS) delegates to DINOv2
        # unchanged, so DINOv2 stays byte-identical.
        if optim_type == OptimizerType.ADAMW8BIT:
            return DINOv31AdamW8bitArgs
        return DINOv2.optimizer_args_cls(optim_type=optim_type)

    @staticmethod
    def transform_cls() -> type[DINOv31Transform]:
        return DINOv31Transform

    def on_fit_start(self) -> None:
        # Skip DINOv2.on_fit_start: its 125k-step from-scratch recommendation
        # does not apply to a short post-train from a pretrained checkpoint.
        return

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        """Tolerate missing ``student_paka_head.``/``teacher_paka_head.`` keys.

        The PaKA heads are randomly initialised and trained from scratch, so an
        init checkpoint (e.g. a DINOv2 backbone) legitimately lacks them. This
        mirrors the distillation methods' handling of their added teacher module.
        """
        incompatible_keys = cast(
            _IncompatibleKeys,
            super().load_state_dict(state_dict, strict=False, assign=assign),
        )
        missing_keys = [
            k
            for k in incompatible_keys.missing_keys
            if not (
                k.startswith("student_paka_head.") or k.startswith("teacher_paka_head.")
            )
        ]
        if strict and (missing_keys or incompatible_keys.unexpected_keys):
            raise RuntimeError(
                f"Unexpected keys in state_dict: {incompatible_keys.unexpected_keys}\n"
                f"Missing keys in state_dict: {missing_keys}"
            )
        return incompatible_keys

    def trainable_modules(self) -> TrainableModules:
        trainable = super().trainable_modules()
        return TrainableModules(
            modules=[*trainable.modules, self.student_paka_head],
            modules_no_weight_decay=trainable.modules_no_weight_decay,
        )

    def on_train_batch_end(
        self,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(outputs=outputs, batch=batch, batch_idx=batch_idx)
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.method_args.momentum_start,
            end_value=self.method_args.momentum_end,
        )
        update_momentum(self.student_paka_head, self.teacher_paka_head, m=momentum)

    def training_step_impl(self, batch: Batch, batch_idx: int) -> TrainingStepResult:
        # super() runs first and sees only the leading [global0, global1,
        # dino_local..] views, so the DINO/iBOT path is the DINOv2 path unchanged.
        views = batch["views"]
        method_args = cast(DINOv31Args, self.method_args)
        n_paka_local = method_args.paka_num_local
        n_clean_global = 2
        n_dino = len(views) - n_clean_global - n_paka_local
        if n_dino < 2:
            raise ValueError(
                f"DINOv31 expected at least 2 global views before the {n_clean_global} "
                f"clean globals + {n_paka_local} paka locals, but got {len(views)} views."
            )
        dino_batch: Batch = {
            "filename": batch["filename"],
            "views": views[:n_dino],
        }
        result = super().training_step_impl(dino_batch, batch_idx)

        if self.trainer.global_step < method_args.paka_start_step:
            return result

        paka = self._paka_loss(
            views=views,
            geometries=batch["geometries"],
            n_dino=n_dino,
            n_clean_global=n_clean_global,
            n_paka_local=n_paka_local,
        )
        log_dict = dict(result.log_dict) if result.log_dict else {}
        log_dict["train_loss/paka_loss"] = paka.detach()
        return TrainingStepResult(
            loss=result.loss + method_args.paka_weight * paka,
            log_dict=log_dict,
        )

    # ------------------------------------------------------------------ PaKA

    def _paka_loss(
        self,
        views: list[Tensor],
        geometries: list[Tensor],
        n_dino: int,
        n_clean_global: int,
        n_paka_local: int,
    ) -> Tensor:
        """Cross-view PaKA: clean-teacher globals vs high-overlap student locals.

        Each student local is ROI-aligned with its parent global (parent =
        ``local_idx % n_global``) over their shared region, resampled to the
        local grid; all pairs are pooled into one batch and the PaKA loss
        averages over the pairs that actually overlap.
        """
        clean_views = views[n_dino : n_dino + n_clean_global]
        paka_views = views[n_dino + n_clean_global :]
        clean_global_views = torch.cat(clean_views)  # [2B, C, H, W]
        paka_local_views = torch.cat(paka_views)  # [KB, C, h, w]

        # Raw backbone patch tokens (no head yet: the head is applied AFTER the
        # ROI alignment because of its nonlinear GELU).
        teacher_globals = self._forward_teacher_clean(clean_global_views)  # [2B, Ng, C]
        student_locals = self._forward_student_paka_locals(
            paka_local_views
        )  # [KB, Nl, C]

        global_h = clean_global_views.shape[2] // self._patch_size
        global_w = clean_global_views.shape[3] // self._patch_size
        local_h = paka_local_views.shape[2] // self._patch_size
        local_w = paka_local_views.shape[3] // self._patch_size

        device = student_locals.device
        dtype = student_locals.dtype
        n_global = 2
        teacher_per_global = teacher_globals.chunk(n_global)  # n_global x [B, Ng, C]
        student_per_local = student_locals.chunk(n_paka_local)  # K x [B, Nl, C]
        # The clean globals cover the same region as the augmented globals, so
        # their geometry is geometries[0..1].
        global_geoms = [
            geometries[g].to(device=device, dtype=dtype) for g in range(n_global)
        ]
        local_geoms = [
            geometries[n_dino + n_clean_global + k].to(device=device, dtype=dtype)
            for k in range(n_paka_local)
        ]

        students: list[Tensor] = []
        teachers: list[Tensor] = []
        valids: list[Tensor] = []
        for local_idx in range(n_paka_local):
            global_idx = local_idx % n_global  # parent-only pairing
            s_al, t_al, valid = self._align_cross_view_pair(
                student_tokens=student_per_local[local_idx],
                student_geom=local_geoms[local_idx],
                student_h=local_h,
                student_w=local_w,
                teacher_tokens=teacher_per_global[global_idx],
                teacher_geom=global_geoms[global_idx],
                teacher_h=global_h,
                teacher_w=global_w,
                out_h=local_h,
                out_w=local_w,
            )
            students.append(s_al)
            teachers.append(t_al)
            valids.append(valid)

        student_all = torch.cat(students, dim=0)
        teacher_all = torch.cat(teachers, dim=0)
        valid_all = torch.cat(valids, dim=0)
        student_all = self.student_paka_head(student_all)
        teacher_all = self.teacher_paka_head(teacher_all)
        paka: Tensor = self.paka_loss(
            student_features=student_all,
            teacher_features=teacher_all,
            mask=~valid_all,
        )
        return paka

    def _align_cross_view_pair(
        self,
        student_tokens: Tensor,
        student_geom: Tensor,
        student_h: int,
        student_w: int,
        teacher_tokens: Tensor,
        teacher_geom: Tensor,
        teacher_h: int,
        teacher_w: int,
        out_h: int,
        out_w: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """ROI-aligns a student/teacher crop pair onto their shared-region grid.

        Returns ``(student_aligned, teacher_aligned, valid)`` of shape
        ``(B, out_h*out_w, C)`` / ``(B, out_h*out_w)``. ``valid`` is False for
        whole images whose two crops do not overlap.
        """
        # Shared region (intersection of the two crops) in original image pixels.
        ix0 = torch.maximum(student_geom[:, 0], teacher_geom[:, 0])
        iy0 = torch.maximum(student_geom[:, 1], teacher_geom[:, 1])
        ix1 = torch.minimum(student_geom[:, 2], teacher_geom[:, 2])
        iy1 = torch.minimum(student_geom[:, 3], teacher_geom[:, 3])
        # Require the overlap to span at least ~one output cell of the student
        # crop in each dimension; tinier overlaps carry no relational signal.
        min_w = (student_geom[:, 2] - student_geom[:, 0]) / max(out_w, 1)
        min_h = (student_geom[:, 3] - student_geom[:, 1]) / max(out_h, 1)
        has_overlap = (ix1 - ix0 >= min_w) & (iy1 - iy0 >= min_h)  # (B,)
        ix0 = torch.where(has_overlap, ix0, student_geom[:, 0])
        iy0 = torch.where(has_overlap, iy0, student_geom[:, 1])
        ix1 = torch.where(has_overlap, ix1, student_geom[:, 2])
        iy1 = torch.where(has_overlap, iy1, student_geom[:, 3])

        student_aligned = self._roi_align_view(
            student_tokens,
            student_geom,
            ix0,
            iy0,
            ix1,
            iy1,
            in_h=student_h,
            in_w=student_w,
            out_h=out_h,
            out_w=out_w,
        )
        teacher_aligned = self._roi_align_view(
            teacher_tokens,
            teacher_geom,
            ix0,
            iy0,
            ix1,
            iy1,
            in_h=teacher_h,
            in_w=teacher_w,
            out_h=out_h,
            out_w=out_w,
        )
        valid = has_overlap[:, None].expand(-1, out_h * out_w)
        return student_aligned, teacher_aligned, valid

    def _roi_align_view(
        self,
        tokens: Tensor,
        geom: Tensor,
        ix0: Tensor,
        iy0: Tensor,
        ix1: Tensor,
        iy1: Tensor,
        in_h: int,
        in_w: int,
        out_h: int,
        out_w: int,
    ) -> Tensor:
        """Resamples one crop's tokens over the shared region onto a common grid.

        The feature map is first un-flipped per the recorded h/v flips so grid
        coordinates match the original image orientation, then ROI-aligned.
        """
        batch_size, _, channels = tokens.shape
        feat = tokens.reshape(batch_size, in_h, in_w, channels).permute(0, 3, 1, 2)
        hflip = geom[:, 6] > 0.5
        vflip = geom[:, 7] > 0.5
        feat = torch.where(hflip[:, None, None, None], feat.flip(-1), feat)
        feat = torch.where(vflip[:, None, None, None], feat.flip(-2), feat)
        feat = feat.contiguous()

        crop_x0, crop_y0, crop_x1, crop_y1 = (
            geom[:, 0],
            geom[:, 1],
            geom[:, 2],
            geom[:, 3],
        )
        crop_w = (crop_x1 - crop_x0).clamp(min=1e-6)
        crop_h = (crop_y1 - crop_y0).clamp(min=1e-6)
        gx0 = ((ix0 - crop_x0) / crop_w * in_w).clamp(0.0, float(in_w))
        gx1 = ((ix1 - crop_x0) / crop_w * in_w).clamp(0.0, float(in_w))
        gy0 = ((iy0 - crop_y0) / crop_h * in_h).clamp(0.0, float(in_h))
        gy1 = ((iy1 - crop_y0) / crop_h * in_h).clamp(0.0, float(in_h))
        boxes = torch.stack([gx0, gy0, gx1, gy1], dim=1)  # (B, 4)
        aligned: Tensor = roi_resample_to_grid(feat, boxes, out_h=out_h, out_w=out_w)
        return aligned

    @torch.no_grad()
    def _forward_teacher_clean(self, x: Tensor) -> Tensor:
        """Clean EMA-teacher dense patch tokens ``[G*B, N, C]`` for ``x``.

        An augmentation-free EMA-teacher forward on the clean global crops, used
        only by PaKA (DINO/iBOT keep the augmented teacher). No head / centering:
        PaKA operates on raw backbone tokens.
        """
        tokens = self.teacher_embedding_model.wrapped_model.forward_features(x)
        patch_tokens: Tensor = tokens["features"]  # [G*B, C, H/p, W/p]
        return patch_tokens.flatten(2).permute(0, 2, 1)  # [G*B, N, C]

    def _forward_student_paka_locals(self, x: Tensor) -> Tensor:
        """Student dense patch tokens ``[K*B, N, C]`` for the paka local crops."""
        tokens = self.student_embedding_model.wrapped_model.forward_features(x)
        patch_tokens: Tensor = tokens["features"]  # [K*B, C, h/p, w/p]
        return patch_tokens.flatten(2).permute(0, 2, 1)  # [K*B, N, C]
