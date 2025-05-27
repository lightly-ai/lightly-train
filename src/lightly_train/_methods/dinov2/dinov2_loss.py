#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#

# References:
#   - https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/dino_clstoken_loss.py
#   - https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py
#
# Modifications Copyright (c) Lightly AG and affiliates:
#   - Import xFormers' cross entropy only if XFORMERS_ENABLED is True
#   - Use dist.is_initialized() to control the all_reduce operation of B in distributed setting
#     in the IBOTPatchLoss' sinkhorn_knopp_teacher
#   - Rename iBOTPatchLoss to IBOTPatchLoss
#   - Add type hints to the functions
#   - Remove dead code
#   - Add TODO for investigating the casting of self.center in IBOTPatchLoss

from __future__ import annotations

import os
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import cross_entropy  # type: ignore[import]

        def lossfunc(t: Tensor, s: Tensor, temp: float):  # type: ignore[no-untyped-def]
            s = s.float()
            t = t.float()
            if s.ndim == 2:
                return -cross_entropy(
                    s.unsqueeze(0), t.unsqueeze(0), temp, bw_inplace=True
                ).squeeze(0)
            elif s.ndim == 3:
                return -cross_entropy(s, t, temp, bw_inplace=True)
            else:
                raise ValueError(
                    f"Invalid tensor shape: {s.shape}. Expected 2D or 3D tensor."
                )

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:

    def lossfunc(t: Tensor, s: Tensor, temp: float):  # type: ignore[no-untyped-def]
        return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)

    XFORMERS_AVAILABLE = False


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.center: torch.Tensor  # Type hint for self.center
        self.updated = True
        self.reduce_handle = None

    @torch.no_grad()
    def softmax_center_teacher(
        self, teacher_output: Tensor, teacher_temp: float
    ) -> Tensor:
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self, teacher_output: Tensor, teacher_temp: float, n_iterations: int = 3
    ) -> Tensor:
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(
        self,
        student_output_list: Tuple[Tensor, ...] | List[Tensor],
        teacher_out_softmaxed_centered_list: Tensor | List[Tensor],
    ) -> Tensor:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        total_loss: Tensor = torch.tensor(0.0, device=student_output_list[0].device)
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output: Tensor) -> None:
        self.updated = False
        self.len_teacher_output = teacher_output.shape[0]
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self) -> None:
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (
                1 - self.center_momentum
            )

            self.updated = True


class IBOTPatchLoss(nn.Module):
    def __init__(
        self,
        patch_out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.center: torch.Tensor  # Type hint for self.center
        self.updated = True
        self.reduce_handle = None

    @torch.no_grad()
    def softmax_center_teacher(
        self, teacher_patch_tokens: Tensor, teacher_temp: float
    ) -> Tensor:
        self.apply_center_update()

        # TODO: self.center uses float32 which might cause unnecessary upcasting in fp16 settings which could slow down training
        # we need to investigate how we should handle the casting in this case
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self,
        teacher_output: Tensor,
        teacher_temp: float,
        n_masked_patches_tensor: Tensor,
        n_iterations: int = 3,
    ) -> Tensor:
        teacher_output = teacher_output.float()
        Q = torch.exp(
            teacher_output / teacher_temp
        ).t()  # Q is K-by-B for consistency with notations from our paper
        B = n_masked_patches_tensor
        if dist.is_initialized():
            dist.all_reduce(B)
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(
        self,
        student_patch_tokens: Tensor,
        teacher_patch_tokens: Tensor,
        student_masks_flat: Tensor,
    ) -> Tensor:
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = torch.sum(
            loss * student_masks_flat.float(), dim=-1
        ) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked: Tensor,
        teacher_patch_tokens_masked: Tensor,
        student_masks_flat: Tensor,
        n_masked_patches: int | None = None,
        masks_weight: Tensor | None = None,
    ) -> Tensor:
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        loss: Tensor = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight

        B: int = student_masks_flat.shape[0]
        return -loss.sum() / B

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens: Tensor) -> None:
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens: Tensor) -> None:
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(
            teacher_patch_tokens.mean(1), dim=0, keepdim=True
        )
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self) -> None:
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (
                1 - self.center_momentum
            )

            self.updated = True
