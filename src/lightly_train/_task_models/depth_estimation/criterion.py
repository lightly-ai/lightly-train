#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Distillation losses for the DepthAnything V3 student.

This module implements an intentional *subset* of the DepthAnything V3 training
objective (arXiv:2511.10647, Eq. 7: ``L_T = α·L_grad + L_gl + L_N + L_sky + L_obj``)
tailored to distilling a ViT-L teacher into a ViT-S student. The mapping is:

- ``SILogLoss`` is the depth term. The paper uses the MoGe global–local loss ``L_gl``
  here; for distilling a single clean teacher a scale-invariant log loss is the standard
  and well-behaved choice, so we substitute it.
- ``GradientMatchingLoss`` is the paper's ``L_grad`` (combined with weight ``α = 0.5``).
- ``SkyDistillLoss`` is the paper's ``L_sky``. The paper supervises the sky mask with
  MSE; we use BCE on the sigmoid sky head because the target is a probability map.

The paper's ``L_N`` (distance-weighted surface-normal loss) and ``L_obj`` (object-mask
loss) are intentionally omitted: ``L_N`` requires camera intrinsics and ``L_obj``
requires object-mask labels, neither of which is available in the distillation
pseudo-labels (only depth and sky maps are stored on disk).

In addition to these output-space terms, ``FeatureAlignmentLoss`` distills the teacher's
intermediate backbone features (the four patch-token tensors fed to the DPT head) into
the student. This is a feature-space term that requires an online teacher and is
therefore not derivable from the on-disk pseudo-labels.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ModuleList


class SILogLoss(Module):
    """Scale-invariant log loss (Eigen et al., as used by MiDaS/DepthAnything).

    Operates on valid pixels only and is invariant to a global scale of the prediction,
    which makes it well suited for relative-depth distillation where the teacher and
    student may differ by a global factor.
    """

    def __init__(self, lambd: float = 0.5, eps: float = 1e-6) -> None:
        """
        Args:
            lambd: Weight of the (squared) mean term that removes the global scale.
                ``lambd=0`` reduces to the log-space MSE; ``lambd=1`` is fully
                scale-invariant.
            eps: Small constant added before the logarithm for numerical stability.
        """
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted depth of shape ``(B, 1, H, W)``, strictly positive.
            target: Target depth of shape ``(B, 1, H, W)``.
            mask: Boolean validity mask of shape ``(B, 1, H, W)``; loss is computed over
                ``True`` pixels only.

        Returns:
            Scalar loss. If no pixel is valid, returns a graph-connected zero so the
            backward pass stays well-defined.
        """
        valid = mask & torch.isfinite(target) & torch.isfinite(pred)
        if not bool(valid.any()):
            return pred.sum() * 0.0

        diff = torch.log(pred[valid] + self.eps) - torch.log(target[valid] + self.eps)
        loss = torch.mean(diff**2) - self.lambd * (torch.mean(diff) ** 2)
        # The scale-invariant term can produce a tiny negative value from floating point
        # error when the prediction matches the target exactly; clamp before the sqrt.
        return torch.sqrt(loss.clamp_min(0.0) + self.eps)


class GradientMatchingLoss(Module):
    """Multi-scale gradient-matching loss (MiDaS/DepthAnything).

    For each scale, the absolute spatial gradients of the log-depth difference are
    averaged over pixels where both neighbours are valid, then the difference and mask
    are downsampled and the process repeats. This sharpens depth discontinuities and
    discourages over-smooth predictions when distilling a higher-capacity teacher.
    """

    def __init__(self, scales: int = 4, eps: float = 1e-6) -> None:
        """
        Args:
            scales: Number of (halving) scales over which the gradients are matched.
            eps: Small constant added before the logarithm for numerical stability.
        """
        super().__init__()
        self.scales = scales
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted depth of shape ``(B, 1, H, W)``, strictly positive.
            target: Target depth of shape ``(B, 1, H, W)``.
            mask: Boolean validity mask of shape ``(B, 1, H, W)``.

        Returns:
            Scalar loss summed over all scales. Returns a graph-connected zero if no
            scale contributes any valid gradient.
        """
        valid = mask & torch.isfinite(target) & torch.isfinite(pred)
        diff = torch.log(pred + self.eps) - torch.log(target + self.eps)
        # Zero out invalid pixels so they do not leak into the finite differences; the
        # per-edge validity mask below ensures they are not counted either.
        diff = torch.where(valid, diff, torch.zeros_like(diff))
        mask_f = valid.to(diff.dtype)

        total = pred.sum() * 0.0
        for _ in range(self.scales):
            if diff.shape[-1] < 2 or diff.shape[-2] < 2:
                break
            total = total + _gradient_term(diff=diff, mask=mask_f)
            # Average-pool to the next coarser scale.
            diff = F.avg_pool2d(diff, kernel_size=2)
            mask_f = F.avg_pool2d(mask_f, kernel_size=2)
        return total


class SkyDistillLoss(Module):
    """Binary cross-entropy between the student and teacher sky confidence maps.

    Both inputs are expected to be probabilities in ``[0, 1]`` (the trainable depth
    configs use a sigmoid sky head); the teacher map is used as a soft BCE target.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred_sky: Tensor, target_sky: Tensor) -> Tensor:
        """
        Args:
            pred_sky: Student sky confidence of shape ``(B, 1, H, W)`` in ``[0, 1]``.
            target_sky: Teacher sky confidence of shape ``(B, 1, H, W)`` in ``[0, 1]``.

        Returns:
            Scalar BCE loss.
        """
        with torch.autocast(device_type=pred_sky.device.type, enabled=False):
            pred = pred_sky.float().clamp(self.eps, 1.0 - self.eps)
            target = target_sky.float().clamp(0.0, 1.0)
            return F.binary_cross_entropy(pred, target)


class RelativeL1Loss(Module):
    """Scale-aware relative-L1 (AbsRel) loss on valid pixels.

    Computes the mean of ``|pred - target| / target`` over valid pixels. Unlike the
    scale-invariant ``SILogLoss`` (``lambd > 0``) and the neighbour-difference-only
    ``GradientMatchingLoss``, this term's gradient with respect to a global scale error
    does not vanish, so it directly pins the absolute scale of a metric-depth student to
    the teacher. It is the training-time counterpart of the unaligned AbsRel validation
    metric. Intended only for metric-depth distillation, where the student's output and
    the teacher target live in the same (canonical-camera) depth space.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """
        Args:
            eps: Small constant added to the target denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted depth of shape ``(B, 1, H, W)``, strictly positive.
            target: Target depth of shape ``(B, 1, H, W)``.
            mask: Boolean validity mask of shape ``(B, 1, H, W)``; loss is computed over
                ``True`` pixels only.

        Returns:
            Scalar loss. If no pixel is valid, returns a graph-connected zero so the
            backward pass stays well-defined.
        """
        valid = mask & torch.isfinite(target) & torch.isfinite(pred)
        if not bool(valid.any()):
            return pred.sum() * 0.0

        return torch.mean(
            torch.abs(pred[valid] - target[valid]) / (target[valid] + self.eps)
        )


class FeatureAlignmentLoss(Module):
    """Feature-space distillation of the teacher's DPT-input patch tokens.

    The student and teacher backbones expose the same number of intermediate patch-token
    tensors that are fed to their DPT heads. They share the patch size and processing
    resolution (e.g. a ViT-L teacher and a ViT-S student both at 14px/504px), so the
    token grids match and features align token-for-token. Only the backbone widths differ
    (e.g. a ``teacher_dim``-wide ViT-L teacher and a ``student_dim``-wide ViT-S student),
    so each student tensor is projected to the teacher width by a learnable per-stage
    ``Linear`` before matching.

    The loss is the mean cosine distance ``1 - cos(student, teacher)`` per token, so the
    student is pulled towards the *direction* of the teacher features and is not required
    to reproduce their absolute magnitude. Cosine distance is width-independent and lies
    in ``[0, 2]`` (``~1.0`` for uncorrelated features), so it stays on the same scale as
    the other depth losses regardless of the teacher width, keeping
    ``feature_align_weight`` interpretable. The projections are training-only and are
    discarded when the student is exported for inference.
    """

    def __init__(
        self,
        *,
        student_dim: int,
        teacher_dim: int,
        num_stages: int = 4,
        eps: float = 1e-6,
    ) -> None:
        """
        Args:
            student_dim: Channel dimension of the student patch tokens.
            teacher_dim: Channel dimension of the teacher patch tokens.
            num_stages: Number of intermediate feature tensors to align (one projection
                per stage).
            eps: Small constant added to the norm for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.projections = ModuleList(
            [Linear(student_dim, teacher_dim) for _ in range(num_stages)]
        )

    def forward(
        self,
        student_feats: Sequence[Tensor],
        teacher_feats: Sequence[Tensor],
    ) -> Tensor:
        """
        Args:
            student_feats: The student's intermediate patch-token tensors, each of shape
                ``(B, N, student_dim)``.
            teacher_feats: The teacher's intermediate patch-token tensors, each of shape
                ``(B, N, teacher_dim)``, detached from the graph by the caller. The token
                count ``N`` matches the student's (shared patch size and resolution).

        Returns:
            Scalar loss averaged over the aligned stages.
        """
        if len(student_feats) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} student feature tensors, got "
                f"{len(student_feats)}."
            )
        if len(teacher_feats) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} teacher feature tensors, got "
                f"{len(teacher_feats)}."
            )

        total = student_feats[0].sum() * 0.0
        for projection, student, teacher in zip(
            self.projections, student_feats, teacher_feats
        ):
            # Project the student tokens to the teacher width; the token grids already
            # match, so the features align token-for-token. Mean cosine distance per
            # token keeps the loss in [0, 2] and independent of the teacher width.
            projected = projection(student)
            cos = F.cosine_similarity(projected, teacher, dim=-1, eps=self.eps)
            total = total + (1.0 - cos).mean()
        return total / len(self.projections)


def _gradient_term(diff: Tensor, mask: Tensor) -> Tensor:
    """Returns the mean absolute horizontal and vertical gradient of ``diff``.

    Gradients are only counted across edges where both neighbouring pixels are fully
    valid (``mask == 1`` on both sides).
    """
    grad_x = torch.abs(diff[..., :, 1:] - diff[..., :, :-1])
    mask_x = mask[..., :, 1:] * mask[..., :, :-1]
    grad_y = torch.abs(diff[..., 1:, :] - diff[..., :-1, :])
    mask_y = mask[..., 1:, :] * mask[..., :-1, :]

    num = (grad_x * mask_x).sum() + (grad_y * mask_y).sum()
    den = mask_x.sum() + mask_y.sum()
    if bool(den == 0):
        return diff.sum() * 0.0
    return num / den
