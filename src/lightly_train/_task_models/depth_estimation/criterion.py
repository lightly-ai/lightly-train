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
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


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
