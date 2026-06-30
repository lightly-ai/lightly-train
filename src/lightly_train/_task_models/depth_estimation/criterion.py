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

- ``GlobalLocalLoss`` is the paper's depth term ``L_gl``, the global–local loss adopted
  from MoGe (arXiv:2410.19115). MoGe defines it on 3D point clouds with importance-
  sampled spherical regions; DA3 applies it to depth directly. Our pseudo-labels store
  depth (and sky) only — no point maps or intrinsics — so we implement the depth-only
  form: a closed-form ROE (robust, L1-optimal) affine alignment of the prediction to the
  target, scored as a depth-weighted L1 residual, summed over a global region and over
  disjoint grid windows at MoGe's three local scales (levels 4/16/64).
- ``GradientMatchingLoss`` is the paper's ``L_grad`` (combined with weight ``α = 0.5``).
- ``SkyDistillLoss`` is the paper's ``L_sky``. The paper supervises the sky mask with
  MSE; we use BCE on the sigmoid sky head because the target is a probability map.
- ``SILogLoss`` is kept available but is *not* part of the DA3 objective; it is wired
  into the trainer with a default weight of ``0.0`` (see ``DepthEstimationTrainArgs``).

The paper's ``L_N`` (distance-weighted surface-normal loss) and ``L_obj`` (object-mask
loss) are intentionally omitted: ``L_N`` requires camera intrinsics and ``L_obj``
requires object-mask labels, neither of which is available in the distillation
pseudo-labels (only depth and sky maps are stored on disk).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class GlobalLocalLoss(Module):
    """Global–local affine-invariant depth loss (DA3 ``L_gl``, adopted from MoGe).

    The prediction is aligned to the target by a robust, L1-optimal affine transform
    ``z ≈ s·ẑ + t`` (the ROE solve, see ``_roe_affine_align``) and the loss is the
    depth-weighted, truncated L1 residual of that alignment. This is computed once over
    the whole valid image (the *global* term) and once per window over disjoint
    axis-aligned grid windows at several scales (the *local* terms). Aligning each local
    window independently rewards locally-correct geometry even when the global scale is
    off, which sharpens fine structure relative to a single global fit.

    MoGe defines this on 3D point clouds with importance-sampled spherical regions and a
    3D solver; with depth-only pseudo-labels (no point maps or intrinsics) the regions
    reduce to 2D image windows and the alignment to a 1D affine fit on depth.
    """

    def __init__(
        self,
        local_levels: tuple[int, ...] = (4, 16, 64),
        trunc: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        """
        Args:
            local_levels: Image-grid scales for the local terms. A level ``l`` tiles the
                image into windows of side ``ceil(min(H, W) / l)``, matching MoGe's three
                scales (1/4, 1/16, 1/64 of the image). The global term is always included
                in addition to these.
            trunc: Upper truncation applied to each (depth-normalized) absolute residual
                before averaging, for robustness to outliers (MoGe's ``min(·, τ)``).
            eps: Small constant guarding the depth weighting ``1 / z``.
        """
        super().__init__()
        self.local_levels = local_levels
        self.trunc = trunc
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted depth of shape ``(B, 1, H, W)``, strictly positive.
            target: Target depth of shape ``(B, 1, H, W)``.
            mask: Boolean validity mask of shape ``(B, 1, H, W)``; the loss is computed
                over ``True`` pixels only.

        Returns:
            Scalar loss: the global term plus the sum of the per-level local terms,
            following MoGe's additive ``L_global + Σ_scales L_local`` structure (each
            term is itself a per-pixel mean residual, so the scales are comparable).
            Returns a graph-connected zero if no pixel is valid.
        """
        valid = mask & torch.isfinite(target) & torch.isfinite(pred)
        if not bool(valid.any()):
            return pred.sum() * 0.0

        b, _, h, w = pred.shape
        # Global term: one region per image spanning every pixel.
        terms = [
            _affine_invariant_term(
                pred=pred.reshape(b, h * w),
                target=target.reshape(b, h * w),
                valid=valid.reshape(b, h * w),
                eps=self.eps,
                trunc=self.trunc,
            )
        ]
        # Local terms: one region per disjoint grid window, per scale. Each region holds
        # the ``window * window`` pixels of one window (invalid ones masked out).
        for level in self.local_levels:
            window = max(1, math.ceil(min(h, w) / level))
            if window >= max(h, w):
                # The window covers the whole image, duplicating the global term.
                continue
            regions_pred, regions_target, regions_valid = _windows(
                pred=pred, target=target, valid=valid, window=window
            )
            terms.append(
                _affine_invariant_term(
                    pred=regions_pred,
                    target=regions_target,
                    valid=regions_valid,
                    eps=self.eps,
                    trunc=self.trunc,
                )
            )
        return torch.stack(terms).sum()


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
        pred = pred_sky.clamp(self.eps, 1.0 - self.eps)
        target = target_sky.clamp(0.0, 1.0)
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


def _affine_invariant_term(
    *, pred: Tensor, target: Tensor, valid: Tensor, eps: float, trunc: float
) -> Tensor:
    """Returns the affine-invariant residual averaged over a set of regions.

    Each region (a row of the ``(M, K)`` inputs) is aligned and scored independently
    over its valid pixels; the per-region scores are averaged. Regions with fewer than
    two valid pixels are degenerate (the affine fit is underdetermined) and are excluded
    from the average.

    Args:
        pred: Predicted depth of shape ``(M, K)`` (``M`` regions, ``K`` pixels each).
        target: Target depth of shape ``(M, K)``.
        valid: Boolean validity mask of shape ``(M, K)``.
        eps: Small constant guarding the ``1 / target`` weighting.
        trunc: Upper truncation on each depth-normalized absolute residual.

    Returns:
        Scalar mean residual over the scorable regions, or a graph-connected zero if no
        region is scorable.
    """
    # Depth-weighted (1 / z) L1, as in MoGe: nearer pixels matter more. Invalid pixels
    # are zero-weighted so they affect neither the alignment nor the residual.
    weight = valid.to(pred.dtype) / (target + eps)
    scale, shift = _roe_affine_align(pred=pred, target=target, weight=weight)
    residual = torch.abs(scale[:, None] * pred + shift[:, None] - target) * weight
    region_weight = weight.sum(dim=1)
    region_residual = (residual.clamp_max(trunc)).sum(dim=1) / region_weight.clamp_min(
        eps
    )

    scorable = valid.sum(dim=1) >= 2
    if not bool(scorable.any()):
        return pred.sum() * 0.0
    return region_residual[scorable].mean()


def _windows(
    *, pred: Tensor, target: Tensor, valid: Tensor, window: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Tiles each ``(B, 1, H, W)`` map into disjoint ``window × window`` regions.

    Returns ``(pred, target, valid)`` reshaped to ``(B * num_windows, window * window)``,
    where the windows tile the image left-to-right, top-to-bottom. The last row/column of
    windows is zero-padded when ``H``/``W`` is not a multiple of ``window``; padded pixels
    are marked invalid so they are ignored by the alignment and residual.
    """
    k = window * window

    def unfold(x: Tensor) -> Tensor:
        # (B, k, num_windows) -> (B, num_windows, k) -> (B * num_windows, k).
        patches = F.unfold(x, kernel_size=window, stride=window)
        return patches.transpose(1, 2).reshape(-1, k)

    # Pad on the bottom/right so H and W are multiples of `window`; the padding is marked
    # invalid via the validity channel below.
    pad_h = (window - pred.shape[-2] % window) % window
    pad_w = (window - pred.shape[-1] % window) % window
    pad = (0, pad_w, 0, pad_h)
    pred_p = F.pad(pred, pad)
    target_p = F.pad(target, pad)
    valid_p = F.pad(valid.to(pred.dtype), pad)

    regions_pred = unfold(pred_p)
    regions_target = unfold(target_p)
    regions_valid = unfold(valid_p) > 0.5
    return regions_pred, regions_target, regions_valid


def _roe_affine_align(
    *, pred: Tensor, target: Tensor, weight: Tensor
) -> tuple[Tensor, Tensor]:
    """Solves ``(s, t) = argmin Σ w_i·|s·pred_i + t − target_i|`` per region (the ROE
    alignment), batched over the rows of the ``(M, K)`` inputs.

    Uses MoGe's observation that the L1 optimum passes exactly through one data point,
    so substituting ``t = target_k − s·pred_k`` for an anchor ``k`` turns the objective
    into a function of ``s`` alone whose minimizer is the weighted median of the pairwise
    slopes ``(target_i − target_k) / (pred_i − pred_k)`` (weighted by ``w_i·|pred_i −
    pred_k|``). The anchor is the weighted-median pixel of ``pred``, which the L1-optimal
    line is guaranteed to be near; with the anchor fixed this is a single closed-form
    weighted median and avoids the full O(K²) search over all anchors.

    Zero-weighted (invalid) pixels never contribute to any weighted median or slope, so
    each region is fit only on its valid pixels. The solve is detached (``no_grad``): the
    returned scale and shift are treated as constants so the alignment does not
    back-propagate, matching MoGe.

    Args:
        pred: Predicted depth of shape ``(M, K)``.
        target: Target depth of shape ``(M, K)``.
        weight: Non-negative per-pixel weight of shape ``(M, K)`` (zero for invalid
            pixels).

    Returns:
        ``(scale, shift)``, each of shape ``(M,)``.
    """
    with torch.no_grad():
        # Per-region anchor: the weighted-median pixel of the prediction.
        anchor = _weighted_median(values=pred, weight=weight)
        denom = pred - anchor[:, None]
        anchor_target = _weighted_median(values=target, weight=weight)
        # Slope of the line from the anchor to each pixel; pixels at the anchor (or
        # invalid) are zero-weighted so their undefined slope is harmless.
        offset = torch.abs(denom) > 0
        safe_denom = torch.where(offset, denom, torch.ones_like(denom))
        slopes = (target - anchor_target[:, None]) / safe_denom
        slope_weight = weight * torch.abs(denom) * offset.to(weight.dtype)
        scale = _weighted_median(values=slopes, weight=slope_weight)
        # A region whose valid predictions are all equal carries no slope information;
        # the prediction then explains nothing, so fall back to scale 0 (shift absorbs
        # the depth) rather than the meaningless median of an all-zero-weight row.
        scale = torch.where(slope_weight.sum(dim=1) > 0, scale, torch.zeros_like(scale))
        # Given the scale, the L1-optimal shift is the weighted median of the residuals.
        shift = _weighted_median(values=target - scale[:, None] * pred, weight=weight)
        return scale, shift


def _weighted_median(*, values: Tensor, weight: Tensor) -> Tensor:
    """Returns the per-row weighted median of ``values`` with non-negative ``weight``.

    The weighted median of a row is the value at which the cumulative weight first
    reaches half of the row's total; it is the minimizer of ``Σ w_i·|m − values_i|``.
    Rows whose weight sums to zero return their first element (the caller treats such
    regions as non-scorable).

    Args:
        values: Values of shape ``(M, K)``.
        weight: Non-negative weights of shape ``(M, K)``.

    Returns:
        Weighted medians of shape ``(M,)``.
    """
    order = torch.argsort(values, dim=1)
    sorted_values = torch.gather(values, dim=1, index=order)
    sorted_weight = torch.gather(weight, dim=1, index=order)
    cumulative = torch.cumsum(sorted_weight, dim=1)
    half = 0.5 * cumulative[:, -1:]
    # First index per row whose cumulative weight reaches the half-total.
    idx = torch.searchsorted(cumulative, half).squeeze(1)
    idx = idx.clamp_max(values.shape[1] - 1)
    return torch.gather(sorted_values, dim=1, index=idx[:, None]).squeeze(1)
