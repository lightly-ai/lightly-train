#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from Mask2Former (https://github.com/facebookresearch/Mask2Former)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

# Modifications Copyright 2026 Lightly AG:
# - Added typed interfaces and Google-style docstrings.
# - Added factored pairwise sigmoid CE and dice cost helpers for matching.
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def point_sample(input: Tensor, point_coords: Tensor, mode: str = "bilinear") -> Tensor:
    """Samples ``input`` at normalized point coordinates.

    Args:
        input: Tensor of shape ``(N, C, H, W)``.
        point_coords: Tensor of shape ``(N, P, 2)`` with coordinates in
            ``[0, 1]`` and ``(x, y)`` order.
        mode: Interpolation mode passed to ``grid_sample``. Use ``"nearest"`` for
            binary ground-truth masks to avoid fractional labels at boundaries.

    Returns:
        Tensor of shape ``(N, C, P)``.
    """
    grid = 2.0 * point_coords - 1.0
    grid = grid.unsqueeze(2)
    return F.grid_sample(
        input=input,
        grid=grid,
        mode=mode,
        align_corners=False,
    ).squeeze(3)


def sigmoid_ce_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Computes summed sigmoid cross-entropy over sampled mask points."""
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    result = loss.mean(1).sum()
    if not isinstance(result, Tensor):
        raise TypeError("Expected sigmoid CE loss to be a tensor.")
    return result


def dice_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    """Computes summed dice loss over sampled mask points."""
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    result = (1 - (numerator + 1) / (denominator + 1)).sum()
    if not isinstance(result, Tensor):
        raise TypeError("Expected dice loss to be a tensor.")
    return result


def sigmoid_ce_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    """Computes pairwise sigmoid cross-entropy costs.

    Uses the factored formulation to avoid materializing a ``(Q, T, P)`` tensor.

    Args:
        inputs: Predicted mask logits of shape ``(Q, P)``.
        targets: Target masks of shape ``(T, P)``.

    Returns:
        Pairwise cost matrix of shape ``(Q, T)``.
    """
    num_points = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    loss = torch.einsum("qp,tp->qt", pos, targets) + torch.einsum(
        "qp,tp->qt", neg, 1 - targets
    )
    result = loss / num_points
    if not isinstance(result, Tensor):
        raise TypeError("Expected sigmoid CE cost to be a tensor.")
    return result


def dice_cost(inputs: Tensor, targets: Tensor) -> Tensor:
    """Computes pairwise dice costs.

    Args:
        inputs: Predicted mask logits of shape ``(Q, P)``.
        targets: Target masks of shape ``(T, P)``.

    Returns:
        Pairwise cost matrix of shape ``(Q, T)``.
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("qp,tp->qt", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    result = 1 - (numerator + 1) / (denominator + 1)
    if not isinstance(result, Tensor):
        raise TypeError("Expected dice cost to be a tensor.")
    return result


def get_uncertain_point_coords_with_randomness(
    logits: Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> Tensor:
    """Samples points using mask-logit uncertainty plus random coverage."""
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    if logits.shape[0] == 0:
        return logits.new_empty((0, num_points, 2))

    num_sampled = int(num_points * oversample_ratio)
    num_uncertain_points = int(num_points * importance_sample_ratio)
    num_random_points = num_points - num_uncertain_points

    point_coords = torch.rand(
        logits.shape[0],
        num_sampled,
        2,
        device=logits.device,
        dtype=logits.dtype,
    )
    point_logits = point_sample(logits[:, None], point_coords).squeeze(1)
    point_uncertainties = -point_logits.abs()

    _, point_indices = torch.topk(
        point_uncertainties,
        k=num_uncertain_points,
        dim=1,
    )
    point_indices = point_indices[:, :, None].expand(-1, -1, 2)
    selected_point_coords = point_coords.gather(dim=1, index=point_indices)

    if num_random_points > 0:
        random_point_coords = torch.rand(
            logits.shape[0],
            num_random_points,
            2,
            device=logits.device,
            dtype=logits.dtype,
        )
        selected_point_coords = torch.cat(
            [selected_point_coords, random_point_coords],
            dim=1,
        )

    return selected_point_coords
