#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def bilinear_grid_sample(
    im: Tensor, grid: Tensor, align_corners: bool = False
) -> Tensor:
    """Pure gather-based equivalent of ``F.grid_sample`` (bilinear, zero padding).

    ``torch.nn.functional.grid_sample`` exports to an ONNX ``GridSample`` op that
    TensorRT converts incorrectly for the deformable-attention sampling used by
    LT-DETR (the engine produces wrong outputs even though the inputs are exact).
    This decomposition into ``Pad`` + ``Gather`` + arithmetic matches
    ``F.grid_sample(mode="bilinear", padding_mode="zeros", align_corners=...)`` to
    within floating-point tolerance and builds a correct TensorRT engine.

    Args:
        im:
            Input feature map of shape ``(N, C, H, W)``.
        grid:
            Sampling grid of shape ``(N, H_out, W_out, 2)`` with coordinates
            normalized to ``[-1, 1]``.
        align_corners:
            Matches the ``align_corners`` argument of ``F.grid_sample``.

    Returns:
        Sampled tensor of shape ``(N, C, H_out, W_out)``.
    """
    n, c, h, w = im.shape
    _, gh, gw, _ = grid.shape

    x = grid[..., 0]
    y = grid[..., 1]
    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.reshape(n, -1)
    y = y.reshape(n, -1)
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Bilinear interpolation weights (computed before clamping so out-of-bounds
    # samples keep their correct weights and only fetch zeros).
    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Zero-pad by one pixel on each side so that out-of-bounds coordinates land on
    # the zero border, reproducing ``padding_mode="zeros"``.
    im = F.pad(im, [1, 1, 1, 1])
    padded_h = h + 2
    padded_w = w + 2
    x0 = (x0 + 1).long().clamp(0, padded_w - 1)
    x1 = (x1 + 1).long().clamp(0, padded_w - 1)
    y0 = (y0 + 1).long().clamp(0, padded_h - 1)
    y1 = (y1 + 1).long().clamp(0, padded_h - 1)

    im = im.reshape(n, c, padded_h * padded_w)

    def _gather(xx: Tensor, yy: Tensor) -> Tensor:
        idx = (xx + yy * padded_w).unsqueeze(1).expand(-1, c, -1)
        return torch.gather(im, 2, idx)

    ia = _gather(x0, y0)
    ib = _gather(x0, y1)
    ic = _gather(x1, y0)
    id_ = _gather(x1, y1)

    return (ia * wa + ib * wb + ic * wc + id_ * wd).reshape(n, c, gh, gw)


def deformable_attention_core_func(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|list): [n_levels, 2]
        value_level_start_index (Tensor|list): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = (
            value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        )
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = (
            sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        )
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .reshape(bs, n_head * c, Len_q)
    )

    return output.permute(0, 2, 1)


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: list[int],
    method="default",
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|list): [n_levels, 2]
        value_level_start_index (Tensor|list): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, _, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)

    # sampling_offsets [8, 480, 8, 12, 2]
    if method in ("default", "bilinear_gather"):
        sampling_grids = 2 * sampling_locations - 1

    elif method == "discrete":
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

        elif method == "bilinear_gather":
            # Gather-based equivalent of the "default" grid_sample that exports to
            # a TensorRT-compatible graph (see ``bilinear_grid_sample``). Used for
            # deployment/export; training keeps the faster fused grid_sample.
            sampling_value_l = bilinear_grid_sample(
                value_l, sampling_grid_l, align_corners=False
            )

        elif method == "discrete":
            # n * m, seq, n, 2
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value.device) + 0.5
            ).to(torch.int64)

            # FIX ME? for rectangle input
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(
                bs * n_head, Len_q * num_points_list[level], 2
            )

            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l: torch.Tensor = value_l[
                s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]
            ]  # n l c

            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level]
            )

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, Len_q, sum(num_points_list)
    )
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


def get_activation(act: str, inpace: bool = True):
    """get activation"""
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == "silu" or act == "swish":
        m = nn.SiLU()

    elif act == "relu":
        m = nn.ReLU()

    elif act == "leaky_relu":
        m = nn.LeakyReLU()

    elif act == "silu":
        m = nn.SiLU()

    elif act == "gelu":
        m = nn.GELU()

    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError("")

    if hasattr(m, "inplace"):
        m.inplace = inpace

    return m
