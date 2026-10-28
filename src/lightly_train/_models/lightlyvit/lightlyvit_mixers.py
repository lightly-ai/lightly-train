#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Experimental token mixers for the LightlyViT efficiency sweep.

These mixers live behind the same contracts as the v1 ``DepthwiseConvMixer``
(local: ``(patch_tokens, grid_size, valid_token_mask) -> patch_tokens``) and
``GlobalSelfAttention`` (global: ``(x, attention_mask, rope) -> x`` with an
optional RoPE table). They are experimental scaffolding for the
``archive/lightlyvit_efficiency_sweep.md`` complex-mixer comparison and are
**not** exposed by the public ``lightlyvit_s16_v1`` factory, which remains the
depthwise-conv + softmax-attention reference.

Padding handling is approximate but safe: local mixers zero padded grid cells
before and after mixing so padded tokens cannot introduce NaNs and only leak
within their local/window neighborhood. That is sufficient for a latency
sweep; a future dense-distillation path would consume the explicit valid-token
mask instead.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@runtime_checkable
class LocalTokenMixer(Protocol):
    """Local mixer contract shared with ``DepthwiseConvMixer``."""

    def forward(
        self,
        patch_tokens: Tensor,
        grid_size: tuple[int, int],
        valid_token_mask: Tensor | None = None,
    ) -> Tensor: ...


@runtime_checkable
class GlobalTokenMixer(Protocol):
    """Global mixer contract shared with ``GlobalSelfAttention``."""

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        rope: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor: ...


def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


def apply_rope_to_qk(
    q: Tensor, k: Tensor, rope: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    """Rotate the trailing patch tokens of ``q``/``k`` with RoPE tables.

    Mirrors DINOv3's ``SelfAttention.apply_rope``
    (``dinov3_src/layers/attention.py``): the leading ``prefix`` tokens (CLS,
    registers, and any head-injected query tokens) are passed through
    unrotated, the rotation runs in the table dtype (fp32) and the result is
    cast back to the input dtype. ``q``/``k`` are ``B x heads x N x head_dim``;
    the tables are ``N_patch x head_dim``.
    """

    q_dtype, k_dtype = q.dtype, k.dtype
    sin, cos = rope
    prefix = q.shape[-2] - sin.shape[-2]
    if prefix < 0:
        raise ValueError(
            f"RoPE table covers {sin.shape[-2]} positions but the sequence has "
            f"only {q.shape[-2]} tokens."
        )
    q = q.to(dtype=sin.dtype)
    k = k.to(dtype=sin.dtype)
    q = torch.cat(
        [q[:, :, :prefix, :], rope_apply(q[:, :, prefix:, :], sin, cos)], dim=-2
    )
    k = torch.cat(
        [k[:, :, :prefix, :], rope_apply(k[:, :, prefix:, :], sin, cos)], dim=-2
    )
    return q.to(dtype=q_dtype), k.to(dtype=k_dtype)


def _apply_valid_mask_2d(x4d: Tensor, valid_token_mask: Tensor | None) -> Tensor:
    """Zero padded spatial cells on a ``B x C x H x W`` tensor."""

    if valid_token_mask is None:
        return x4d
    h, w = x4d.shape[-2:]
    mask = valid_token_mask[:, :h, :w].to(dtype=x4d.dtype).unsqueeze(1)
    return x4d * mask


class DepthwiseSeparableConvMixer(nn.Module):
    """Depthwise-separable conv mixer (DW k×k + pointwise), mobile style.

    Unlike the plain ``DepthwiseConvMixer`` this adds a 1×1 pointwise
    projection, re-introducing channel mixing inside the token-mixing sublayer
    (which the v1 reference deliberately leaves to the MLP). The sweep tests
    whether that extra channel mixing buys anything at the cost of a pointwise
    MatMul per local block.
    """

    def __init__(self, dim: int, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.dw = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(
        self,
        patch_tokens: Tensor,
        grid_size: tuple[int, int],
        valid_token_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, num_tokens, channels = patch_tokens.shape
        grid_height, grid_width = grid_size
        if num_tokens != grid_height * grid_width:
            raise ValueError("Patch token count does not match grid size.")
        x = patch_tokens.transpose(1, 2).reshape(
            batch_size, channels, grid_height, grid_width
        )
        x = _apply_valid_mask_2d(x, valid_token_mask)
        x = self.pw(self.dw(x))
        x = _apply_valid_mask_2d(x, valid_token_mask)
        return x.flatten(2).transpose(1, 2).contiguous()


class WindowAttentionMixer(nn.Module):
    """Non-shifted window attention over the patch grid.

    Partition the H×W grid into non-overlapping ``window_size × window_size``
    windows and run standard multi-head self-attention within each window.
    The grid is internally padded to a multiple of the window size and cropped
    back, so non-divisible shapes work. This is the direct latency/quality
    alternative to the depthwise-conv mixer that ``archive/efficientbackbone.md``
    deferred.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 7) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        if window_size < 1:
            raise ValueError("window_size must be positive.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        patch_tokens: Tensor,
        grid_size: tuple[int, int],
        valid_token_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, num_tokens, channels = patch_tokens.shape
        grid_height, grid_width = grid_size
        if num_tokens != grid_height * grid_width:
            raise ValueError("Patch token count does not match grid size.")
        ws = self.window_size
        x = patch_tokens.transpose(1, 2).reshape(
            batch_size, channels, grid_height, grid_width
        )
        x = _apply_valid_mask_2d(x, valid_token_mask)
        pad_h = (-grid_height) % ws
        pad_w = (-grid_width) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        gh, gw = hp // ws, wp // ws
        # B, C, gh, ws, gw, ws -> B, gh*gw, ws*ws, C
        x = x.reshape(batch_size, channels, gh, ws, gw, ws)
        windows = x.permute(0, 2, 4, 3, 5, 1).reshape(
            batch_size, gh * gw, ws * ws, channels
        )
        qkv = (
            self.qkv(windows)
            .reshape(batch_size, gh * gw, ws * ws, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = (q * self.scale) @ k.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        out = attention @ v  # B, gh*gw, heads, ws*ws, head_dim
        out = out.transpose(2, 3).reshape(batch_size, gh * gw, ws * ws, channels)
        out = self.proj(out)
        # Unpartition: B, gh*gw, ws*ws, C -> B, C, gh, ws, gw, ws -> B, C, hp, wp
        out = (
            out.reshape(batch_size, gh, gw, ws, ws, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch_size, channels, hp, wp)
        )
        out = out[:, :, :grid_height, :grid_width]
        out = _apply_valid_mask_2d(out, valid_token_mask)
        return out.flatten(2).transpose(1, 2).contiguous()


class LocalAttentionMixer(nn.Module):
    """Sliding-window (neighborhood) attention over the patch grid.

    Each query attends to a ``window_size × window_size`` neighborhood centered
    on it (stride-1, overlapping windows), implemented with ``unfold``. This
    sits between the depthwise conv (fixed, content-independent weights) and
    full window attention (non-overlapping): it gives attention-quality local
    mixing with a receptive field larger than a single window partition.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 7) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        if window_size % 2 == 0 or window_size < 3:
            raise ValueError("window_size must be an odd integer >= 3.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        patch_tokens: Tensor,
        grid_size: tuple[int, int],
        valid_token_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, num_tokens, channels = patch_tokens.shape
        grid_height, grid_width = grid_size
        if num_tokens != grid_height * grid_width:
            raise ValueError("Patch token count does not match grid size.")
        k = self.window_size
        x = patch_tokens.transpose(1, 2).reshape(
            batch_size, channels, grid_height, grid_width
        )
        x = _apply_valid_mask_2d(x, valid_token_mask)
        # Full QKV projection over all tokens.
        qkv = (
            self.qkv(x.flatten(2).transpose(1, 2))
            .reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k_all, v_all = qkv[0], qkv[1], qkv[2]  # B, heads, N, head_dim
        # Per-head channel map for neighborhood gathering.
        k_map = k_all.reshape(
            batch_size * self.num_heads, self.head_dim, grid_height, grid_width
        )
        v_map = v_all.reshape(
            batch_size * self.num_heads, self.head_dim, grid_height, grid_width
        )
        pad = k // 2
        kh = F.unfold(k_map, kernel_size=k, padding=pad)  # BH, head_dim*k*k, N
        vh = F.unfold(v_map, kernel_size=k, padding=pad)
        nbh = k * k
        # BH, N, head_dim, nbh -> BH, N, nbh, head_dim
        kh = kh.reshape(
            batch_size * self.num_heads, self.head_dim, nbh, num_tokens
        ).permute(0, 3, 2, 1)
        vh = vh.reshape(
            batch_size * self.num_heads, self.head_dim, nbh, num_tokens
        ).permute(0, 3, 2, 1)
        q = q.reshape(batch_size * self.num_heads, num_tokens, 1, self.head_dim)
        attention = (q * self.scale) @ kh.transpose(-2, -1)  # BH, N, 1, nbh
        attention = attention.softmax(dim=-1)
        out = attention @ vh  # BH, N, 1, head_dim
        out = out.reshape(batch_size, self.num_heads, num_tokens, self.head_dim)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(batch_size, channels, grid_height, grid_width)
        out = _apply_valid_mask_2d(out, valid_token_mask)
        return out.flatten(2).transpose(1, 2).contiguous()


class LinearGlobalAttention(nn.Module):
    """Linear (kernelized) global self-attention for the global blocks.

    Replaces softmax attention with ``φ(q) (φ(k)ᵀ v)`` using the
    ``elu(x)+1`` feature map, so cost is linear in the token count. This is
    the ``archive/efficientbackbone.md`` ``MobileViTv2``-style alternative that
    addresses the "global attention still scales quadratically" risk at high
    resolution. ``attention_mask`` is honored by zeroing masked key/value
    contributions to the linear-attention context.

    When a RoPE table is supplied, raw ``q``/``k`` are rotated *before* the
    feature map. ``elu+1`` is not magnitude-preserving, so the exact
    relative-offset algebra of RoPE does not survive the kernel; this is a
    deliberate approximation that still injects position into the similarity,
    gated by the v8 pre-flight A/B (see the quality ledger).
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    @staticmethod
    def _feature(x: Tensor) -> Tensor:
        return F.elu(x) + 1.0

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        rope: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, heads, N, head_dim
        if rope is not None:
            q, k = apply_rope_to_qk(q=q, k=k, rope=rope)
        q = self._feature(q)
        k = self._feature(k)
        # The global block's mask is a per-key validity mask (True == valid)
        # that is identical across queries: it marks padded positions to
        # exclude. Collapse any mask shape to (B, Nk) and zero those keys.
        if attention_mask is not None:
            key_mask = attention_mask
            if key_mask.dtype != torch.bool:
                key_mask = key_mask > 0
            if key_mask.ndim == 2:
                key_valid_1d = key_mask  # B, Nk
            else:
                # ndim 3 (B, Nq, Nk) or 4 (B, H, Nq, Nk): AND over every axis
                # except batch and the final key axis.
                reduce_dims = tuple(range(1, key_mask.ndim - 1))
                key_valid_1d = key_mask.all(dim=reduce_dims)  # B, Nk
            k = k * key_valid_1d[:, None, :, None].to(dtype=k.dtype)
            v = v * key_valid_1d[:, None, :, None].to(dtype=v.dtype)
        # Linear attention: context = kᵀ v  (heads, d, d); out = q @ context.
        context = k.transpose(-2, -1) @ v  # B, heads, head_dim, head_dim
        out = q @ context  # B, heads, N, head_dim
        denom = q @ k.sum(dim=-2, keepdim=False).unsqueeze(-1)  # B, heads, N, 1
        out = out / denom.clamp(min=self.eps)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj(out)  # type: ignore[no-any-return]
