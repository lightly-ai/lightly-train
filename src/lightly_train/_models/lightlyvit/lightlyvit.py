#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""The fixed, single-scale LightlyViT v1 backbone.

LightlyViT keeps a conventional ViT token interface while replacing most full
self-attention blocks with a cheap spatial depthwise-convolution mixer.  The
implementation deliberately has no hierarchy or deployment-specific branch:
the only architectural difference between local and global blocks is their
token-mixing sublayer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lightly_train._activation_checkpointing import maybe_checkpoint
from lightly_train._models.dinov3.dinov3_src.layers.rope_position_encoding import (
    RopePositionEmbedding,
)
from lightly_train._models.lightlyvit.lightlyvit_mixers import (
    DepthwiseSeparableConvMixer,
    LinearGlobalAttention,
    LocalAttentionMixer,
    WindowAttentionMixer,
    apply_rope_to_qk,
)


@dataclass(frozen=True)
class LightlyViTGridMetadata:
    """Spatial metadata for one padded LightlyViT input.

    ``original_grid_size`` counts patch cells that overlap the unpadded image.
    It consequently uses ceiling division.  ``fully_valid_grid_size`` is
    available for consumers that prefer to discard partial boundary patches.
    In v1 the patch grid is padded only to the patch size, so
    ``original_grid_size`` and ``padded_grid_size`` are normally equal.  Both
    are retained explicitly so downstream dense consumers do not need to infer
    geometry from tensor shapes.
    """

    original_size: tuple[int, int]
    padded_size: tuple[int, int]
    original_grid_size: tuple[int, int]
    padded_grid_size: tuple[int, int]
    valid_token_mask: Tensor
    patch_size: int

    @property
    def original_image_size(self) -> tuple[int, int]:
        """Alias for :attr:`original_size`."""

        return self.original_size

    @property
    def padded_image_size(self) -> tuple[int, int]:
        """Alias for :attr:`padded_size`."""

        return self.padded_size

    @property
    def grid_size(self) -> tuple[int, int]:
        """The grid that token tensors are laid out on."""

        return self.padded_grid_size

    @property
    def fully_valid_grid_size(self) -> tuple[int, int]:
        """Number of patch cells fully covered by the original image."""

        height, width = self.original_size
        return height // self.patch_size, width // self.patch_size

    @property
    def num_patches(self) -> int:
        """Number of tokens in the padded patch grid."""

        height, width = self.padded_grid_size
        return height * width


@dataclass(frozen=True)
class LightlyViTBackboneOutput:
    """Typed result produced by :class:`LightlyViT` feature extraction."""

    patch_tokens: Tensor
    features: Tensor
    cls_token: Tensor
    register_tokens: Tensor
    valid_token_mask: Tensor
    grid_metadata: LightlyViTGridMetadata
    prenorm_tokens: Tensor

    @property
    def dense_features(self) -> Tensor:
        """Alias for the dense ``B x C x H_p x W_p`` feature map."""

        return self.features

    @property
    def grid(self) -> LightlyViTGridMetadata:
        """Alias for :attr:`grid_metadata`."""

        return self.grid_metadata

    @property
    def tokens(self) -> Tensor:
        """The normalized CLS/register/patch token sequence."""

        return torch.cat(
            [
                self.cls_token.unsqueeze(1),
                self.register_tokens,
                self.patch_tokens,
            ],
            dim=1,
        )


@dataclass(frozen=True)
class _LightlyViTVariantSpec:
    """Frozen family configuration.

    All three variants use the linear global mixer.  ``T/16`` and ``S/16``
    pair it with RoPE position encoding (the recipes their released
    checkpoints were trained with);  ``M/16`` uses a learned position encoding.
    """

    width: int
    depth: int
    num_heads: int
    global_block_indices: tuple[int, ...]


_LIGHTLYVIT_T16_SPEC = _LightlyViTVariantSpec(
    width=224,
    depth=12,
    num_heads=4,
    global_block_indices=(3, 7, 11),
)
_LIGHTLYVIT_S16_SPEC = _LightlyViTVariantSpec(
    width=432,
    depth=12,
    num_heads=9,
    global_block_indices=(3, 7, 11),
)
_LIGHTLYVIT_M16_SPEC = _LightlyViTVariantSpec(
    width=640,
    depth=16,
    num_heads=10,
    # One global block every four, matching the S/16 cadence extended to depth 16.
    global_block_indices=(3, 7, 11, 15),
)


class LayerScale(nn.Module):
    """Per-channel residual scaling used by both LightlyViT block types."""

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class DropPath(nn.Module):
    """Stochastic depth, kept as a small local implementation for v1."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


class MLP(nn.Module):
    """The conventional GELU feed-forward sublayer used in every block."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))  # type: ignore[no-any-return]


class DepthwiseConvMixer(nn.Module):
    """A 2D local mixer operating only on a patch-token suffix."""

    def __init__(self, dim: int, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("DepthwiseConvMixer requires an odd kernel size.")
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )

    def forward(
        self,
        patch_tokens: Tensor,
        grid_size: tuple[int, int],
        valid_token_mask: Tensor | None = None,
    ) -> Tensor:
        """Spatially mix ``B x (H*W) x C`` patch tokens.

        Special tokens are deliberately not accepted here.  This makes it
        impossible for a local mixer to accidentally touch CLS, registers, or
        query tokens injected by an EoMT-style head.
        """

        batch_size, num_tokens, channels = patch_tokens.shape
        grid_height, grid_width = grid_size
        if num_tokens != grid_height * grid_width:
            raise ValueError(
                "Patch token count does not match grid size: "
                f"{num_tokens} != {grid_height} * {grid_width}."
            )

        x = patch_tokens.transpose(1, 2).reshape(
            batch_size, channels, grid_height, grid_width
        )
        if valid_token_mask is not None:
            valid = valid_token_mask.to(dtype=x.dtype).unsqueeze(1)
            x = x * valid
        x = self.conv(x)
        if valid_token_mask is not None:
            x = x * valid_token_mask.to(dtype=x.dtype).unsqueeze(1)
        return x.flatten(2).transpose(1, 2).contiguous()  # type: ignore[no-any-return]


class GlobalSelfAttention(nn.Module):
    """Plain multi-head self-attention with an optional standard mask."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension {dim} must be divisible by {num_heads} heads."
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        rope: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Apply attention.

        Boolean masks use the conventional ``True == allowed`` convention and
        may be shaped ``B x N`` (key mask) or broadcastable to ``B x H x N x
        N``.  Floating masks are interpreted as additive logits biases.

        When ``rope`` is supplied, the trailing patch tokens of ``q``/``k``
        are rotated with the table before the similarity product (exact RoPE;
        ``v`` and the special-token prefix stay unrotated).
        """

        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if rope is not None:
            q, k = apply_rope_to_qk(q=q, k=k, rope=rope)

        attention = (q * self.scale) @ k.transpose(-2, -1)
        if attention_mask is not None:
            mask = self._prepare_attention_mask(
                attention_mask=attention_mask,
                batch_size=batch_size,
                num_tokens=num_tokens,
            )
            if mask.dtype == torch.bool:
                attention = attention.masked_fill(
                    ~mask, torch.finfo(attention.dtype).min
                )
            else:
                attention = attention + mask.to(dtype=attention.dtype)
        attention = attention.softmax(dim=-1)
        x = attention @ v
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        return self.proj(x)  # type: ignore[no-any-return]

    @staticmethod
    def _prepare_attention_mask(
        attention_mask: Tensor, batch_size: int, num_tokens: int
    ) -> Tensor:
        if attention_mask.ndim == 2:
            if attention_mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    "A 2D attention mask must have shape B x N, got "
                    f"{tuple(attention_mask.shape)}."
                )
            return attention_mask[:, None, None, :]
        if attention_mask.ndim in (3, 4):
            if attention_mask.shape[0] != batch_size:
                raise ValueError(
                    "Attention mask batch dimension does not match input: "
                    f"{attention_mask.shape[0]} != {batch_size}."
                )
            return attention_mask
        raise ValueError(
            "Attention mask must have shape B x N, B x N x N, or B x H x N x N."
        )


LOCAL_MIXER_KINDS = ("dw_conv", "dw_separable", "window", "local")
GLOBAL_MIXER_KINDS = ("softmax", "linear")
POS_ENCODING_KINDS = ("learned_absolute", "rope")


def _build_local_mixer(
    *,
    kind: str,
    dim: int,
    kernel_size: int,
    window_size: int,
    num_heads: int,
) -> nn.Module:
    if kind == "dw_conv":
        return DepthwiseConvMixer(dim=dim, kernel_size=kernel_size)
    if kind == "dw_separable":
        return DepthwiseSeparableConvMixer(dim=dim, kernel_size=kernel_size)
    if kind == "window":
        return WindowAttentionMixer(
            dim=dim, num_heads=num_heads, window_size=window_size
        )
    if kind == "local":
        return LocalAttentionMixer(
            dim=dim, num_heads=num_heads, window_size=window_size
        )
    raise ValueError(
        f"Unknown local_mixer_kind {kind!r}; choose from {LOCAL_MIXER_KINDS}."
    )


def _build_global_mixer(*, kind: str, dim: int, num_heads: int) -> nn.Module:
    if kind == "softmax":
        return GlobalSelfAttention(dim=dim, num_heads=num_heads)
    if kind == "linear":
        return LinearGlobalAttention(dim=dim, num_heads=num_heads)
    raise ValueError(
        f"Unknown global_mixer_kind {kind!r}; choose from {GLOBAL_MIXER_KINDS}."
    )


class LightlyViTBlock(nn.Module):
    """One local-convolution or full-attention LightlyViT block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        is_global: bool,
        mlp_ratio: float = 4.0,
        layer_scale_init_value: float = 1e-5,
        drop_path: float = 0.0,
        mixer_kernel_size: int = 7,
        local_mixer_kind: str = "dw_conv",
        global_mixer_kind: str = "softmax",
        window_size: int = 7,
    ) -> None:
        super().__init__()
        self.is_global = is_global
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn: nn.Module | None
        self.local_mixer: nn.Module | None
        if is_global:
            self.attn = _build_global_mixer(
                kind=global_mixer_kind, dim=dim, num_heads=num_heads
            )
            self.local_mixer = None
        else:
            self.attn = None
            self.local_mixer = _build_local_mixer(
                kind=local_mixer_kind,
                dim=dim,
                kernel_size=mixer_kernel_size,
                window_size=window_size,
                num_heads=num_heads,
            )
        self.ls1 = LayerScale(dim, init_value=layer_scale_init_value)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio)
        self.ls2 = LayerScale(dim, init_value=layer_scale_init_value)
        self.drop_path2 = DropPath(drop_path)

    def forward(
        self,
        tokens: Tensor,
        grid_size: tuple[int, int],
        *,
        num_special_tokens: int,
        valid_token_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        rope: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Execute one block with an arbitrary special-token prefix.

        The local spatial sublayer only receives the patch suffix.  The common
        MLP remains a standard ViT MLP over all tokens, so local blocks retain
        the intended CLS/register feed-forward update while never spatially
        mixing special tokens.  ``rope`` is only consumed by global blocks.
        """

        if num_special_tokens < 0 or num_special_tokens > tokens.shape[1]:
            raise ValueError(
                "num_special_tokens must describe a prefix of the token sequence."
            )
        patch_tokens = tokens[:, num_special_tokens:]
        grid_height, grid_width = grid_size
        if patch_tokens.shape[1] != grid_height * grid_width:
            raise ValueError(
                "Patch suffix length does not match the supplied grid size: "
                f"{patch_tokens.shape[1]} != {grid_height} * {grid_width}."
            )

        if self.is_global:
            assert self.attn is not None
            tokens = tokens + self.drop_path1(
                self.ls1(
                    self.attn(
                        self.norm1(tokens), attention_mask=attention_mask, rope=rope
                    )
                )
            )
        else:
            assert self.local_mixer is not None
            mixed_patches = self.local_mixer(
                self.norm1(patch_tokens),
                grid_size=grid_size,
                valid_token_mask=valid_token_mask,
            )
            tokens = torch.cat(
                [
                    tokens[:, :num_special_tokens],
                    patch_tokens + self.drop_path1(self.ls1(mixed_patches)),
                ],
                dim=1,
            )

        return tokens + self.drop_path2(  # type: ignore[no-any-return]
            self.ls2(self.mlp(self.norm2(tokens)))
        )


class LightlyViT(nn.Module):
    """A fixed-family single-scale ViT with periodic global attention.

    ``lightlyvit_s16`` is the supported public construction route.  The
    constructor accepts dimensions to keep unit tests and future frozen family
    variants local to this implementation, but it does not make additional
    package model names public.
    """

    def __init__(
        self,
        *,
        image_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 432,
        depth: int = 12,
        num_heads: int = 9,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 4,
        global_block_indices: Sequence[int] = (3, 7, 11),
        layer_scale_init_value: float = 1e-5,
        drop_path_rate: float = 0.0,
        mixer_kernel_size: int = 7,
        local_mixer_kind: str = "dw_conv",
        global_mixer_kind: str = "softmax",
        window_size: int = 7,
        pos_encoding: str = "learned_absolute",
        rope_base: float = 100.0,
        rope_rescale_coords: float | None = 2.0,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if local_mixer_kind not in LOCAL_MIXER_KINDS:
            raise ValueError(
                f"local_mixer_kind must be one of {LOCAL_MIXER_KINDS}, "
                f"got {local_mixer_kind!r}."
            )
        if global_mixer_kind not in GLOBAL_MIXER_KINDS:
            raise ValueError(
                f"global_mixer_kind must be one of {GLOBAL_MIXER_KINDS}, "
                f"got {global_mixer_kind!r}."
            )
        if window_size < 1:
            raise ValueError("window_size must be positive.")
        if pos_encoding not in POS_ENCODING_KINDS:
            raise ValueError(
                f"pos_encoding must be one of {POS_ENCODING_KINDS}, got {pos_encoding!r}."
            )
        if pos_encoding == "rope" and embed_dim % (4 * num_heads) != 0:
            raise ValueError(
                "pos_encoding='rope' requires embed_dim to be divisible by "
                f"4 * num_heads ({embed_dim} vs {4 * num_heads})."
            )
        if mixer_kernel_size < 3 or mixer_kernel_size % 2 == 0:
            raise ValueError(
                "mixer_kernel_size must be an odd integer >= 3, "
                f"got {mixer_kernel_size}."
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if num_register_tokens < 0:
            raise ValueError("num_register_tokens must be non-negative.")
        global_indices = tuple(int(index) for index in global_block_indices)
        if len(set(global_indices)) != len(global_indices):
            raise ValueError("global_block_indices must not contain duplicates.")
        if any(index < 0 or index >= depth for index in global_indices):
            raise ValueError(
                "global_block_indices must be valid zero-based block indices."
            )

        base_image_size = _to_2tuple(image_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.num_features = self.embed_dim
        self.num_blocks = int(depth)
        self.n_blocks = self.num_blocks
        self.num_heads = int(num_heads)
        self.mixer_kernel_size = int(mixer_kernel_size)
        self.local_mixer_kind = str(local_mixer_kind)
        self.global_mixer_kind = str(global_mixer_kind)
        self.window_size = int(window_size)
        self.pos_encoding = str(pos_encoding)
        self.num_register_tokens = int(num_register_tokens)
        # Alias matching the DINOv3 term so generic adapter code can read either.
        self.n_storage_tokens = self.num_register_tokens
        self.global_block_indices = global_indices
        self.base_image_size = base_image_size
        self.base_grid_size = _ceil_grid_size(base_image_size, self.patch_size)

        self.patch_embed = nn.Conv2d(
            in_chans,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.embed_dim))
        self.register_tokens = nn.Parameter(
            torch.empty(1, self.num_register_tokens, self.embed_dim)
        )
        self.mask_token = nn.Parameter(torch.empty(1, 1, self.embed_dim))

        if self.pos_encoding == "rope":
            # Rotary positions are applied to q/k inside every global block
            # from the actual (padded) grid, so there is no patch-grid table
            # and nothing to interpolate; CLS/register keep their grid-free
            # learned embeddings below. Local conv blocks receive no explicit
            # positional input (DINOv3 ConvNeXt precedent). The configuration
            # mirrors every shipped DINOv3 builder: axial tables normalized
            # per axis ("separate"), and a train-time log-uniform coordinate
            # rescale in [0.5, 2] for resolution robustness (inert at eval).
            self.rope_embed = RopePositionEmbedding(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                base=rope_base,
                normalize_coords="separate",
                rescale_coords=rope_rescale_coords,
                dtype=torch.float32,
            )
        else:
            # Keep patch positions genuinely two-dimensional so interpolation
            # is unambiguous for non-square inputs.
            self.pos_embed = nn.Parameter(
                torch.empty(1, self.embed_dim, *self.base_grid_size)
            )
        self.cls_pos_embed = nn.Parameter(torch.empty(1, 1, self.embed_dim))
        self.register_pos_embed = nn.Parameter(
            torch.empty(1, self.num_register_tokens, self.embed_dim)
        )

        drop_path_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                LightlyViTBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    is_global=index in self.global_block_indices,
                    mlp_ratio=mlp_ratio,
                    layer_scale_init_value=layer_scale_init_value,
                    drop_path=drop_path_rates[index],
                    mixer_kernel_size=self.mixer_kernel_size,
                    local_mixer_kind=self.local_mixer_kind,
                    global_mixer_kind=self.global_mixer_kind,
                    window_size=self.window_size,
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.head = nn.Identity()
        # Activation checkpointing flags; set through LightlyViTModelWrapper.
        self._activation_checkpointing = False
        self._activation_checkpointing_every_n_blocks = 1
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize all trainable components with standard ViT defaults."""

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        if self.pos_encoding == "learned_absolute":
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.register_pos_embed, std=0.02)
        self.apply(_init_module_weights)

    def interpolate_pos_encoding(self, grid_size: tuple[int, int]) -> Tensor:
        """Return learned patch positions interpolated to ``grid_size``."""

        if self.pos_encoding != "learned_absolute":
            raise RuntimeError(
                "interpolate_pos_encoding is only available with "
                f"pos_encoding='learned_absolute', not {self.pos_encoding!r}."
            )
        if grid_size == self.base_grid_size:
            pos_embed = self.pos_embed
        else:
            pos_embed = F.interpolate(
                self.pos_embed,
                size=grid_size,
                mode="bicubic",
                align_corners=False,
            )
        return pos_embed.flatten(2).transpose(1, 2).contiguous()

    def prepare_tokens_with_metadata(
        self, x: Tensor, masks: Tensor | None = None
    ) -> tuple[Tensor, LightlyViTGridMetadata]:
        """Patchify an image and return standard tokens plus spatial metadata."""

        if x.ndim != 4:
            raise ValueError(
                f"Expected image tensor with shape B x C x H x W, got {tuple(x.shape)}."
            )
        batch_size, _, image_height, image_width = x.shape
        padded_height = _ceil_to_multiple(image_height, self.patch_size)
        padded_width = _ceil_to_multiple(image_width, self.patch_size)
        if padded_height != image_height or padded_width != image_width:
            x = F.pad(
                x,
                (0, padded_width - image_width, 0, padded_height - image_height),
            )

        patch_tokens = self.patch_embed(x).flatten(2).transpose(1, 2).contiguous()
        grid_size = (padded_height // self.patch_size, padded_width // self.patch_size)
        patch_mask = self._normalize_patch_mask(
            masks=masks,
            batch_size=batch_size,
            grid_size=grid_size,
            device=patch_tokens.device,
        )
        if patch_mask is not None:
            patch_tokens = torch.where(
                patch_mask.unsqueeze(-1),
                self.mask_token.to(dtype=patch_tokens.dtype),
                patch_tokens,
            )

        # A token is valid when its patch overlaps the original image.  After
        # patch-size padding every produced cell does overlap, but computing the
        # mask from the original extent makes this contract explicit and keeps it
        # correct if grid padding is added later.
        original_grid_size = _ceil_grid_size(
            (image_height, image_width), self.patch_size
        )
        rows = torch.arange(grid_size[0], device=patch_tokens.device)
        cols = torch.arange(grid_size[1], device=patch_tokens.device)
        valid_token_mask = (
            (rows[:, None] < original_grid_size[0])
            & (cols[None, :] < original_grid_size[1])
        ).expand(batch_size, -1, -1)
        metadata = LightlyViTGridMetadata(
            original_size=(image_height, image_width),
            padded_size=(padded_height, padded_width),
            original_grid_size=original_grid_size,
            padded_grid_size=grid_size,
            valid_token_mask=valid_token_mask,
            patch_size=self.patch_size,
        )

        if self.pos_encoding == "learned_absolute":
            patch_tokens = patch_tokens + self.interpolate_pos_encoding(grid_size).to(
                dtype=patch_tokens.dtype
            )
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) + self.cls_pos_embed
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        register_tokens = register_tokens + self.register_pos_embed
        tokens = torch.cat([cls_tokens, register_tokens, patch_tokens], dim=1)
        return tokens, metadata

    def prepare_tokens_with_masks(
        self, x: Tensor, masks: Tensor | None = None
    ) -> tuple[Tensor, tuple[int, int]]:
        """DINO-style compatibility helper returning tokens and padded grid size."""

        tokens, metadata = self.prepare_tokens_with_metadata(x=x, masks=masks)
        return tokens, metadata.grid_size

    def execute_block(
        self,
        block_index: int,
        tokens: Tensor,
        grid: LightlyViTGridMetadata | tuple[int, int],
        *,
        num_special_tokens: int | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Execute a numbered block with any caller-defined special prefix.

        This is the small backbone-adapter seam used by EoMT.  Local blocks
        spatially mix only the suffix after ``num_special_tokens``; global blocks
        may update every token through full attention.
        """

        if block_index < 0 or block_index >= self.num_blocks:
            raise IndexError(
                f"block_index must be in [0, {self.num_blocks}), got {block_index}."
            )
        if num_special_tokens is None:
            num_special_tokens = 1 + self.num_register_tokens

        if isinstance(grid, LightlyViTGridMetadata):
            grid_size = grid.grid_size
            valid_token_mask = grid.valid_token_mask
        else:
            grid_size = grid
            valid_token_mask = None

        block = self.blocks[block_index]
        rope: tuple[Tensor, Tensor] | None = None
        if block.is_global:
            key_valid_mask = self._prefix_key_valid_mask(
                tokens=tokens,
                num_special_tokens=num_special_tokens,
                grid_size=grid_size,
                valid_token_mask=valid_token_mask,
            )
            attention_mask = _combine_attention_masks(
                key_valid_mask=key_valid_mask,
                attention_mask=attention_mask,
            )
            if self.pos_encoding == "rope":
                # Computed inside the (possibly checkpointed) block call so
                # every execution path — forward_features, intermediate-layer
                # extraction, and the EoMT adapter seam — stays RoPE-correct
                # without caller changes. Non-reentrant checkpointing
                # preserves RNG state, so the train-time rescale_coords draw
                # replays identically on recompute.
                rope = self.rope_embed(H=grid_size[0], W=grid_size[1])
        return block(  # type: ignore[no-any-return]
            tokens,
            grid_size=grid_size,
            num_special_tokens=num_special_tokens,
            valid_token_mask=valid_token_mask,
            attention_mask=attention_mask,
            rope=rope,
        )

    def normalize_tokens(self, tokens: Tensor) -> Tensor:
        """Normalize a complete token sequence with the backbone output norm."""

        return self.norm(tokens)  # type: ignore[no-any-return]

    def _run_block(
        self,
        block_index: int,
        tokens: Tensor,
        metadata: LightlyViTGridMetadata,
    ) -> Tensor:
        """Run one block, activation-checkpointed when enabled on the wrapper."""

        return maybe_checkpoint(
            self.execute_block,
            block_index,
            tokens,
            metadata,
            num_special_tokens=1 + self.num_register_tokens,
            # getattr defaults keep instances unpickled from checkpoints saved
            # before the activation-checkpointing attributes existed working.
            use_activation_checkpointing=getattr(
                self, "_activation_checkpointing", False
            ),
            block_index=block_index,
            every_n_blocks=getattr(self, "_activation_checkpointing_every_n_blocks", 1),
        )

    def forward_features(
        self, x: Tensor, masks: Tensor | None = None
    ) -> LightlyViTBackboneOutput:
        """Return normalized typed backbone features for an image batch."""

        tokens, metadata = self.prepare_tokens_with_metadata(x=x, masks=masks)
        for block_index in range(self.num_blocks):
            tokens = self._run_block(block_index, tokens, metadata)
        return self._make_output(tokens=tokens, metadata=metadata, normalize=True)

    def forward_intermediate_features(
        self,
        x: Tensor,
        layer_indices: Sequence[int],
        masks: Tensor | None = None,
    ) -> list[LightlyViTBackboneOutput]:
        """Return normalized features at arbitrary zero-based block indices.

        Results preserve the caller's requested order, including a repeated
        layer index, which makes this safer than relying on execution order for
        dense consumers.
        """

        indices = self._validate_layer_indices(layer_indices)
        tokens, metadata = self.prepare_tokens_with_metadata(x=x, masks=masks)
        requested = set(indices)
        outputs: dict[int, Tensor] = {}
        for block_index in range(self.num_blocks):
            tokens = self._run_block(block_index, tokens, metadata)
            if block_index in requested:
                outputs[block_index] = tokens
        return [
            self._make_output(tokens=outputs[index], metadata=metadata, normalize=True)
            for index in indices
        ]

    @overload
    def get_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: Literal[False] = False,
        return_extra_tokens: Literal[False] = False,
        norm: bool = True,
        masks: Tensor | None = None,
    ) -> tuple[Tensor, ...]: ...

    @overload
    def get_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: Literal[True],
        return_extra_tokens: Literal[False] = False,
        norm: bool = True,
        masks: Tensor | None = None,
    ) -> tuple[tuple[Tensor, Tensor], ...]: ...

    @overload
    def get_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: Literal[False] = False,
        return_extra_tokens: Literal[True],
        norm: bool = True,
        masks: Tensor | None = None,
    ) -> tuple[tuple[Tensor, Tensor], ...]: ...

    @overload
    def get_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: Literal[True],
        return_extra_tokens: Literal[True],
        norm: bool = True,
        masks: Tensor | None = None,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], ...]: ...

    def get_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
        masks: Tensor | None = None,
    ) -> tuple[Tensor | tuple[Tensor, ...], ...]:
        """DINO-compatible arbitrary intermediate-layer extraction.

        When ``n`` is an integer, it selects the final ``n`` blocks.  A sequence
        selects explicit zero-based block indices in exactly the supplied order.
        """

        if isinstance(n, int):
            if n < 1 or n > self.num_blocks:
                raise ValueError(f"n must be in [1, {self.num_blocks}], got {n}.")
            layer_indices: Sequence[int] = tuple(
                range(self.num_blocks - n, self.num_blocks)
            )
        else:
            layer_indices = n
        indices = self._validate_layer_indices(layer_indices)

        tokens, metadata = self.prepare_tokens_with_metadata(x=x, masks=masks)
        requested = set(indices)
        token_outputs: dict[int, Tensor] = {}
        for block_index in range(self.num_blocks):
            tokens = self._run_block(block_index, tokens, metadata)
            if block_index in requested:
                token_outputs[block_index] = (
                    self.normalize_tokens(tokens) if norm else tokens
                )

        outputs: list[Tensor | tuple[Tensor, ...]] = []
        grid_height, grid_width = metadata.grid_size
        prefix_size = 1 + self.num_register_tokens
        for index in indices:
            token_output = token_outputs[index]
            patch_output = token_output[:, prefix_size:]
            if reshape:
                patch_output = (
                    patch_output.reshape(
                        patch_output.shape[0],
                        grid_height,
                        grid_width,
                        self.embed_dim,
                    )
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            cls_output = token_output[:, 0]
            register_output = token_output[:, 1:prefix_size]
            if return_class_token and return_extra_tokens:
                outputs.append((patch_output, cls_output, register_output))
            elif return_class_token:
                outputs.append((patch_output, cls_output))
            elif return_extra_tokens:
                outputs.append((patch_output, register_output))
            else:
                outputs.append(patch_output)
        return tuple(outputs)

    def forward(
        self,
        x: Tensor,
        masks: Tensor | None = None,
        *,
        is_training: bool = False,
    ) -> Tensor | LightlyViTBackboneOutput:
        """Return CLS features by default, or the typed output for SSL callers."""

        output = self.forward_features(x=x, masks=masks)
        if is_training:
            return output
        return self.head(output.cls_token)  # type: ignore[no-any-return]

    def _make_output(
        self,
        tokens: Tensor,
        metadata: LightlyViTGridMetadata,
        *,
        normalize: bool,
    ) -> LightlyViTBackboneOutput:
        prenorm_tokens = tokens
        if normalize:
            tokens = self.normalize_tokens(tokens)
        prefix_size = 1 + self.num_register_tokens
        patch_tokens = tokens[:, prefix_size:]
        grid_height, grid_width = metadata.grid_size
        features = (
            patch_tokens.reshape(
                patch_tokens.shape[0], grid_height, grid_width, self.embed_dim
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return LightlyViTBackboneOutput(
            patch_tokens=patch_tokens,
            features=features,
            cls_token=tokens[:, 0],
            register_tokens=tokens[:, 1:prefix_size],
            valid_token_mask=metadata.valid_token_mask,
            grid_metadata=metadata,
            prenorm_tokens=prenorm_tokens,
        )

    def _normalize_patch_mask(
        self,
        masks: Tensor | None,
        batch_size: int,
        grid_size: tuple[int, int],
        device: torch.device,
    ) -> Tensor | None:
        if masks is None:
            return None
        if masks.shape[0] != batch_size:
            raise ValueError(
                "Mask batch dimension does not match input: "
                f"{masks.shape[0]} != {batch_size}."
            )
        grid_height, grid_width = grid_size
        if masks.ndim == 3:
            if tuple(masks.shape[1:]) != grid_size:
                raise ValueError(
                    "A 3D mask must match the padded patch grid, got "
                    f"{tuple(masks.shape[1:])} instead of {grid_size}."
                )
            masks = masks.reshape(batch_size, -1)
        elif masks.ndim == 2:
            if masks.shape[1] != grid_height * grid_width:
                raise ValueError(
                    "A 2D mask must have one entry per padded patch, got "
                    f"{masks.shape[1]} instead of {grid_height * grid_width}."
                )
        else:
            raise ValueError("masks must have shape B x N or B x H_p x W_p.")
        return masks.to(device=device, dtype=torch.bool)

    def _validate_layer_indices(self, layer_indices: Sequence[int]) -> tuple[int, ...]:
        indices = tuple(int(index) for index in layer_indices)
        if not indices:
            raise ValueError("At least one layer index must be requested.")
        invalid = tuple(
            index for index in indices if index < 0 or index >= self.num_blocks
        )
        if invalid:
            raise ValueError(
                f"Layer indices must be in [0, {self.num_blocks}), got {invalid}."
            )
        return indices

    def _prefix_key_valid_mask(
        self,
        tokens: Tensor,
        num_special_tokens: int,
        grid_size: tuple[int, int],
        valid_token_mask: Tensor | None,
    ) -> Tensor | None:
        if valid_token_mask is None:
            return None
        expected_patch_tokens = grid_size[0] * grid_size[1]
        if tokens.shape[1] - num_special_tokens != expected_patch_tokens:
            raise ValueError(
                "Token sequence does not contain the expected patch suffix for "
                "the supplied grid."
            )
        prefix = torch.ones(
            tokens.shape[0],
            num_special_tokens,
            device=tokens.device,
            dtype=torch.bool,
        )
        return torch.cat([prefix, valid_token_mask.reshape(tokens.shape[0], -1)], dim=1)


def lightlyvit_t16(
    *,
    in_chans: int = 3,
    num_input_channels: int | None = None,
    image_size: int | tuple[int, int] = 224,
    patch_size: int = 16,
    drop_path_rate: float = 0.0,
    **kwargs: object,
) -> LightlyViT:
    """Construct the public LightlyViT v1 Tiny model, T/16.

    Identical construction contract to :func:`lightlyvit_s16`: ``image_size``
    only selects the learned-position base grid, and the token width, depth,
    heads, register count, and attention schedule are fixed. Like M/16, the
    Tiny model defaults to the **linear** global mixer: T/16 is the robotics
    deploy tier (distilled in the v7 16:10 campaign) and never had a
    softmax-only public life, so defaulting to linear keeps its checkpoints
    loadable into the default-constructed model.
    """

    if num_input_channels is not None:
        if in_chans != 3 and in_chans != num_input_channels:
            raise ValueError(
                "in_chans and num_input_channels disagree: "
                f"{in_chans} != {num_input_channels}."
            )
        in_chans = num_input_channels
    if patch_size != 16:
        raise ValueError(
            f"lightlyvit_t16 has a fixed patch size of 16, got {patch_size}."
        )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "lightlyvit_t16 only accepts image/input-channel/drop-path arguments; "
            f"unexpected arguments: {unexpected}."
        )
    spec = _LIGHTLYVIT_T16_SPEC
    return LightlyViT(
        image_size=image_size,
        patch_size=16,
        in_chans=in_chans,
        embed_dim=spec.width,
        depth=spec.depth,
        num_heads=spec.num_heads,
        mlp_ratio=4.0,
        num_register_tokens=4,
        global_block_indices=spec.global_block_indices,
        global_mixer_kind="linear",
        pos_encoding="rope",
        drop_path_rate=drop_path_rate,
    )


def lightlyvit_s16(
    *,
    in_chans: int = 3,
    num_input_channels: int | None = None,
    image_size: int | tuple[int, int] = 224,
    patch_size: int = 16,
    drop_path_rate: float = 0.0,
    **kwargs: object,
) -> LightlyViT:
    """Construct the public LightlyViT Small model, S/16.

    ``image_size`` only selects the position-encoding base grid; positions are
    interpolated for every actual input shape.  The token width, depth, heads,
    register count, and attention schedule are deliberately fixed.  S/16 ships
    with the linear global mixer and RoPE position encoding: this is the
    recipe the released RoboDINO-S checkpoints were trained with.
    """

    if num_input_channels is not None:
        if in_chans != 3 and in_chans != num_input_channels:
            raise ValueError(
                "in_chans and num_input_channels disagree: "
                f"{in_chans} != {num_input_channels}."
            )
        in_chans = num_input_channels
    if patch_size != 16:
        raise ValueError(
            f"lightlyvit_s16 has a fixed patch size of 16, got {patch_size}."
        )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "lightlyvit_s16 only accepts image/input-channel/drop-path arguments; "
            f"unexpected arguments: {unexpected}."
        )
    spec = _LIGHTLYVIT_S16_SPEC
    return LightlyViT(
        image_size=image_size,
        patch_size=16,
        in_chans=in_chans,
        embed_dim=spec.width,
        depth=spec.depth,
        num_heads=spec.num_heads,
        mlp_ratio=4.0,
        num_register_tokens=4,
        global_block_indices=spec.global_block_indices,
        global_mixer_kind="linear",
        pos_encoding="rope",
        drop_path_rate=drop_path_rate,
    )


def lightlyvit_m16(
    *,
    in_chans: int = 3,
    num_input_channels: int | None = None,
    image_size: int | tuple[int, int] = 224,
    patch_size: int = 16,
    drop_path_rate: float = 0.0,
    **kwargs: object,
) -> LightlyViT:
    """Construct the public LightlyViT Medium model, M/16.

    Identical construction contract to :func:`lightlyvit_s16`: ``image_size``
    only selects the position-encoding base grid, and the token width, depth,
    heads, register count, and attention schedule are fixed.  M/16 ships with
    the linear global mixer and RoPE position encoding: this is the recipe the
    released RoboDINO-M checkpoints were trained with.
    """

    if num_input_channels is not None:
        if in_chans != 3 and in_chans != num_input_channels:
            raise ValueError(
                "in_chans and num_input_channels disagree: "
                f"{in_chans} != {num_input_channels}."
            )
        in_chans = num_input_channels
    if patch_size != 16:
        raise ValueError(
            f"lightlyvit_m16 has a fixed patch size of 16, got {patch_size}."
        )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            "lightlyvit_m16 only accepts image/input-channel/drop-path arguments; "
            f"unexpected arguments: {unexpected}."
        )
    spec = _LIGHTLYVIT_M16_SPEC
    return LightlyViT(
        image_size=image_size,
        patch_size=16,
        in_chans=in_chans,
        embed_dim=spec.width,
        depth=spec.depth,
        num_heads=spec.num_heads,
        mlp_ratio=4.0,
        num_register_tokens=4,
        global_block_indices=spec.global_block_indices,
        global_mixer_kind="linear",
        pos_encoding="rope",
        drop_path_rate=drop_path_rate,
    )


def _combine_attention_masks(
    key_valid_mask: Tensor | None, attention_mask: Tensor | None
) -> Tensor | None:
    """Combine a default key-validity mask with an optional caller mask."""

    if key_valid_mask is None:
        return attention_mask
    if attention_mask is None:
        return key_valid_mask
    if attention_mask.dtype == torch.bool:
        if attention_mask.ndim == 2:
            return attention_mask & key_valid_mask
        return attention_mask & key_valid_mask[:, None, None, :]

    invalid_bias = torch.zeros_like(key_valid_mask, dtype=attention_mask.dtype)
    invalid_bias = invalid_bias.masked_fill(~key_valid_mask, float("-inf"))
    if attention_mask.ndim == 2:
        return attention_mask + invalid_bias
    return attention_mask + invalid_bias[:, None, None, :]


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _ceil_grid_size(image_size: tuple[int, int], patch_size: int) -> tuple[int, int]:
    return tuple(
        _ceil_to_multiple(value, patch_size) // patch_size for value in image_size
    )  # type: ignore[return-value]


def _to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected a two-dimensional image size, got {value}.")
    return int(value[0]), int(value[1])


def _init_module_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
