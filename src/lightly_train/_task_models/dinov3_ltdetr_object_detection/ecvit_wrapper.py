#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    ConvNormLayer,
)

_ECVITT_URL = "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvitt.pth"


class RopePositionEmbedding(nn.Module):
    def __init__(self, embed_dim: int, *, num_heads: int) -> None:
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError("Head dimension must be divisible by 4 for 2D RoPE.")
        self.head_dim = head_dim
        self.register_buffer(
            "periods",
            torch.empty(head_dim // 4),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        dtype = torch.get_default_dtype()
        self.periods.data.copy_(
            100.0 ** (2 * torch.arange(self.head_dim // 4, dtype=dtype) / (self.head_dim // 2))
        )

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.periods.dtype
        dd = {"device": device, "dtype": dtype}

        coords_h = torch.arange(0.5, H, **dd) / H
        coords_w = torch.arange(0.5, W, **dd) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).repeat(1, 2)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.SiLU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvPyramidPatchEmbed(nn.Module):
    def __init__(self, *, embed_dim: int = 192, patch_size: int = 16, in_chans: int = 3) -> None:
        super().__init__()
        if patch_size != 16:
            raise ValueError("Only patch_size=16 is supported for ECViT tiny.")

        num_stages = int(math.log2(patch_size)) - 1
        ratios = [2**i for i in range(num_stages, 0, -1)]
        channels = [embed_dim // r for r in ratios]

        self.convs = nn.ModuleList(
            [
                ConvNormLayer(in_ch, out_ch, 3, 2, act="relu")
                for in_ch, out_ch in zip([in_chans] + channels[:-1], channels)
            ]
        )
        self.proj = nn.Conv2d(channels[-1], embed_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x)
        return self.proj(x)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, rope_sincos: tuple[Tensor, Tensor] | None = None) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        if rope_sincos is not None:
            sin, cos = rope_sincos
            q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]
            q_patch = apply_rope(q_patch, sin, cos)
            k_patch = apply_rope(k_patch, sin, cos)
            q = torch.cat((q_cls, q_patch), dim=2)
            k = torch.cat((k_cls, k_patch), dim=2)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return self.proj_drop(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob is None or self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.SiLU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        ffn_layer: type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(in_features=dim, hidden_features=int(dim * ffn_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x: Tensor, rope_sincos: tuple[Tensor, Tensor] | None = None) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), rope_sincos=rope_sincos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        return_layers: list[int] | tuple[int, ...] = (10, 11),
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.return_layers = list(return_layers)
        self.patch_embed = ConvPyramidPatchEmbed(
            embed_dim=embed_dim,
            patch_size=patch_size,
            in_chans=in_chans,
        )
        self.patch_size = patch_size
        self.register_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    act_layer=nn.GELU,
                )
                for i in range(depth)
            ]
        )
        self.rope_embed = RopePositionEmbedding(embed_dim=embed_dim, num_heads=num_heads)
        self.init_weights()

    def init_weights(self) -> None:
        self.apply(self._init_vit_weights)
        self.rope_embed._init_weights()
        nn.init.trunc_normal_(self.register_token, std=0.02)

    def _init_vit_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: Tensor) -> list[Tensor]:
        outs: list[Tensor] = []
        x_embed = self.patch_embed(x)
        _, _, h, w = x_embed.shape

        x_embed = x_embed.flatten(2).transpose(1, 2)
        register_token = self.register_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((register_token, x_embed), dim=1)
        rope_sincos = self.rope_embed(H=h, W=w)

        for i, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos=rope_sincos)
            if i in self.return_layers:
                outs.append(x[:, 1:])
        return outs


class ECViTTinyAdapter(nn.Module):
    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        load_weights: bool = True,
        interaction_indexes: list[int] | tuple[int, ...] = (10, 11),
        embed_dim: int = 192,
        num_heads: int = 3,
        patch_size: int = 16,
        proj_dim: int | None = None,
        num_levels: int = 3,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if list(interaction_indexes) != [10, 11]:
            raise ValueError("Only ECViT tiny interaction indexes [10, 11] are supported.")
        if patch_size != 16:
            raise ValueError("Only ECViT tiny patch_size=16 is supported.")
        if num_levels != 3:
            raise ValueError("Only num_levels=3 is supported for ECViT tiny.")

        self.backbone = VisionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            return_layers=list(interaction_indexes),
            patch_size=patch_size,
            ffn_ratio=ffn_ratio,
        )
        self.interaction_indexes = list(interaction_indexes)
        self.patch_size = patch_size
        self.num_levels = num_levels
        self.proj_dim = [proj_dim] * num_levels if proj_dim is not None else [embed_dim]
        self.projector = nn.ModuleList(
            [ConvNormLayer(embed_dim, dim, kernel_size=1, stride=1) for dim in self.proj_dim]
        )

        if load_weights:
            self._load_weights(weights_path)

    def _load_weights(self, weights_path: str | Path | None) -> None:
        if weights_path is None:
            raise ValueError(
                "ECViT tiny requires `backbone_weights` when load_weights=True. "
                f"Reference weights: {_ECVITT_URL}"
            )
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"ECViT weights file not found: '{path}'")
        state = torch.load(path, map_location="cpu")
        self.backbone.load_state_dict(state, strict=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h_c, w_c = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs = x.shape[0]

        return_layers = self.backbone(x)
        fused_feats = torch.mean(torch.stack(return_layers), dim=0)

        proj_feats: list[Tensor] = []
        fused_feats = fused_feats.transpose(1, 2).contiguous().view(bs, -1, h_c, w_c)
        for i in range(self.num_levels):
            scale = 2 ** (1 - i)
            resize_h = int(h_c * scale)
            resize_w = int(w_c * scale)
            feature = F.interpolate(
                fused_feats,
                size=[resize_h, resize_w],
                mode="bilinear",
                align_corners=False,
            )
            proj_feats.append(feature)

        if len(self.projector) == 1:
            proj_feats[-1] = self.projector[-1](proj_feats[-1])
        else:
            proj_feats = [layer(feat) for layer, feat in zip(self.projector, proj_feats)]

        return proj_feats[0], proj_feats[1], proj_feats[2]
