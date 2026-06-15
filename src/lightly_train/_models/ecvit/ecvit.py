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
# limitations under the License.
#
"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.

Modified from https://huggingface.co/spaces/Hila/RobustViT/blob/main/ViT/ViT_new.py

# Modifications Copyright 2026 Lightly AG:
- Ported the ECViT backbone adapter to Lightly.
- Removed EdgeCrafter registry/distributed dependencies.
- Added typed LTDETR-compatible tuple output.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lightly_train._models.dinov3.dinov3_src.layers.ffn_layers import Mlp
from lightly_train._models.dinov3.dinov3_src.layers.rope_position_encoding import (
    RopePositionEmbedding,
)
from lightly_train._models.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
)
from lightly_train._task_models.object_detection_components.hybrid_encoder import (
    ConvNormLayer,
)
from lightly_train.types import PathLike

_DEFAULT = object()


ECVIT_PRETRAINED_URLS: dict[str, str] = {
    "ecvitt": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvitt.pth",
    "ecvittplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvittplus.pth",
    "ecvits": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvits.pth",
    "ecvitsplus": "https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecvitsplus.pth",
}


ECVIT_PRESETS: dict[str, dict[str, int | None | float]] = {
    "ecvitt": {
        "embed_dim": 192,
        "num_heads": 3,
        "proj_dim": None,
        "ffn_ratio": 4.0,
    },
    "ecvittplus": {
        "embed_dim": 256,
        "num_heads": 4,
        "proj_dim": None,
        "ffn_ratio": 4.0,
    },
    "ecvits": {
        "embed_dim": 384,
        "num_heads": 6,
        "proj_dim": 256,
        "ffn_ratio": 4.0,
    },
    "ecvitsplus": {
        "embed_dim": 384,
        "num_heads": 6,
        "proj_dim": 256,
        "ffn_ratio": 6.0,
    },
}


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class ConvPyramidPatchEmbed(nn.Module):
    def __init__(self, embed_dim: int = 192, patch_size: int = 16, act: str = "relu"):
        super().__init__()

        if patch_size != 16:
            raise NotImplementedError(
                "Only support patch_size=16 for ConvPyramidPatchEmbed."
            )
        num_stages = int(math.log2(patch_size)) - 1
        ratios = [2**i for i in range(num_stages, 0, -1)]
        channels = [embed_dim // r for r in ratios]

        self.convs = nn.ModuleList(
            [
                ConvNormLayer(  # type: ignore[no-untyped-call]
                    in_ch, out_ch, 3, 2, act=act
                )
                for in_ch, out_ch in zip([3] + channels[:-1], channels)
            ]
        )

        self.proj = nn.Conv2d(
            channels[-1], embed_dim, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.proj(x))


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(
    tensor: Tensor, mean: float, std: float, a: float, b: float
) -> Tensor:
    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: Tensor, rope_sincos: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope_sincos is not None:
            sin, cos = rope_sincos
            q_cls, q_patch = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patch = k[:, :, :1, :], k[:, :, 1:, :]

            q_patch = apply_rope(q_patch, sin, cos)
            k_patch = apply_rope(k_patch, sin, cos)

            q = torch.cat((q_cls, q_patch), dim=2)
            k = torch.cat((k_cls, k_patch), dim=2)

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop
        )
        x = x.transpose(1, 2).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
        norm_layer: type[nn.Module] | partial[Any] = nn.LayerNorm,
        ffn_layer: type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * ffn_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self, x: Tensor, rope_sincos: tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        attn_output = self.attn(self.norm1(x), rope_sincos=rope_sincos)
        x = x + self.drop_path(attn_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


EMBED_LAYER_REGISTRY: dict[str, type[nn.Module]] = {
    "ConvPyramidPatchEmbed": ConvPyramidPatchEmbed,
    "PatchEmbed": PatchEmbed,
}

FFN_LAYER_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": Mlp,
}


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
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
        return_layers: list[int] | None = None,
        embed_layer: type[nn.Module] = ConvPyramidPatchEmbed,
        norm_layer: type[nn.Module] | partial[Any] | None = None,
        act_layer: type[nn.Module] | None = None,
        ffn_layer: type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.return_layers = return_layers or [3, 7, 11]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # Preserve EdgeCrafter behavior: GELU is used for transformer MLPs.
        act_layer = nn.GELU

        patch_embed: nn.Module
        if embed_layer == PatchEmbed:
            patch_embed = embed_layer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            patch_embed = embed_layer(embed_dim=embed_dim, patch_size=patch_size)
        self.patch_embed: nn.Module = patch_embed
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
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=ffn_layer,
                )
                for i in range(depth)
            ]
        )

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100.0,
            normalize_coords="separate",
            shift_coords=None,
            jitter_coords=None,
            rescale_coords=None,
            dtype=None,
            device=None,
        )
        self.init_weights()

    def init_weights(self) -> None:
        self.apply(self._init_vit_weights)
        cast(Any, self.rope_embed)._init_weights()
        trunc_normal_(self.register_token, std=0.02)

    def _init_vit_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_with_grid(self, x: Tensor) -> tuple[list[Tensor], tuple[int, int]]:
        outs = []
        x_embed = cast(Tensor, self.patch_embed(x))
        _, _, H, W = x_embed.shape

        x_embed = x_embed.flatten(2).transpose(1, 2)
        register_token = self.register_token.expand(x_embed.shape[0], -1, -1)
        x = torch.cat((register_token, x_embed), dim=1)
        sin, cos = self.rope_embed(H=H, W=W)
        sin = sin.to(device=x_embed.device, dtype=x_embed.dtype)
        cos = cos.to(device=x_embed.device, dtype=x_embed.dtype)
        rope_sincos = sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

        for i, blk in enumerate(self.blocks):
            x = blk(x, rope_sincos=rope_sincos)
            if i in self.return_layers:
                outs.append(x[:, 1:])
        return outs, (H, W)

    def forward(self, x: Tensor) -> list[Tensor]:
        outs, _ = self.forward_with_grid(x)
        return outs


class ECViTModelWrapper(nn.Module, ModelWrapper, ArchitectureInfoGettable):
    """EdgeCrafter ECViT backbone wrapper for LTDETR-style feature pyramids.

    The forward path intentionally follows EdgeCrafter's ECViT adapter:
    selected ECViT token outputs are averaged, reshaped to a spatial map,
    interpolated to three levels, projected, and returned as ``(P3, P4, P5)``.
    """

    def __init__(
        self,
        name: str,
        weights_path: PathLike | None = None,
        interaction_indexes: list[int] | None = None,
        embed_dim: int | object = _DEFAULT,
        num_heads: int | object = _DEFAULT,
        patch_size: int = 16,
        proj_dim: int | None | object = _DEFAULT,
        num_levels: int = 3,
        embed_layer: str = "ConvPyramidPatchEmbed",
        ffn_layer: str = "mlp",
        ffn_ratio: float | object = _DEFAULT,
        skip_load_backbone: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if name not in ECVIT_PRESETS:
            raise ValueError(
                f"Unknown ECViT model name: {name}. Available: {list(ECVIT_PRESETS)}"
            )
        if embed_layer not in EMBED_LAYER_REGISTRY:
            raise ValueError(
                f"Unknown embed_layer: {embed_layer}. "
                f"Available: {list(EMBED_LAYER_REGISTRY)}"
            )
        if ffn_layer not in FFN_LAYER_REGISTRY:
            raise ValueError(
                f"Unknown ffn_layer: {ffn_layer}. Available: {list(FFN_LAYER_REGISTRY)}"
            )
        if num_levels != 3:
            raise NotImplementedError(
                "Only support num_levels=3 for ECViTModelWrapper."
            )

        preset = ECVIT_PRESETS[name]
        resolved_embed_dim = cast(
            int, preset["embed_dim"] if embed_dim is _DEFAULT else embed_dim
        )
        resolved_num_heads = cast(
            int, preset["num_heads"] if num_heads is _DEFAULT else num_heads
        )
        if proj_dim is _DEFAULT:
            resolved_proj_dim = preset["proj_dim"]
            if resolved_proj_dim is not None and not isinstance(resolved_proj_dim, int):
                raise TypeError("Preset proj_dim must be an int or None.")
        else:
            if proj_dim is not None and not isinstance(proj_dim, int):
                raise TypeError("proj_dim must be an int or None.")
            resolved_proj_dim = proj_dim
        resolved_ffn_ratio = cast(
            float, preset["ffn_ratio"] if ffn_ratio is _DEFAULT else ffn_ratio
        )
        resolved_interaction_indexes = interaction_indexes or [10, 11]

        self.name = name
        self.interaction_indexes = resolved_interaction_indexes
        self.patch_size = patch_size
        self.num_levels = num_levels
        self.embed_dim = resolved_embed_dim
        self.num_heads = resolved_num_heads
        self.proj_dim = (
            [resolved_proj_dim] * num_levels
            if resolved_proj_dim is not None
            else [resolved_embed_dim]
        )

        self.backbone = VisionTransformer(
            embed_dim=resolved_embed_dim,
            num_heads=resolved_num_heads,
            return_layers=resolved_interaction_indexes,
            patch_size=patch_size,
            embed_layer=EMBED_LAYER_REGISTRY[embed_layer],
            ffn_layer=FFN_LAYER_REGISTRY[ffn_layer],
            ffn_ratio=resolved_ffn_ratio,
            **kwargs,
        )

        if weights_path is not None and not skip_load_backbone:
            self._load_backbone_weights(weights_path=weights_path)

        self.projector = nn.ModuleList(
            [
                ConvNormLayer(  # type: ignore[no-untyped-call]
                    resolved_embed_dim, dim, kernel_size=1, stride=1
                )
                for dim in self.proj_dim
            ]
        )
        self._pool = nn.AdaptiveAvgPool2d((1, 1))

    @property
    def backbone_model(self) -> nn.Module:
        return self.backbone

    def _load_backbone_weights(self, weights_path: PathLike) -> None:
        state = _load_torch_checkpoint(Path(weights_path))
        state_dict = _unwrap_state_dict(state)
        self.backbone.load_state_dict(state_dict, strict=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        bs = x.shape[0]

        return_layers, (H_c, W_c) = self.backbone.forward_with_grid(x)
        if len(return_layers) == 0:
            raise RuntimeError(
                "ECViT backbone returned no layers. Check interaction_indexes."
            )

        fused_feats = torch.mean(torch.stack(return_layers), dim=0)
        fused_feats = fused_feats.transpose(1, 2).contiguous().view(bs, -1, H_c, W_c)

        proj_feats = []
        for i in range(self.num_levels):
            scale = 2 ** (1 - i)
            resize_H = max(1, int(H_c * scale))
            resize_W = max(1, int(W_c * scale))
            feature = F.interpolate(
                fused_feats,
                size=[resize_H, resize_W],
                mode="bilinear",
                align_corners=False,
            )
            proj_feats.append(feature)

        if len(self.projector) == 1:
            proj_feats[-1] = self.projector[-1](proj_feats[-1])
        else:
            proj_feats = [
                layer(feat) for layer, feat in zip(self.projector, proj_feats)
            ]

        if len(proj_feats) != 3:
            raise RuntimeError(
                f"Expected 3 ECViT feature levels, got {len(proj_feats)}."
            )
        return proj_feats[0], proj_feats[1], proj_feats[2]

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        return {"features": self.forward(x)[-1]}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def feature_dim(self) -> int:
        return self.proj_dim[-1]

    def get_model(self) -> "ECViTModelWrapper":
        return self

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "transformer", "norm_type": "layernorm"}


def _load_torch_checkpoint(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"ECViT weights_path does not exist: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_tensor_state_dict(state: object) -> Mapping[str, Tensor] | None:
    if not isinstance(state, Mapping):
        return None

    tensor_state_dict: dict[str, Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            return None
        tensor_state_dict[key] = value
    return tensor_state_dict


def _unwrap_state_dict(state: object) -> Mapping[str, Tensor]:
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a state dict mapping, got {type(state)!r}.")

    for key in ("state_dict", "model", "backbone"):
        maybe_state_dict = state.get(key)
        state_dict = _to_tensor_state_dict(maybe_state_dict)
        if state_dict is not None:
            return state_dict

    state_dict = _to_tensor_state_dict(state)
    if state_dict is not None:
        return state_dict

    raise TypeError("Expected checkpoint to contain a tensor state dict.")
