#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Portions of this file are based on Depth Anything 3:
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# Licensed under the Apache License, Version 2.0.
#
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lightly_train._models.dinov2_vit.dinov2_vit_src.layers import (
    MemEffAttention,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.layers import (
    NestedTensorBlock as Block,
)
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)

_BACKBONE_SPECS: dict[str, dict[str, int]] = {
    "vits": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "vitb": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "vitl": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "vitg": {"embed_dim": 1536, "depth": 40, "num_heads": 24},
}


class DepthAnythingV3MonoNet(nn.Module):
    """Depth Anything V3 monocular relative-depth network.

    The defaults match the official ``da3mono-large.yaml`` configuration:
    ViT-L/14 features from blocks 4, 11, 17, and 23 followed by the DA3 DPT head.
    The module expects images with shape ``(B, 1, 3, H, W)`` and returns a dict with
    ``depth`` and, when enabled, ``sky`` tensors of shape ``(B, 1, H, W)``.
    """

    def __init__(
        self,
        *,
        backbone_name: str = "vitl",
        out_layers: Sequence[int] = (4, 11, 17, 23),
        image_size: int = 518,
        patch_size: int = 14,
        dim_in: int | None = None,
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        output_dim: int = 1,
        use_sky_head: bool = True,
        backbone_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        spec = _resolve_backbone_spec(backbone_name, backbone_args=backbone_args)
        dim_in = spec["embed_dim"] if dim_in is None else dim_in

        self.backbone = _DepthAnythingV3DINOv2Backbone(
            backbone_name=backbone_name,
            out_layers=tuple(out_layers),
            image_size=image_size,
            patch_size=patch_size,
            backbone_args=spec,
        )
        self.head = _DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim,
            features=features,
            out_channels=tuple(out_channels),
            use_sky_head=use_sky_head,
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 5:
            raise ValueError(
                f"Expected input shape (B, 1, C, H, W), got {tuple(x.shape)}."
            )
        feats, _ = self.backbone(x)
        out = self.head(feats=feats, H=x.shape[-2], W=x.shape[-1], patch_start_idx=0)
        _set_mono_sky_regions_to_max_depth(out)
        return out


class _DepthAnythingV3DINOv2Backbone(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str,
        out_layers: tuple[int, ...],
        image_size: int,
        patch_size: int,
        backbone_args: dict[str, int],
    ) -> None:
        super().__init__()
        self.out_layers = out_layers
        self.pretrained = _make_dinov2_backbone(
            backbone_name=backbone_name,
            image_size=image_size,
            patch_size=patch_size,
            **backbone_args,
        )

    def forward(
        self,
        x: Tensor,
        **_: Any,
    ) -> tuple[list[tuple[Tensor, Tensor]], list[Tensor]]:
        b, s, c, h, w = x.shape
        x_flat = x.reshape(b * s, c, h, w)
        intermediate = self.pretrained.get_intermediate_layers(
            x_flat,
            n=self.out_layers,
            reshape=False,
            return_class_token=True,
            norm=True,
        )

        feats: list[tuple[Tensor, Tensor]] = []
        for patch_tokens, class_token in intermediate:
            _, num_tokens, dim = patch_tokens.shape
            feats.append(
                (
                    patch_tokens.reshape(b, s, num_tokens, dim),
                    class_token.reshape(b, s, dim),
                )
            )
        return feats, []


class _DPTHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = False,
        down_ratio: int = 1,
        head_name: str = "depth",
        use_sky_head: bool = True,
        sky_name: str = "sky",
        sky_activation: str = "relu",
    ) -> None:
        super().__init__()
        if len(out_channels) != 4:
            raise ValueError("DA3 DPT expects exactly four feature stages.")

        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.head_main = head_name
        self.sky_name = sky_name
        self.out_dim = output_dim
        self.has_conf = output_dim > 1
        self.use_sky_head = use_sky_head
        self.sky_activation = sky_activation
        self.intermediate_layer_idx = (0, 1, 2, 3)

        self.norm = nn.Identity()
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0],
                    out_channels[0],
                    kernel_size=4,
                    stride=4,
                ),
                nn.ConvTranspose2d(
                    out_channels[1],
                    out_channels[1],
                    kernel_size=2,
                    stride=2,
                ),
                nn.Identity(),
                nn.Conv2d(
                    out_channels[3],
                    out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.scratch = _make_scratch(tuple(out_channels), features)
        self.scratch.refinenet1 = _FeatureFusionBlock(features)
        self.scratch.refinenet2 = _FeatureFusionBlock(features)
        self.scratch.refinenet3 = _FeatureFusionBlock(features)
        self.scratch.refinenet4 = _FeatureFusionBlock(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1,
            head_features_1 // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1),
        )
        if self.use_sky_head:
            self.scratch.sky_output_conv2 = nn.Sequential(
                nn.Conv2d(
                    head_features_1 // 2,
                    head_features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, 1, kernel_size=1),
            )

    def forward(
        self,
        feats: list[tuple[Tensor, Tensor]],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> dict[str, Tensor]:
        b, s, num_tokens, dim = feats[0][0].shape
        feats_flat = [feat[0].reshape(b * s, num_tokens, dim) for feat in feats]
        out = self._forward_impl(
            feats=feats_flat,
            H=H,
            W=W,
            patch_start_idx=patch_start_idx,
        )
        return {key: value.reshape(b, s, *value.shape[1:]) for key, value in out.items()}

    def _forward_impl(
        self,
        feats: list[Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> dict[str, Tensor]:
        batch_size, _, channels = feats[0].shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(batch_size, channels, patch_h, patch_w)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W=W, H=H)
            resized_feats.append(self.resize_layers[stage_idx](x))

        fused = self._fuse(resized_feats)
        h_out = int(patch_h * self.patch_size / self.down_ratio)
        w_out = int(patch_w * self.patch_size / self.down_ratio)

        fused = self.scratch.output_conv1(fused)
        fused = _custom_interpolate(
            fused,
            size=(h_out, w_out),
            mode="bilinear",
            align_corners=True,
        )
        if self.pos_embed:
            fused = self._add_pos_embed(fused, W=W, H=H)

        main_logits = self.scratch.output_conv2(fused)
        out: dict[str, Tensor] = {}
        if self.has_conf:
            fmap = main_logits.permute(0, 2, 3, 1)
            pred = _apply_activation(fmap[..., :-1], self.activation)
            conf = _apply_activation(fmap[..., -1], self.conf_activation)
            out[self.head_main] = pred.squeeze(-1)
            out[f"{self.head_main}_conf"] = conf
        else:
            out[self.head_main] = _apply_activation(
                main_logits, self.activation
            ).squeeze(1)

        if self.use_sky_head:
            sky_logits = self.scratch.sky_output_conv2(fused)
            out[self.sky_name] = _apply_activation(
                sky_logits, self.sky_activation
            ).squeeze(1)
        return out

    def _fuse(self, feats: list[Tensor]) -> Tensor:
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        return self.scratch.refinenet1(out, l1_rn)

    def _add_pos_embed(self, x: Tensor, W: int, H: int) -> Tensor:
        pos = _create_uv_grid(
            width=x.shape[-1],
            height=x.shape[-2],
            aspect_ratio=W / H,
            dtype=x.dtype,
            device=x.device,
        )
        pos_embed = _position_grid_to_embed(pos, embed_dim=x.shape[1]) * 0.1
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed


class _ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.activation = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class _FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, has_residual: bool = True) -> None:
        super().__init__()
        self.has_residual = has_residual
        self.resConfUnit1 = _ResidualConvUnit(features) if has_residual else None
        self.resConfUnit2 = _ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)

    def forward(self, *xs: Tensor, size: tuple[int, int] | None = None) -> Tensor:
        out = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            out = out + self.resConfUnit1(xs[1])
        out = self.resConfUnit2(out)
        if size is None:
            out = _custom_interpolate(
                out, scale_factor=2.0, mode="bilinear", align_corners=True
            )
        else:
            out = _custom_interpolate(
                out, size=size, mode="bilinear", align_corners=True
            )
        return self.out_conv(out)


def _make_dinov2_backbone(
    *,
    backbone_name: str,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
) -> DinoVisionTransformer:
    if backbone_name not in _BACKBONE_SPECS and backbone_name != "custom":
        raise ValueError(f"Unsupported DA3 DINOv2 backbone '{backbone_name}'.")
    return DinoVisionTransformer(
        img_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=0,
        init_values=1.0,
        block_chunks=0,
    )


def _resolve_backbone_spec(
    backbone_name: str, backbone_args: dict[str, Any] | None
) -> dict[str, int]:
    spec = dict(_BACKBONE_SPECS.get(backbone_name, {}))
    if backbone_args is not None:
        spec.update(backbone_args)
    if not {"embed_dim", "depth", "num_heads"}.issubset(spec):
        raise ValueError(
            "backbone_args must define 'embed_dim', 'depth', and 'num_heads' "
            "when using a custom DA3 backbone."
        )
    return {key: int(spec[key]) for key in ("embed_dim", "depth", "num_heads")}


def _make_scratch(in_shape: Sequence[int], out_shape: int) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(  # type: ignore[attr-defined]
        in_shape[0], out_shape, kernel_size=3, padding=1, bias=False
    )
    scratch.layer2_rn = nn.Conv2d(  # type: ignore[attr-defined]
        in_shape[1], out_shape, kernel_size=3, padding=1, bias=False
    )
    scratch.layer3_rn = nn.Conv2d(  # type: ignore[attr-defined]
        in_shape[2], out_shape, kernel_size=3, padding=1, bias=False
    )
    scratch.layer4_rn = nn.Conv2d(  # type: ignore[attr-defined]
        in_shape[3], out_shape, kernel_size=3, padding=1, bias=False
    )
    return scratch


def _set_mono_sky_regions_to_max_depth(out: dict[str, Tensor]) -> None:
    if "depth" not in out or "sky" not in out:
        return

    non_sky_mask = out["sky"] < 0.3
    if non_sky_mask.sum() <= 10 or (~non_sky_mask).sum() <= 10:
        return

    non_sky_depth = out["depth"][non_sky_mask]
    if non_sky_depth.numel() > 100_000:
        idx = torch.randint(
            0,
            non_sky_depth.numel(),
            (100_000,),
            device=non_sky_depth.device,
        )
        non_sky_depth = non_sky_depth[idx]
    non_sky_max = torch.quantile(non_sky_depth, 0.99)

    depth = out["depth"].clone()
    depth[~non_sky_mask] = non_sky_max
    out["depth"] = depth


def _apply_activation(x: Tensor, activation: str) -> Tensor:
    if activation == "exp":
        return torch.exp(x)
    if activation == "expp1":
        return torch.exp(x) + 1
    if activation == "expm1":
        return torch.expm1(x)
    if activation == "relu":
        return torch.relu(x)
    if activation == "sigmoid":
        return torch.sigmoid(x)
    if activation == "softplus":
        return F.softplus(x)
    if activation == "tanh":
        return torch.tanh(x)
    if activation == "linear":
        return x
    raise ValueError(f"Unsupported activation '{activation}'.")


def _create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def _position_grid_to_embed(pos_grid: Tensor, embed_dim: int) -> Tensor:
    height, width, grid_dim = pos_grid.shape
    if grid_dim != 2:
        raise ValueError(f"Expected 2D position grid, got {grid_dim}D.")
    pos_flat = pos_grid.reshape(-1, grid_dim)
    emb_x = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0])
    emb_y = _make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1])
    return torch.cat([emb_x, emb_y], dim=-1).reshape(height, width, embed_dim)


def _make_sincos_pos_embed(embed_dim: int, pos: Tensor, omega_0: float = 100) -> Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("Sin/cos position embedding dimension must be even.")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / omega_0 ** (omega / (embed_dim / 2.0))
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).to(pos.dtype)


def _custom_interpolate(
    x: Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> Tensor:
    if size is None:
        if scale_factor is None:
            raise ValueError("Either size or scale_factor must be provided.")
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    int_max = 1_610_612_736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]
    if total <= int_max:
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    chunks = torch.chunk(x, chunks=(total // int_max) + 1, dim=0)
    return torch.cat(
        [
            F.interpolate(c, size=size, mode=mode, align_corners=align_corners)
            for c in chunks
        ],
        dim=0,
    ).contiguous()
