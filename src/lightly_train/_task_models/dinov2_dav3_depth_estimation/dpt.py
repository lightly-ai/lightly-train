#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from lightly_train._task_models.depth_estimation_components.head_utils import (
    Permute,
    create_uv_grid,
    custom_interpolate,
    position_grid_to_embed,
)


class DPT(nn.Module):
    """
    DPT for dense prediction (main head + optional sky head, sky always 1 channel).

    Ported from the official Depth Anything V3 ``DPT``. This is the monocular-inference
    specialization: a single image, so there is no sequence/time dimension to chunk over
    and no GS ``images`` branch. The building blocks and the per-stage computation are
    kept as close to the reference as possible.

    Returns:
      - Main head:
        * If output_dim>1: { head_name, f"{head_name}_conf" }
        * If output_dim==1: { head_name }
      - Sky head (if use_sky_head=True): { sky_name }
    """

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
        # ---- sky head (fixed 1 channel) ----
        use_sky_head: bool = True,
        sky_name: str = "sky",
        sky_activation: str = "relu",  # 'sigmoid' / 'relu' / 'linear'
        use_ln_for_heads: bool = False,  # If needed, apply LayerNorm on intermediate features of both heads
        norm_type: str = "idt",  # use to match legacy GS-DPT head, "idt" / "layer"
        fusion_block_inplace: bool = False,
    ) -> None:
        super().__init__()

        # -------------------- configuration --------------------
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio

        # Names
        self.head_main = head_name
        self.sky_name = sky_name

        # Main head: output dimension and confidence switch
        self.out_dim = output_dim
        self.has_conf = output_dim > 1

        # Sky head parameters (always 1 channel)
        self.use_sky_head = use_sky_head
        self.sky_activation = sky_activation

        # Fixed 4 intermediate outputs
        self.intermediate_layer_idx: tuple[int, int, int, int] = (0, 1, 2, 3)

        # -------------------- token pre-norm + per-stage projection --------------------
        self.norm: nn.Module
        if norm_type == "layer":
            self.norm = nn.LayerNorm(dim_in)
        elif norm_type == "idt":
            self.norm = nn.Identity()
        else:
            raise Exception(
                f"Unknown norm_type {norm_type}, should be 'layer' or 'idt'."
            )
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0)
                for oc in out_channels
            ]
        )

        # -------------------- Spatial re-size (align to common scale before fusion) -----
        # Design consistent with original: relative to patch grid (x4, x2, x1, /2)
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        # -------------------- scratch: stage adapters + main fusion chain ---------------
        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Main fusion chain
        self.scratch.refinenet1 = _make_fusion_block(
            features, inplace=fusion_block_inplace
        )
        self.scratch.refinenet2 = _make_fusion_block(
            features, inplace=fusion_block_inplace
        )
        self.scratch.refinenet3 = _make_fusion_block(
            features, inplace=fusion_block_inplace
        )
        self.scratch.refinenet4 = _make_fusion_block(
            features, has_residual=False, inplace=fusion_block_inplace
        )

        # Heads (shared neck1; then split into two heads)
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )

        ln_seq: list[nn.Module] = (
            [
                Permute((0, 2, 3, 1)),
                nn.LayerNorm(head_features_2),
                Permute((0, 3, 1, 2)),
            ]
            if use_ln_for_heads
            else []
        )

        # Main head
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            *ln_seq,
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

        # Sky head (fixed 1 channel)
        if self.use_sky_head:
            self.scratch.sky_output_conv2 = nn.Sequential(
                nn.Conv2d(
                    head_features_1 // 2,
                    head_features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                *ln_seq,
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            )

    # -------------------------------------------------------------------------
    # Public forward
    # -------------------------------------------------------------------------
    def forward(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> dict[str, torch.Tensor]:
        """Monocular specialization of the official ``DPT.forward``.

        A single image, so there is no sequence/time dimension S to chunk over and no
        GS ``images`` kwarg. The DINOv2 backbone already returns only patch tokens, so
        ``patch_start_idx == 0``.

        Args:
            feats: List of 4 patch-token tensors, each ``[B, N_patch, C]``.
            H, W:  Input image dimensions.

        Returns:
            A dict of ``[B, 1, H/down_ratio, W/down_ratio]`` maps.
        """
        out = self._forward_impl(feats, H, W, patch_start_idx=0)
        # The official forward re-views [B*S, ...] back to [B, S, ...]; with S == 1 this
        # restores the channel dimension so depth/sky are [B, 1, H, W].
        return {k: v.unsqueeze(1) for k, v in out.items()}

    # -------------------------------------------------------------------------
    # Internal forward (single chunk)
    # -------------------------------------------------------------------------
    def _forward_impl(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> dict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]  # [B*S, N_patch, C]
            x = self.norm(x)
            # permute -> contiguous before reshape to keep conv input contiguous
            x = (
                x.permute(0, 2, 1).contiguous().reshape(B, C, ph, pw)
            )  # [B*S, C, ph, pw]

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)  # Align scale
            resized_feats.append(x)

        # 2) Fusion pyramid (main branch only)
        fused = self._fuse(resized_feats)

        # 3) Upsample to target resolution, optionally add position encoding again
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused = self.scratch.output_conv1(fused)
        fused = custom_interpolate(
            fused, (h_out, w_out), mode="bilinear", align_corners=True
        )
        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)

        # 4) Shared neck1
        feat = fused

        # 5) Main head: logits -> activation
        main_logits = self.scratch.output_conv2(feat)
        outs: dict[str, torch.Tensor] = {}
        if self.has_conf:
            fmap = main_logits.permute(0, 2, 3, 1)
            pred = self._apply_activation_single(fmap[..., :-1], self.activation)
            conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)
            outs[self.head_main] = pred.squeeze(1)
            outs[f"{self.head_main}_conf"] = conf.squeeze(1)
        else:
            outs[self.head_main] = self._apply_activation_single(
                main_logits, self.activation
            ).squeeze(1)

        # 6) Sky head (fixed 1 channel)
        if self.use_sky_head:
            sky_logits = self.scratch.sky_output_conv2(feat)
            outs[self.sky_name] = self._apply_sky_activation(sky_logits).squeeze(1)

        return outs

    # -------------------------------------------------------------------------
    # Subroutines
    # -------------------------------------------------------------------------
    def _fuse(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """
        4-layer top-down fusion, returns finest scale features (after fusion, before neck1).
        """
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        # 4 -> 3 -> 2 -> 1
        out: torch.Tensor = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        return out

    def _apply_activation_single(
        self, x: torch.Tensor, activation: str = "linear"
    ) -> torch.Tensor:
        """
        Apply activation to single channel output, maintaining semantic consistency with value branch in multi-channel case.
        Supports: exp / relu / sigmoid / softplus / tanh / linear / expp1
        """
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "expm1":
            return torch.expm1(x)
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        # Default linear
        return x

    def _apply_sky_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sky head activation (fixed 1 channel):
          * 'sigmoid' -> Sigmoid probability map
          * 'relu'    -> ReLU positive domain output
          * 'linear'  -> Original value (logits)
        """
        act = (
            self.sky_activation.lower()
            if isinstance(self.sky_activation, str)
            else self.sky_activation
        )
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "relu":
            return torch.relu(x)
        # 'linear'
        return x

    def _add_pos_embed(
        self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1
    ) -> torch.Tensor:
        """Simple UV position encoding directly added to feature map."""
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe


# -----------------------------------------------------------------------------
# Building blocks (preserved, consistent with original)
# -----------------------------------------------------------------------------


def _make_fusion_block(
    features: int,
    size: tuple[int, int] | None = None,
    has_residual: bool = True,
    groups: int = 1,
    inplace: bool = False,
) -> nn.Module:
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=inplace),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


class _Scratch(nn.Module):
    """Namespace container for the DPT fusion chain and head convolutions.

    The official Depth Anything V3 implementation attaches these submodules as
    dynamic attributes on a bare ``nn.Module``. Declaring them here preserves the
    same ``state_dict`` keys while giving mypy concrete, callable types instead of
    the ``Tensor | Module`` union returned by ``nn.Module.__getattr__``.
    """

    layer1_rn: nn.Module
    layer2_rn: nn.Module
    layer3_rn: nn.Module
    layer4_rn: nn.Module
    refinenet1: nn.Module
    refinenet2: nn.Module
    refinenet3: nn.Module
    refinenet4: nn.Module
    output_conv1: nn.Module
    output_conv2: nn.Module
    sky_output_conv2: nn.Module


def _make_scratch(
    in_shape: Sequence[int], out_shape: int, groups: int = 1, expand: bool = False
) -> _Scratch:
    scratch = _Scratch()
    # Optional expansion by stage
    c1 = out_shape
    c2 = out_shape * (2 if expand else 1)
    c3 = out_shape * (4 if expand else 1)
    c4 = out_shape * (8 if expand else 1)

    scratch.layer1_rn = nn.Conv2d(in_shape[0], c1, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], c2, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], c3, 3, 1, 1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], c4, 3, 1, 1, bias=False, groups=groups)
    return scratch


class ResidualConvUnit(nn.Module):
    """Lightweight residual convolution block for fusion"""

    def __init__(
        self, features: int, activation: nn.Module, bn: bool, groups: int = 1
    ) -> None:
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Top-down fusion block: (optional) residual merge + upsampling + 1x1 contraction"""

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        deconv: bool = False,
        bn: bool = False,
        expand: bool = False,
        align_corners: bool = True,
        size: tuple[int, int] | None = None,
        has_residual: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual

        self.resConfUnit1 = (
            ResidualConvUnit(features, activation, bn, groups=groups)
            if has_residual
            else None
        )
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups)

        out_features = (features // 2) if expand else features
        self.out_conv = nn.Conv2d(
            features, out_features, 1, 1, 0, bias=True, groups=groups
        )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(
        self, *xs: torch.Tensor, size: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """
        xs:
          - xs[0]: Top branch input
          - xs[1]: Lateral input (can do residual addition with top branch)
        """
        y: torch.Tensor = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = self.skip_add.add(y, self.resConfUnit1(xs[1]))

        y = self.resConfUnit2(y)

        # Upsampling
        up_kwargs: dict[str, Any]
        if (size is None) and (self.size is None):
            up_kwargs = {"scale_factor": 2}
        elif size is None:
            up_kwargs = {"size": self.size}
        else:
            up_kwargs = {"size": size}

        y = custom_interpolate(
            y, **up_kwargs, mode="bilinear", align_corners=self.align_corners
        )
        y = self.out_conv(y)
        return y
