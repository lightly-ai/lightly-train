#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.#

# Modifications Copyright 2025 Lightly AG:
# - Modified load_state_dict to handle different number of input channels
# - Added `_compute_resize_matrix`, `_apply_resampling` and `resample_patch_embed`
#   from TIMM (https://github.com/huggingface/pytorch-image-models) to
#   support flexible patch sizes.

from __future__ import annotations

import math
from typing import Callable, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lightly_train._models import _model_helpers


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        if hasattr(self, "register_load_state_dict_pre_hook"):
            self.register_load_state_dict_pre_hook(
                _model_helpers.patch_embed_adjust_input_channels_hook
            )
        else:
            # Backwards compatibility for PyTorch <= 2.4
            self._register_load_state_dict_pre_hook(
                _model_helpers.patch_embed_adjust_input_channels_hook, with_module=True
            )

    def resample_conv_weight(
        self,
        weight: Tensor,
        target_patch_size: int,
    ) -> Tensor:
        """Resample conv2d patch embedding weights for a new patch size.

        Args:
            weight: Conv2d weight tensor of shape [embed_dim, in_chans, patch_h, patch_w]
            target_patch_size: Target (patch_h, patch_w) to resample to

        Returns:
            Resampled weight tensor
        """
        # Comparison assumes square patch sizes.
        if target_patch_size == weight.shape[-1]:
            return weight

        # Resample using existing function.
        weight_resampled = resample_patch_embed(
            weight,
            new_size=[target_patch_size, target_patch_size],
        )

        return weight_resampled

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


def _compute_resize_matrix(
    old_size: tuple[int, int],
    new_size: tuple[int, int],
    interpolation: str,
    antialias: bool,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute the resize matrix used for Pi-Resize.

    Source:
    https://github.com/huggingface/pytorch-image-models/blob/90cae8c5ab83d2d7e23b1fb2bab1559bfdcbc7e9/timm/layers/patch_embed.py#L266

    Args:
        old_size: Original spatial size (height, width).
        new_size: Target spatial size (height, width).
        interpolation: Interpolation mode used for resizing basis vectors.
        antialias: Whether to apply antialiasing during interpolation.
        device: Device on which to create the resize matrix.
        dtype: Data type of the resize matrix.

    Returns:
        Resize matrix of shape (new_height * new_width, old_height * old_width).
    """
    old_h, old_w = old_size
    new_h, new_w = new_size
    old_total = old_h * old_w
    new_total = new_h * new_w

    eye_matrix = torch.eye(old_total, device=device, dtype=dtype)
    basis_vectors_batch = eye_matrix.reshape(old_total, 1, old_h, old_w)
    resized_basis_vectors_batch = F.interpolate(
        basis_vectors_batch,
        size=new_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )  # Output shape: (old_total, 1, new_h, new_w)
    resize_matrix = (
        resized_basis_vectors_batch.squeeze(1)
        .permute(1, 2, 0)
        .reshape(new_total, old_total)
    )
    return resize_matrix  # Shape: (new_total, old_total)


def _apply_resampling(
    patch_embed: Tensor,
    pinv_matrix: Tensor,
    new_size_tuple: Tuple[int, int],
    orig_dtype: torch.dtype,
    intermediate_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Apply Pi-Resize resampling using a pseudoinverse resize matrix.

    Source:
    https://github.com/huggingface/pytorch-image-models/blob/90cae8c5ab83d2d7e23b1fb2bab1559bfdcbc7e9/timm/layers/patch_embed.py#L293

    Args:
        patch_embed: Patch embedding projection weights of shape
            (out_channels, in_channels, height, width).
        pinv_matrix: Pseudoinverse resize matrix of shape (old_hw, new_hw).
        new_size_tuple: Target spatial size (height, width).
        orig_dtype: Original dtype of `patch_embed` to restore after resampling.
        intermediate_dtype: Dtype used for intermediate computation.

    Returns:
        Resampled patch embedding weights with spatial size `new_size_tuple`.
    """
    c_out, c_in, *_ = patch_embed.shape
    patch_embed = patch_embed.reshape(c_out, c_in, -1).to(dtype=intermediate_dtype)
    pinv_matrix = pinv_matrix.to(dtype=intermediate_dtype)
    resampled_patch_embed = (
        patch_embed @ pinv_matrix
    )  # (C_out, C_in, P_old * P_old) @ (P_old * P_old, P_new * P_new)
    resampled_patch_embed = resampled_patch_embed.reshape(
        c_out, c_in, *new_size_tuple
    ).to(dtype=orig_dtype)
    return resampled_patch_embed


def resample_patch_embed(
    patch_embed: Tensor,
    new_size: list[int],
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> Tensor:
    """
    Resample ViT patch-embedding weights using Pi-Resize.

    Source:
    https://github.com/huggingface/pytorch-image-models/blob/90cae8c5ab83d2d7e23b1fb2bab1559bfdcbc7e9/timm/layers/patch_embed.py#L311

    Implements the pseudoinverse (Pi-Resize) method from:
    https://arxiv.org/pdf/2212.08013

    Args:
        patch_embed: Patch embedding projection weights of shape
            (out_channels, in_channels, height, width).
        new_size: Target spatial size [height, width].
        interpolation: Interpolation mode for building the resize operator.
        antialias: Whether to apply antialiasing.

    Returns:
        Resampled patch embedding weights with spatial size `new_size`.
    """
    assert len(patch_embed.shape) == 4, (
        "Input tensor should be 4D (out_ch, in_ch, h, w)"
    )
    assert len(new_size) == 2, "New shape should only be hw (height, width)"

    old_size_tuple: Tuple[int, int] = tuple(patch_embed.shape[-2:])
    new_size_tuple: Tuple[int, int] = tuple(new_size)

    if old_size_tuple == new_size_tuple:
        return patch_embed

    device = patch_embed.device
    orig_dtype = patch_embed.dtype

    resize_mat = _compute_resize_matrix(
        old_size_tuple, new_size_tuple, interpolation, antialias, device, torch.float32
    )
    pinv_matrix = torch.linalg.pinv(
        resize_mat
    )  # Calculates the pseudoinverse matrix used for resampling
    resampled_patch_embed = _apply_resampling(
        patch_embed, pinv_matrix, new_size_tuple, orig_dtype, torch.float32
    )
    return resampled_patch_embed
