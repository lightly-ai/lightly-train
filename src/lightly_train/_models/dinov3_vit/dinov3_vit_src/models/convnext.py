# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
from torch import Tensor, nn


logger = logging.getLogger("dinov3")


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    r"""
    Code adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.pyConvNeXt

    A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        patch_size (int | None): Pseudo patch size. Used to resize feature maps to those of a ViT with a given patch size. If None, no resizing is performed
    """

    def __init__(
        self,
        # original ConvNeXt arguments
        in_chans: int = 3,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        # DINO arguments
        patch_size: int | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        # ==== ConvNeXt's original init =====
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # ==== End of ConvNeXt's original init =====

        # ==== DINO adaptation ====
        self.head = nn.Identity()  # remove classification head
        self.embed_dim = dims[-1]
        self.embed_dims = dims  # per layer dimensions
        self.n_blocks = len(self.downsample_layers)  # 4
        self.chunked_blocks = False
        self.n_storage_tokens = 0  # no registers

        self.norms = nn.ModuleList([nn.Identity() for i in range(3)])
        self.norms.append(self.norm)

        self.patch_size = patch_size
        self.input_pad_size = 4  # first convolution with kernel_size = 4, stride = 4

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.reset_parameters()
        if isinstance(module, LayerNorm):
            module.weight = nn.Parameter(torch.ones(module.normalized_shape))
            module.bias = nn.Parameter(torch.zeros(module.normalized_shape))
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        output = []
        for x, masks in zip(x_list, masks_list):
            h, w = x.shape[-2:]
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            x_pool = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
            x = torch.flatten(x, 2).transpose(1, 2)

            # concat [CLS] and patch tokens as (N, HW + 1, C), then normalize
            x_norm = self.norm(torch.cat([x_pool.unsqueeze(1), x], dim=1))
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_storage_tokens": x_norm[:, 1 : self.n_storage_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )

        return output

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

    def _get_intermediate_layers(self, x, n=1):
        h, w = x.shape[-2:]
        output, total_block_len = [], len(self.downsample_layers)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i in range(total_block_len):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in blocks_to_take:
                x_pool = x.mean([-2, -1])
                x_patches = x
                if self.patch_size is not None:
                    # Resize output feature maps to that of a ViT with given patch_size
                    x_patches = nn.functional.interpolate(
                        x,
                        size=(h // self.patch_size, w // self.patch_size),
                        mode="bilinear",
                        antialias=True,
                    )
                output.append(
                    [
                        x_pool,  # CLS (B x C)
                        x_patches,  # B x C x H x W
                    ]
                )
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        outputs = self._get_intermediate_layers(x, n)

        if norm:
            nchw_shapes = [out[-1].shape for out in outputs]
            if isinstance(n, int):
                norms = self.norms[-n:]
            else:
                norms = [self.norms[i] for i in n]
            outputs = [
                (
                    norm(cls_token),  # N x C
                    norm(patches.flatten(-2, -1).permute(0, 2, 1)),  # N x HW x C
                )
                for (cls_token, patches), norm in zip(outputs, norms)
            ]
            if reshape:
                outputs = [
                    (cls_token, patches.permute(0, 2, 1).reshape(*nchw).contiguous())
                    for (cls_token, patches), nchw in zip(outputs, nchw_shapes)
                ]
        elif not reshape:
            # force B x N x C format for patch tokens
            outputs = [(cls_token, patches.flatten(-2, -1).permute(0, 2, 1)) for (cls_token, patches) in outputs]
        class_tokens = [out[0] for out in outputs]
        outputs = [out[1] for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}


def get_convnext_arch(arch_name):
    size_dict = None
    query_sizename = arch_name.split("_")[1]
    try:
        size_dict = convnext_sizes[query_sizename]
    except KeyError:
        raise NotImplementedError("didn't recognize vit size string")

    return partial(
        ConvNeXt,
        **size_dict,
    )
