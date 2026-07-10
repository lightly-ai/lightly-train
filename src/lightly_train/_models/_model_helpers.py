#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

logger = logging.getLogger(__name__)


def patch_embed_adjust_input_channels_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Hook to adjust the number of channels in the state dict to the number of
    channels in the module.
    """
    in_chans: Tensor = module.in_chans  # type: ignore
    proj_weight_key = f"{prefix}proj.weight"
    proj_weight = state_dict.get(proj_weight_key)
    if proj_weight is not None:
        weights_in_chans = proj_weight.shape[1]
        if weights_in_chans > in_chans:
            # Drop last channels
            logger.info(
                f"Loading pretrained weights with {weights_in_chans} input channels, "
                f"but model has {in_chans} input channels. Keeping only the "
                f"first {in_chans} channels of the pretrained weights."
            )
            proj_weight = proj_weight[:, :in_chans, :, :]
        elif weights_in_chans < in_chans:
            # Repeat channels to initialize extra channels
            logger.info(
                f"Loading pretrained weights with {weights_in_chans} input channels, "
                f"but model has {in_chans} input channels. Repeating the "
                "channels of the pretrained weights to initialize the extra "
                "channels."
            )
            repeat_times = in_chans // weights_in_chans
            remainder = in_chans % weights_in_chans
            proj_weight = proj_weight.repeat(1, repeat_times, 1, 1)
            if remainder > 0:
                proj_weight = torch.cat(
                    [proj_weight, proj_weight[:, :remainder, :, :]], dim=1
                )
        state_dict[proj_weight_key] = proj_weight


def interpolate_pos_embed_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Bicubic-resize a mismatched ``pos_embed`` so a checkpoint trained at one
    image size restores into a DINOv2 ViT at another (e.g. a 224px DINOv2
    checkpoint into a 518px model).

    Only fires on a genuine shape mismatch with a square patch grid; a no-op
    otherwise. Mutates ``state_dict`` in place.

    Note: assumes a single leading non-patch (cls) token. Models with register
    tokens (``num_register_tokens > 0``) carry additional leading tokens that
    this heuristic would mis-parse; the common DINOv2 init path (``vits14``,
    no registers) is handled correctly.
    """
    key = f"{prefix}pos_embed"
    value = state_dict.get(key)
    if value is None:
        return
    target: Tensor = module.pos_embed  # type: ignore[attr-defined]
    if value.shape == target.shape:
        return
    # Expect [1, 1 + n_patches, dim] (cls token + a square patch grid).
    if value.dim() != 3 or value.shape[0] != 1 or value.shape[-1] != target.shape[-1]:
        return
    n_old = value.shape[1] - 1
    n_new = target.shape[1] - 1
    grid_old = int(round(math.sqrt(n_old)))
    grid_new = int(round(math.sqrt(n_new)))
    if grid_old * grid_old != n_old or grid_new * grid_new != n_new:
        return
    dim = value.shape[-1]
    cls_token = value[:, :1]
    patches = value[:, 1:].reshape(1, grid_old, grid_old, dim).permute(0, 3, 1, 2)
    patches = torch.nn.functional.interpolate(
        patches,
        size=(grid_new, grid_new),
        mode="bicubic",
        align_corners=False,
    )
    patches = patches.permute(0, 2, 3, 1).reshape(1, n_new, dim)
    state_dict[key] = torch.cat([cls_token, patches], dim=1).to(value.dtype)
    logger.info(
        f"Interpolated '{key}' pos_embed {tuple(value.shape)} -> "
        f"{tuple(target.shape)} for checkpoint load."
    )
