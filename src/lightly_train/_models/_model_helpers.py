#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Any

import torch
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

    proj_weight_key = f"{prefix}proj.weight"
    proj_weight = state_dict.get(proj_weight_key)
    if proj_weight is not None:
        weights_in_chans = proj_weight.shape[1]
        if weights_in_chans > module.in_chans:
            # Drop last channels
            logger.info(
                f"Loading pretrained weights with {weights_in_chans} input channels, "
                f"but model has {module.in_chans} input channels. Keeping only the "
                f"first {module.in_chans} channels of the pretrained weights."
            )
            proj_weight = proj_weight[:, : module.in_chans, :, :]
        elif weights_in_chans < module.in_chans:
            # Repeat channels to initialize extra channels
            logger.info(
                f"Loading pretrained weights with {weights_in_chans} input channels, "
                f"but model has {module.in_chans} input channels. Repeating the "
                "channels of the pretrained weights to initialize the extra "
                "channels."
            )
            repeat_times = module.in_chans // weights_in_chans
            remainder = module.in_chans % weights_in_chans
            proj_weight = proj_weight.repeat(1, repeat_times, 1, 1)
            if remainder > 0:
                proj_weight = torch.cat(
                    [proj_weight, proj_weight[:, :remainder, :, :]], dim=1
                )
        state_dict[proj_weight_key] = proj_weight
