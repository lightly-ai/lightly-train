#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Any, Dict, List

from torch.optim.optimizer import Optimizer

from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._optim.trainable_modules import TrainableModules


class DINOv2AdamWViTSBArgs(AdamWArgs):
    lr: float = 0.004
    weight_decay: float = 0.04


class DINOv2AdamWViTLGArgs(AdamWArgs):
    lr: float = 2e-4
    weight_decay: float = 0.04


def get_vit_lr_decay_rate(
    name: str,
    lr_decay_rate: float,
    num_layers: int = 12,
    force_is_backbone: bool = False,
    chunked_blocks: bool = False,
) -> float:
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
        force_is_backbone (bool): force to use backbone.
        chunked_blocks (bool): if the blocks are chunked.
    Returns:
        float: lr decay rate for the given parameter.
    """

    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".register_tokens" in name
        ):
            layer_id = 0
        elif force_is_backbone and (
            "pos_embed" in name
            or "patch_embed" in name
            or "mask_token" in name
            or "cls_token" in name
            or "register_tokens" in name
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        elif chunked_blocks and "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_optimizer_with_decay(
    optim_args: DINOv2AdamWViTSBArgs | DINOv2AdamWViTLGArgs,
    trainable_modules: TrainableModules,
    lr_scale: float,
    layerwise_decay: float,
    patch_embed_lr_multiplier: float,
) -> Optimizer:
    """
    Create an optimizer with layerwise learning rate decay and weight decay for different ViT blocks.

    Args:
        optim_args (DINOv2AdamWArgs): optimizer arguments.
        trainable_modules (TrainableModules): trainable modules.
        lr_scale (float): learning rate scale.
        layerwise_decay (float): base lr decay rate.
        patch_embed_lr_multiplier (float): multiplier for patch embedding layer.
    Returns:
        Optimizer: optimizer with decay.
    """

    all_param_groups: List[Dict[str, Any]] = []
    for module in trainable_modules.modules:  # TODO: FSDP sharding
        chunked_blocks = False
        if hasattr(module, "n_blocks"):  # chunked fsdp
            n_blocks = module.n_blocks
            chunked_blocks = module.chunked_blocks
        elif hasattr(module, "blocks"):  # first code branch
            n_blocks = len(module.blocks)
        elif hasattr(module, "backbone"):  # second code branch
            n_blocks = len(module.backbone.blocks)
        else:
            n_blocks = 0  # else code branch

        for name, param in module.named_parameters():
            name = name.replace("_fsdp_wrapped_module.", "")
            if not param.requires_grad:
                continue
            decay_rate = get_vit_lr_decay_rate(
                name=name,
                lr_decay_rate=layerwise_decay,
                num_layers=n_blocks,
                force_is_backbone=n_blocks > 0,
                chunked_blocks=chunked_blocks,
            )
            d = {
                "name": name,
                "params": param,
                "lr": optim_args.lr * decay_rate,
                "weight_decay": optim_args.weight_decay,
                "foreach": True,
            }

            if (
                name.endswith(".bias") or "norm" in name or "gamma" in name
            ):  # disable weight decay for bias and norm layers and layerscale gamma
                d.update({"weight_decay": 0.0})

            if "patch_embed" in name:  # multiplier for patch embedding layer
                d.update({"lr": d["lr"] * patch_embed_lr_multiplier})  # type: ignore[operator]

            all_param_groups.append(d)

    return optim_args.get_optimizer(params=all_param_groups, lr_scale=lr_scale)
