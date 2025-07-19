#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#

# References:
#   - https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/masking.py
#   - https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/collate.py
#   - https://github.com/facebookresearch/dinov2/blob/main/dinov2/utils/param_groups.py
#
# Modifications Copyright 2025 Lightly AG:
# - all: added type hints and slightly modified the inputs and their types for some arguments according to the changes
# - all: rename some variables
# - collate: remove collated_global_crops, collated_local_crops, upperbound, and n_masked_patches
# - param_groups: adjusted the parameter structure
# - param_groups: feed the parameter groups directly to the optimizer
# - get_optimizer_with_decay, get_vit_lr_decay_rate: removed the different options
#       as in this codebase only one version is supported also check if model is
#       DinoVisionTransformer to validate it is the "backbone"
# - get_fused_param_groups: Adapted from original code to work with our group structure.

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.trainable_modules import TrainableModules


class MaskingGenerator:
    def __init__(
        self,
        input_size: int | tuple[int, int],
        max_num_patches: int,
        min_num_patches: int = 4,
        min_aspect: float = 0.3,
        max_aspect: float | None = None,
    ) -> None:
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width

        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self) -> str:
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self) -> tuple[int, int]:
        return self.height, self.width

    def _mask(self, mask: np.ndarray, max_mask_patches: int) -> int:  # type: ignore[type-arg]
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches: int = 0) -> np.ndarray:  # type: ignore[type-arg]
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


def create_collated_masks(
    mask_ratio_min: float,
    mask_ratio_max: float,
    n_masked_crops: int,
    n_crops: int,
    mask_generator: MaskingGenerator,
) -> Dict[str, torch.Tensor]:
    n_patch_tokens = mask_generator.num_patches
    probs = np.linspace(mask_ratio_min, mask_ratio_max, n_masked_crops + 1)

    masks_list: List[torch.Tensor] = []
    for i in range(0, n_masked_crops):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(
            torch.BoolTensor(
                mask_generator(int(n_patch_tokens * random.uniform(prob_min, prob_max)))
            )
        )
    for i in range(n_masked_crops, n_crops):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)  # [G*B, H/p*W/p]
    mask_indices_list = collated_masks.flatten().nonzero().flatten()  # [M,]
    masks_weight = (
        (1 / collated_masks.sum(-1).clamp(min=1.0))
        .unsqueeze(-1)
        .expand_as(collated_masks)[collated_masks]
    )  # [M,]

    return {
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
    }


def get_vit_lr_decay_rate(
    name: str,
    lr_decay_rate: float,
    num_layers: int = 12,
    chunked_blocks: bool = False,
) -> float:
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
        chunked_blocks (bool): if the blocks are chunked.
    Returns:
        float: lr decay rate for the given parameter.
    """

    layer_id = num_layers + 1
    if (
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
    optim_args: OptimizerArgs,
    trainable_modules: TrainableModules,
    layerwise_decay: float,
    patch_embed_lr_multiplier: float,
) -> Optimizer:
    """
    Create an optimizer with layerwise learning rate decay and weight decay for different ViT blocks.

    Args:
        optim_args (OptimizerArgs): optimizer arguments.
        trainable_modules (TrainableModules): trainable modules.
        layerwise_decay (float): base lr decay rate.
        patch_embed_lr_multiplier (float): multiplier for patch embedding layer.
    Returns:
        Optimizer: optimizer with decay.
    """

    all_param_groups: List[Dict[str, Any]] = []
    for module in trainable_modules.modules:
        # NOTE: If you change behavior of parameters here then make sure to also
        # double check get_fused_param_groups whether it needs any updates.

        is_backbone = False
        if isinstance(module, DinoVisionTransformer):
            is_backbone = True

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            decay_rate = 1.0
            if is_backbone:
                decay_rate = get_vit_lr_decay_rate(
                    name=name,
                    lr_decay_rate=layerwise_decay,
                    num_layers=module.n_blocks,
                    chunked_blocks=module.chunked_blocks,
                )
            d = {
                "name": name,
                "params": [param],
                "lr": optim_args.lr * decay_rate,  # type: ignore[attr-defined]
                "weight_decay": optim_args.weight_decay,  # type: ignore[attr-defined]
                "foreach": True,
            }  # TODO: ignore to be removed after improving optimizer args

            if (
                name.endswith(".bias") or "norm" in name or "gamma" in name
            ):  # disable weight decay for bias and norm layers and layerscale gamma
                d.update({"weight_decay": 0.0})

            if "patch_embed" in name:  # multiplier for patch embedding layer
                d.update({"lr": d["lr"] * patch_embed_lr_multiplier})  # type: ignore[operator]

            all_param_groups.append(d)

    fused_param_groups = get_fused_param_groups(all_param_groups)
    return optim_args.get_optimizer(params=fused_param_groups, lr_scale=1.0)


def get_fused_param_groups(param_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fuses parameter groups with the same properties.

    This is slightly more efficient for the optimizer but the main benefit is that it
    reduces the number of log messages (one per group).

    The fused groups are named after the first parameter in the group.
    """
    fused = {}
    for group in param_groups:
        ids = {k: v for k, v in group.items() if k not in ["params", "name"]}
        # Add head and last_layer because they are treated differently in
        # DINOv2.on_before_optimizer_step
        ids["head"] = "head" in group["name"]
        ids["last_layer"] = "last_layer" in group["name"]
        group_id = "_".join(f"{k}={v}" for k, v in ids.items())
        if group_id not in fused:
            fused[group_id] = group
        else:
            fused[group_id]["params"].extend(group["params"])
    return list(fused.values())
