#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import random
from typing import Any, Iterable, List

import numpy as np
import torch
from lightly.models.utils import get_weight_decay_parameters
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from lightly_train._modules.teachers.dinov2.layers.layer_scale import LayerScale
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.trainable_modules import TrainableModules


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = (
            num_masking_patches if max_num_patches is None else max_num_patches
        )

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
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

    def __call__(self, num_masking_patches=0):
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
) -> dict[str, torch.Tensor]:
    n_patch_tokens = mask_generator.num_patches
    probs = torch.linspace(mask_ratio_min, mask_ratio_max, n_masked_crops + 1)

    masks_list = []
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

    collated_masks = torch.stack(masks_list).flatten(1)  # [G*B, H*W]
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


def linear_warmup_schedule(
    step: int,
    warmup_steps: int,
    start_value: float,
    end_value: float,
) -> float:  # TODO: import from LightlySSL after new release
    if warmup_steps < 0:
        raise ValueError(f"Warmup steps {warmup_steps} can't be negative.")
    if step < 0:
        raise ValueError(f"Current step number {step} can't be negative.")
    if start_value < 0:
        raise ValueError(f"Start value {start_value} can't be negative.")
    if end_value <= 0:
        raise ValueError(f"End value {end_value} can't be non-positive.")
    if start_value > end_value:
        raise ValueError(
            f"Start value {start_value} must be less than or equal to end value {end_value}."
        )
    if step < warmup_steps:
        return start_value + step / warmup_steps * (end_value - start_value)
    else:
        return end_value


def get_layer_scale_modules(
    modules: Iterable[Module],
) -> List[Module]:
    """Get the parameters of the layer scale gamma to cancel the weight decay.

    Args:
        modules:
            List of modules to get the parameters from.
    Returns:
        List of modules of the layer scale gamma.
    """
    layer_scale_gamma = []

    for module in modules:
        for modules in module.modules():
            if isinstance(modules, LayerScale):
                layer_scale_gamma.append(modules)

    return layer_scale_gamma


def get_vit_lr_decay_rate(
    name,
    lr_decay_rate=1.0,
    num_layers=12,
    force_is_backbone=False,
    chunked_blocks=False,
):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
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


def get_params_groups_with_decay(
    model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0
):  # TODO: fuse with LightlySSL's configure_optimizer
    chunked_blocks = False
    if hasattr(model, "n_blocks"):
        n_blocks = model.n_blocks
        chunked_blocks = model.chunked_blocks
    elif hasattr(model, "blocks"):
        n_blocks = len(model.blocks)
    elif hasattr(model, "backbone"):
        n_blocks = len(model.backbone.blocks)
    else:
        n_blocks = 0
    all_param_groups = []

    for name, param in model.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "")
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name,
            lr_decay_rate,
            num_layers=n_blocks,
            force_is_backbone=n_blocks > 0,
            chunked_blocks=chunked_blocks,
        )
        d = {"name": name, "params": param, "lr_multiplier": decay_rate}

        if "patch_embed" in name:
            d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

        all_param_groups.append(d)

    return all_param_groups


def get_optimizer_with_layerwise_lr_decay(
    optim_args: OptimizerArgs,
    trainable_modules: TrainableModules,
    lr_scale: float,
) -> Optimizer:  # adapted from lightly_train._optim.optimizer_helpers
    params_weight_decay, params_no_weight_decay = get_weight_decay_parameters(
        modules=trainable_modules.modules
    )
    if trainable_modules.modules_no_weight_decay is not None:
        for m in trainable_modules.modules_no_weight_decay:
            params_no_weight_decay.extend(m.parameters())

    params: list[dict[str, Any]] = [{"name": "params", "params": params_weight_decay}]
    if params_no_weight_decay:
        params.append(
            {
                "name": "params_no_weight_decay",
                "params": params_no_weight_decay,
                "weight_decay": 0.0,
            }
        )
    return optim_args.get_optimizer(params=params, lr_scale=lr_scale)
