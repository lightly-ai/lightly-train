#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import re
from typing import Union

import numpy as np
import pytest
import torch

from lightly_train._methods.dinov2 import utils
from lightly_train._methods.dinov2.dinov2 import DINOv2AdamWViTArgs
from lightly_train._methods.dinov2.utils import MaskingGenerator, create_collated_masks

from ... import helpers


class TestMaskingGenerator:
    def setup_method(self) -> None:
        self.grid_size = 16

    @pytest.mark.parametrize("grid_size", [14, 16])
    def test_get_shape_and_repr(self, grid_size: int) -> None:
        masking_generator = MaskingGenerator(
            input_size=(grid_size, grid_size), max_num_patches=int(0.5 * grid_size**2)
        )

        assert masking_generator.get_shape() == (grid_size, grid_size)

        repr_str = repr(masking_generator)
        # (the log‐aspect‐ratio values depend on min_aspect/max_aspect; we just check overall pattern)
        assert re.match(
            rf"Generator\({grid_size},\s*{grid_size}\s*->\s*\[\d+\s*~\s*\d+\],\s*max\s*=\s*[-\d\.]+\s*~\s*[-\d\.]+\)",
            repr_str,
        )

    @pytest.mark.parametrize(
        [
            "n_masked_patch_tokens_min",
            "n_masked_patch_tokens_max",
            "masking_ratio",
        ],
        [
            (0, 0, 0.0),
            (0, 128, 0.0),
            (4, 4, 1.0),
            (4, 128, 0.125),
            (4, 128, 0.25),
            (4, 128, 0.5),
            (4, 128, 1.0),
        ],
    )
    def test_masking_generator_call(
        self,
        n_masked_patch_tokens_min: int,
        n_masked_patch_tokens_max: int,
        masking_ratio: float,
    ) -> None:
        n_masked_patch_tokens = int(masking_ratio * self.grid_size**2)

        masking_generator = MaskingGenerator(
            input_size=(self.grid_size, self.grid_size),
            min_num_patches=n_masked_patch_tokens_min,
            max_num_patches=n_masked_patch_tokens_max,
        )

        mask = masking_generator(n_masked_patch_tokens)

        assert mask.dtype == np.bool_
        assert mask.shape == (self.grid_size, self.grid_size)
        assert n_masked_patch_tokens_min <= mask.sum() <= n_masked_patch_tokens

    @pytest.mark.parametrize(
        "aspect_ratio, masking_percentage, is_masked",
        [
            (
                0.1,
                0.005,
                False,
            ),  # min masking_percentage allowed for height>=1 is 0.5**2 / (A*G**2) = 0.0098 > 0.005
            (0.1, 0.05, True),
            (
                0.1,
                0.5,
                False,
            ),  # max masking_percentage allowed for width<=G is (G+0.5)**2*A / (G**2) = 0.106 < 0.5
            (
                2.0,
                0.001,
                False,
            ),  # min masking_percentage allowed for width>=1 is 0.5**2*A / (G**2) = 0.0019 > 0.001
            (2.0, 0.01, True),
            (
                2.0,
                1.0,
                False,
            ),  # max masking_percentage allowed for height<=G is (G+0.5)**2 / (A*G**2) = 0.532 < 1.0
        ],
    )
    def test_masking_generator__aspect_ratio_validity(
        self,
        aspect_ratio: float,
        masking_percentage: Union[float, int],
        is_masked: bool,
    ) -> None:
        # For testing purposes use a masking_percentage to reparameterize the number of masked patches allowed, in this case, let A be the aspect ratio and G the grid size.:
        # 1. the minimum masking_percentage allowed should also be 0.5**2 / (A*G**2) to ensure that the height to be at least 1
        # 2. the minimum masking_percentage allowed should be 0.5**2*A / G**2 to ensure that the width to be at least 1
        # 3. the maximum masking_percentage allowed should be (G+0.5)**2 / (A*G**2) to ensure that the height does not exceed the grid_size
        # 4. the maximum masking_percentage allowed should also be (G+0.5)**2*A / G**2 to ensure that the width does not exceed the grid_size
        # the exact masking_percentage can be slightly different due to int()
        n_masked_patch_tokens = int(masking_percentage * self.grid_size**2)

        masking_generator = MaskingGenerator(
            input_size=(self.grid_size, self.grid_size),
            min_num_patches=n_masked_patch_tokens,
            max_num_patches=n_masked_patch_tokens,
            min_aspect=aspect_ratio,
            max_aspect=aspect_ratio,
        )

        mask = masking_generator(n_masked_patch_tokens)
        assert mask.any() == is_masked

    @pytest.mark.parametrize("square_size", [2, 3, 4])
    def test_masking_generator__aspect_ratio_square(self, square_size: int) -> None:
        """With aspect ratio 1.0 and num_mask=min_num_masks_per_block we expect a single, square masked block."""

        masking_generator = MaskingGenerator(
            input_size=(self.grid_size, self.grid_size),
            max_num_patches=square_size**2,
            min_num_patches=square_size**2,
            min_aspect=1.0,
            max_aspect=1.0,
        )

        mask = masking_generator(square_size**2)
        assert mask.sum(axis=0).max() == square_size
        assert mask.sum(axis=1).max() == square_size


class TestCreateCollatedMasks:
    def setup_method(self) -> None:
        self.grid_size = 16
        self.masking_generator = MaskingGenerator(
            input_size=(self.grid_size, self.grid_size),
            max_num_patches=int(0.5 * self.grid_size**2),
        )

    @pytest.mark.parametrize("expected_n_crops", [1, 2, 4, 8])
    def test_create_collated_masks__dtype_output_size(
        self, expected_n_crops: int
    ) -> None:
        masks = create_collated_masks(
            mask_ratio_min=0.1,
            mask_ratio_max=0.5,
            n_masked_crops=min(2, expected_n_crops),
            n_crops=expected_n_crops,
            mask_generator=self.masking_generator,
        )

        collated_masks = masks["collated_masks"]
        assert collated_masks.dtype == torch.bool

        mask_shape = collated_masks.shape
        assert mask_shape == (expected_n_crops, self.grid_size**2)

    @pytest.mark.parametrize("expected_n_masked_crops", [0, 1, 2, 3, 4])
    def test_create_collated_masks__n_masked_crops(
        self, expected_n_masked_crops: int
    ) -> None:
        masks = create_collated_masks(
            mask_ratio_min=0.1,
            mask_ratio_max=0.5,
            n_masked_crops=expected_n_masked_crops,
            n_crops=max(4, expected_n_masked_crops),
            mask_generator=self.masking_generator,
        )

        collated_masks = masks["collated_masks"]

        n_masked_crops = sum(m.sum() > 0 for m in collated_masks)
        assert n_masked_crops == expected_n_masked_crops

    @pytest.mark.parametrize(
        "mask_ratio_min, mask_ratio_max",
        [(0.1, 0.5), (0.5, 0.8), (1.0, 1.0)],
    )
    def test_create_collated_masks__mask_ratio_min_max(
        self, mask_ratio_min: float, mask_ratio_max: float
    ) -> None:
        masks = create_collated_masks(
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            n_masked_crops=2,
            n_crops=4,
            mask_generator=self.masking_generator,
        )

        collated_masks = masks["collated_masks"]
        for mask in collated_masks:
            n_patch_tokens = mask.numel()
            n_masked_patch_tokens = mask.sum().item()
            if n_masked_patch_tokens == 0:
                continue

            # Check if the number of masked patches is within the specified range
            # Divide lower bound by 4 because the bound is not strict as fewer patches than
            # min_image_mask_ratio * num_patches can be masked. This is because there is a
            # limited number of attempts to find a valid mask that satisfies all constraints.
            assert (
                mask_ratio_min * n_patch_tokens / 4
                <= n_masked_patch_tokens
                <= mask_ratio_max * n_patch_tokens
            )

    def test_create_collated_masks__mask_ratio_zero(
        self,
    ) -> None:
        masks = create_collated_masks(
            mask_ratio_min=0.0,
            mask_ratio_max=0.0,
            n_masked_crops=4,
            n_crops=4,
            mask_generator=self.masking_generator,
        )

        collated_masks = masks["collated_masks"]
        for mask in collated_masks:
            assert not mask.any()


def test_get_optimizer_with_decay() -> None:
    dinov2 = helpers.get_method_dinov2()
    trainable_modules = dinov2.trainable_modules()
    optim = utils.get_optimizer_with_decay(
        optim_args=DINOv2AdamWViTArgs(),
        trainable_modules=trainable_modules,
        layerwise_decay=dinov2.method_args.layerwise_decay,
        patch_embed_lr_multiplier=dinov2.method_args.patch_embed_lr_multiplier,
    )

    # Map fused params back to their original names.
    param_to_name = {}
    for module in list(trainable_modules.modules) + list(
        trainable_modules.modules_no_weight_decay
    ):
        param_to_name.update({p: n for n, p in module.named_parameters()})
    groups = []
    for group in optim.param_groups:
        groups.append({param_to_name[p] for p in group["params"]})

    # Hardcoded to make 100% sure that the groups are correct. If something fails here
    # then there is probably an issue in the grouping logic or the way we set lr, wd, or
    # other parameters for the different parameters.
    expected_groups = [
        {"cls_token", "pos_embed", "mask_token"},
        {"patch_embed.proj.weight"},
        {"patch_embed.proj.bias"},
        {
            "blocks.0.norm1.weight",
            "blocks.0.norm1.bias",
            "blocks.0.attn.qkv.bias",
            "blocks.0.attn.proj.bias",
            "blocks.0.ls1.gamma",
            "blocks.0.norm2.weight",
            "blocks.0.norm2.bias",
            "blocks.0.mlp.fc1.bias",
            "blocks.0.mlp.fc2.bias",
            "blocks.0.ls2.gamma",
        },
        {
            "blocks.0.attn.qkv.weight",
            "blocks.0.attn.proj.weight",
            "blocks.0.mlp.fc1.weight",
            "blocks.0.mlp.fc2.weight",
        },
        {
            "blocks.1.norm1.weight",
            "blocks.1.norm1.bias",
            "blocks.1.attn.qkv.bias",
            "blocks.1.attn.proj.bias",
            "blocks.1.ls1.gamma",
            "blocks.1.norm2.weight",
            "blocks.1.norm2.bias",
            "blocks.1.mlp.fc1.bias",
            "blocks.1.mlp.fc2.bias",
            "blocks.1.ls2.gamma",
        },
        {
            "blocks.1.attn.qkv.weight",
            "blocks.1.attn.proj.weight",
            "blocks.1.mlp.fc1.weight",
            "blocks.1.mlp.fc2.weight",
        },
        {
            "norm.weight",
            "norm.bias",
        },
        {
            "dino_head.mlp.0.weight",
            "dino_head.mlp.2.weight",
            "dino_head.mlp.4.weight",
        },
        {
            "dino_head.mlp.0.bias",
            "dino_head.mlp.2.bias",
            "dino_head.mlp.4.bias",
        },
        {
            "dino_head.last_layer.parametrizations.weight.original0",
            "dino_head.last_layer.parametrizations.weight.original1",
        },
    ]
    assert groups == expected_groups
