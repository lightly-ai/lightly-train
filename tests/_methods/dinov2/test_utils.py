#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import re

import numpy as np
import pytest
import torch

from lightly_train._methods.dinov2.utils import MaskingGenerator, create_collated_masks


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

    # Type ignore because untyped decorator makes function untyped.
    @pytest.mark.parametrize(
        "min_aspect_ratio,max_aspect_ratio",
        [
            (0.1, None),
            (0.1, 0.3),
            (0.1, 3.0),
            (0.3, 0.3),
        ],
    )
    def test_masking_generator__aspect_ratio(
        self, min_aspect_ratio: float, max_aspect_ratio: float | None
    ) -> None:
        n_masked_patch_tokens = int(0.5 * self.grid_size**2)

        masking_generator = MaskingGenerator(
            input_size=(self.grid_size, self.grid_size),
            max_num_patches=n_masked_patch_tokens,
            min_aspect=min_aspect_ratio,
            max_aspect=max_aspect_ratio,
        )

        mask = masking_generator(n_masked_patch_tokens)
        assert mask.sum() > 0

    @pytest.mark.parametrize("square_size", [2, 3, 4])
    def test_masking_generator__aspect_ratio_one(self, square_size: int) -> None:
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

        num_crops_masked = sum(m.sum() > 0 for m in collated_masks)
        assert num_crops_masked == expected_n_masked_crops

    @pytest.mark.parametrize(
        "mask_ratio_min, mask_ratio_max",
        [(0.0, 0.0), (0.1, 0.5), (1.0, 1.0)],
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
            num_patches = mask.numel()
            num_masked_patches = mask.sum().item()
            if num_masked_patches == 0:
                continue

            # Check if the number of masked patches is within the specified range
            # Divide lower bound by 4 because the bound is not strict as fewer patches than
            # min_image_mask_ratio * num_patches can be masked. This is because there is a
            # limited number of attempts to find a valid mask that satisfies all constraints.
            assert (
                mask_ratio_min * num_patches / 4
                <= num_masked_patches
                <= mask_ratio_max * num_patches
            )
