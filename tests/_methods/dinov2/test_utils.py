#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest

from lightly_train._methods.dinov2.utils import MaskingGenerator, create_collated_masks


class TestCreateCollatedMasks:
    def setup_method(self):
        self.grid_size = 16
        self.masking_generator = MaskingGenerator(
            input_size=(self.grid_size, self.grid_size),
            max_num_patches=int(0.5 * self.grid_size**2),
        )

    def test_create_collated_masks__output_size(self) -> None:
        num_crops = 4

        masks = create_collated_masks(
            mask_ratio_min=0.1,
            mask_ratio_max=0.5,
            n_masked_crops=2,
            n_crops=4,
            mask_generator=self.masking_generator,
        )

        collated_masks = masks["collated_masks"]

        mask_shape = collated_masks.shape
        assert mask_shape == (num_crops, self.grid_size**2)

    @pytest.mark.parametrize(
        "num_crops, expected_num_crops_masked",
        [
            (1, 0),
            (1, 1),
            (4, 2),
            (4, 3),
        ],
    )
    def test_create_collated_masks__num_crops_masked(
        self, num_crops: int, expected_num_crops_masked: int
    ) -> None:
        masks = create_collated_masks(
            mask_ratio_min=0.1,
            mask_ratio_max=0.5,
            n_masked_crops=expected_num_crops_masked,
            n_crops=num_crops,
            mask_generator=self.masking_generator,
        )

        collated_masks = masks["collated_masks"]

        num_crops_masked = sum(m.sum() > 0 for m in collated_masks)
        assert num_crops_masked == expected_num_crops_masked

    @pytest.mark.parametrize(
        "mask_ratio_min, mask_ratio_max",
        [(0.0, 0.0), (0.1, 0.5), (1.0, 1.0)],
    )
    def test_create_collated_masks__min_max_image_mask_ratio(
        self, mask_ratio_min: float, mask_ratio_max: float
    ) -> None:
        masks = create_collated_masks(
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            n_masked_crops=4,
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
