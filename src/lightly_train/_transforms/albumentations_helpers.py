#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightning_utilities.core.imports import RequirementCache

# Normalize supports the `normalization` argument.
ALBUMENTATIONS_GEQ_1_4_4 = RequirementCache("albumentations>=1.4.4")
# BboxParams supports the `clip` argument.
ALBUMENTATIONS_GEQ_1_4_5 = RequirementCache("albumentations>=1.4.5")
# `crop_bbox_by_coords` no longer takes crop height and width.
ALBUMENTATIONS_GEQ_1_4_11 = RequirementCache("albumentations>=1.4.11")
# Bboxes are numpy arrays instead of lists of tuples.
ALBUMENTATIONS_GEQ_1_4_15 = RequirementCache("albumentations>=1.4.15")
# Transforms use `get_params_dependent_on_data` and seeded `py_random` and
# `random_generator`.
ALBUMENTATIONS_GEQ_1_4_21 = RequirementCache("albumentations>=1.4.21")
# Solarize uses `threshold_range` instead of `threshold`.
ALBUMENTATIONS_GEQ_1_4_22 = RequirementCache("albumentations>=1.4.22")
# Albumentations 2 API: `size` instead of `height` and `width`, `fill` instead of
# `value`. Also needed for `record_geometry` in view transforms.
ALBUMENTATIONS_GEQ_2_0_0 = RequirementCache("albumentations>=2.0.0")
# BboxParams supports the `filter_invalid_bboxes` argument.
ALBUMENTATIONS_GEQ_2_0_1 = RequirementCache("albumentations>=2.0.1")
