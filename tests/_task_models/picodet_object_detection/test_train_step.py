#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch

from lightly_train._task_models.picodet_object_detection.pico_head import (
    bbox2distance,
)


def test_dfl_targets_large_distance_clamped_after_stride() -> None:
    """Test that large distances are clamped AFTER stride division, not before."""
    # Point at (200, 200) with stride 8, box from (0, 0) to (400, 400)
    points = torch.tensor([[200.0, 200.0]])
    strides = torch.tensor([[8.0]])
    bboxes = torch.tensor([[0.0, 0.0, 400.0, 400.0]])
    reg_max = 7

    # Correct: get raw distances, divide by stride, then clamp.
    # 200 pixels / 8 stride = 25 -> clamped to 6.99
    distances_pixels = bbox2distance(points, bboxes, reg_max=None)
    distances_feature = distances_pixels / strides
    distances_correct = distances_feature.clamp(min=0, max=reg_max - 0.01)
    expected_correct = torch.tensor([[6.99, 6.99, 6.99, 6.99]])
    torch.testing.assert_close(distances_correct, expected_correct)

    # Wrong: clamp in pixels first, then divide by stride gives ~0.87 instead of 6.99
    distances_wrong = bbox2distance(points, bboxes, reg_max=7.0)
    distances_wrong = distances_wrong / strides
    assert not torch.allclose(distances_wrong, expected_correct)
    assert distances_wrong.max().item() < 1.0
