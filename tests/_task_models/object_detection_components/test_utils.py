#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lightly_train._task_models.object_detection_components.utils import (
    _yolo_to_xyxy,
    bilinear_grid_sample,
    deformable_attention_core_func_v2,
)


def test_yolo_to_xyxy_accepts_1d_box() -> None:
    boxes = [torch.tensor([0.5, 0.5, 0.2, 0.4], dtype=torch.float32)]
    converted = _yolo_to_xyxy(boxes)

    assert len(converted) == 1
    assert converted[0].shape == (1, 4)
    expected = torch.tensor([[0.4, 0.3, 0.6, 0.7]], dtype=torch.float32)
    torch.testing.assert_close(converted[0], expected)


def test_yolo_to_xyxy_accepts_empty_boxes() -> None:
    boxes = [torch.zeros((0,), dtype=torch.float32)]
    converted = _yolo_to_xyxy(boxes)

    assert len(converted) == 1
    assert converted[0].shape == (0, 4)


def test_yolo_to_xyxy_accepts_two_boxes() -> None:
    boxes = [
        torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.4],
                [0.25, 0.75, 0.1, 0.2],
            ],
            dtype=torch.float32,
        )
    ]
    converted = _yolo_to_xyxy(boxes)

    assert len(converted) == 1
    assert converted[0].shape == (2, 4)
    expected = torch.tensor(
        [
            [0.4, 0.3, 0.6, 0.7],
            [0.2, 0.65, 0.3, 0.85],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(converted[0], expected)


@pytest.mark.parametrize(
    ("h", "w", "hg", "wg"),
    [(80, 80, 300, 3), (40, 40, 300, 6), (20, 20, 300, 3), (13, 17, 50, 4)],
)
def test_bilinear_grid_sample__matches_grid_sample(
    h: int, w: int, hg: int, wg: int
) -> None:
    # The gather-based implementation must match F.grid_sample (bilinear, zero
    # padding, align_corners=False), including for out-of-bounds coordinates which
    # deformable attention produces.
    torch.manual_seed(0)
    im = torch.randn(4, 12, h, w)
    grid = torch.rand(4, hg, wg, 2) * 2.6 - 1.3  # spans outside [-1, 1]

    expected = F.grid_sample(
        im, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    got = bilinear_grid_sample(im, grid, align_corners=False)

    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_deformable_attention_core__bilinear_gather_matches_default() -> None:
    # The "bilinear_gather" method (used for TensorRT-safe export) must reproduce
    # the default grid_sample-based sampling numerically.
    torch.manual_seed(0)
    bs, n_head, c = 2, 8, 16
    value_spatial_shapes = [(20, 20), (10, 10)]
    num_points_list = [4, 4]
    len_q = 30
    value_len = sum(h * w for h, w in value_spatial_shapes)

    value = torch.randn(bs, value_len, n_head, c)
    sampling_locations = torch.rand(bs, len_q, n_head, sum(num_points_list), 2)
    attention_weights = torch.rand(bs, len_q, n_head, sum(num_points_list))

    default = deformable_attention_core_func_v2(
        method="default",
        value=value,
        value_spatial_shapes=value_spatial_shapes,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        num_points_list=num_points_list,
    )
    gather = deformable_attention_core_func_v2(
        method="bilinear_gather",
        value=value,
        value_spatial_shapes=value_spatial_shapes,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        num_points_list=num_points_list,
    )

    torch.testing.assert_close(gather, default, atol=1e-5, rtol=1e-5)
