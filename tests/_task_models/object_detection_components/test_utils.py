#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch

from lightly_train._task_models.object_detection_components.utils import _yolo_to_xyxy


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
