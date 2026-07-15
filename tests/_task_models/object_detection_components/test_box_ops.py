#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging

import torch
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from lightly_train._task_models.object_detection_components import box_ops
from lightly_train._task_models.object_detection_components.box_ops import (
    sanitize_boxes_cxcywh_normalized,
)


def test_sanitize_boxes_cxcywh_normalized_clamps_without_warning(
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(box_ops, "_invalid_bbox_warning_emitted", False)
    boxes = torch.tensor(
        [[-0.5, 0.25, 1.5, 0.75], [0.2, 1.2, 0.0, -0.1]],
        dtype=torch.float32,
    )

    with caplog.at_level(logging.WARNING, logger=box_ops.__name__):
        sanitized = sanitize_boxes_cxcywh_normalized(boxes)

    expected = torch.tensor(
        [[0.0, 0.25, 1.0, 0.75], [0.2, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(sanitized, expected)
    assert caplog.records == []


def test_sanitize_boxes_cxcywh_normalized_replaces_invalid_values(
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(box_ops, "_invalid_bbox_warning_emitted", False)
    boxes = torch.tensor(
        [[torch.nan, torch.inf, -torch.inf, 0.5]],
        dtype=torch.float32,
    )

    with caplog.at_level(logging.WARNING, logger=box_ops.__name__):
        sanitized = sanitize_boxes_cxcywh_normalized(boxes)

    expected = torch.tensor([[0.0, 1.0, 0.0, 0.5]], dtype=torch.float32)
    torch.testing.assert_close(sanitized, expected)
    assert len(caplog.records) == 1
    assert "Found invalid predicted bbox values" in caplog.records[0].message


def test_sanitize_boxes_cxcywh_normalized_warns_only_once(
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(box_ops, "_invalid_bbox_warning_emitted", False)
    boxes = torch.tensor([[torch.nan, 0.2, 0.3, 0.4]], dtype=torch.float32)

    with caplog.at_level(logging.WARNING, logger=box_ops.__name__):
        sanitize_boxes_cxcywh_normalized(boxes)
        sanitize_boxes_cxcywh_normalized(boxes)

    assert len(caplog.records) == 1


def test_sanitize_boxes_cxcywh_normalized_enforces_min_size_px() -> None:
    # 4 px on a 64 px image == 4/64 = 0.0625 in normalized units.
    image_size = (64, 64)
    boxes = torch.tensor(
        [
            # Sub-minimum width: w must clamp up to 4 px.
            [0.5, 0.5, 3 / 64, 4 / 64],
            # Sub-minimum height: h must clamp up to 4 px.
            [0.5, 0.5, 4 / 64, 3 / 64],
            # Already at the limit: untouched.
            [0.5, 0.5, 4 / 64, 4 / 64],
            # Well above the limit: untouched.
            [0.5, 0.5, 32 / 64, 32 / 64],
        ],
        dtype=torch.float32,
    )

    sanitized = sanitize_boxes_cxcywh_normalized(
        boxes, image_size=image_size, min_size_px=4.0
    )

    expected = torch.tensor(
        [
            [0.5, 0.5, 4 / 64, 4 / 64],
            [0.5, 0.5, 4 / 64, 4 / 64],
            [0.5, 0.5, 4 / 64, 4 / 64],
            [0.5, 0.5, 32 / 64, 32 / 64],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(sanitized, expected)


def test_sanitize_boxes_cxcywh_normalized_min_size_px_clamps_to_unit() -> None:
    # If min_size_px exceeds the image dimension, the clamp saturates at 1.0.
    sanitized = sanitize_boxes_cxcywh_normalized(
        torch.tensor([[0.5, 0.5, 0.01, 0.01]]),
        image_size=(8, 8),
        min_size_px=64.0,
    )
    torch.testing.assert_close(sanitized, torch.tensor([[0.5, 0.5, 1.0, 1.0]]))


def test_sanitize_boxes_cxcywh_normalized_min_size_px_zero_is_noop() -> None:
    boxes = torch.tensor([[0.5, 0.5, 0.001, 0.001]], dtype=torch.float32)
    sanitized = sanitize_boxes_cxcywh_normalized(
        boxes, image_size=(640, 640), min_size_px=0.0
    )
    torch.testing.assert_close(sanitized, boxes)
