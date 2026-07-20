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
