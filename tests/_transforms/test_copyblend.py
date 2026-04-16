#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from lightly_train._transforms.copyblend import CopyBlend


def _make_item(
    *,
    image: NDArray[np.uint8],
    bboxes: NDArray[np.float64],
    class_labels: NDArray[np.int64],
) -> dict[str, Any]:
    return {"image": image, "bboxes": bboxes, "class_labels": class_labels}


class TestCopyBlend:
    def test__call__returns_input_unchanged_when_pool_empty(self) -> None:
        np.random.seed(0)
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        empty_bboxes = np.zeros((0, 4), dtype=np.float64)
        empty_labels = np.zeros((0,), dtype=np.int64)
        batch = [
            _make_item(
                image=image.copy(), bboxes=empty_bboxes, class_labels=empty_labels
            ),
            _make_item(
                image=image.copy(), bboxes=empty_bboxes, class_labels=empty_labels
            ),
        ]

        transform = CopyBlend(area_threshold=1, num_objects=1, expand_ratios=(0.0, 0.0))
        result = transform(batch=batch)

        assert result is batch

    def test__call__returns_input_unchanged_when_all_boxes_below_area_threshold(
        self,
    ) -> None:
        np.random.seed(0)
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        bboxes = np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64)
        labels = np.array([1], dtype=np.int64)
        batch = [
            _make_item(image=image.copy(), bboxes=bboxes, class_labels=labels),
            _make_item(image=image.copy(), bboxes=bboxes, class_labels=labels),
        ]

        # area_threshold is much larger than any possible object area in a 32x32 image.
        transform = CopyBlend(
            area_threshold=10_000, num_objects=1, expand_ratios=(0.0, 0.0)
        )
        result = transform(batch=batch)

        assert result is batch

    def test__call__blends_and_appends_boxes(self) -> None:
        np.random.seed(0)
        image_a = np.full((32, 32, 3), 50, dtype=np.uint8)
        image_b = np.full((32, 32, 3), 200, dtype=np.uint8)
        bbox = np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float64)
        label = np.array([3], dtype=np.int64)
        batch = [
            _make_item(image=image_a, bboxes=bbox.copy(), class_labels=label.copy()),
            _make_item(image=image_b, bboxes=bbox.copy(), class_labels=label.copy()),
        ]

        transform = CopyBlend(
            area_threshold=1,
            num_objects=1,
            expand_ratios=(0.0, 0.0),
            beta_range=(0.5, 0.5),
        )
        result = transform(batch=batch)

        assert len(result) == 2
        for original, updated in zip(batch, result):
            assert updated["image"].shape == original["image"].shape
            assert updated["image"].dtype == original["image"].dtype

            assert updated["bboxes"].shape == (2, 4)
            assert updated["bboxes"].dtype == np.float64
            appended_bbox = updated["bboxes"][-1]
            assert appended_bbox.shape == (4,)
            assert np.all(appended_bbox >= 0.0)
            assert np.all(appended_bbox <= 1.0)

            assert updated["class_labels"].shape == (2,)
            assert updated["class_labels"].dtype == np.int64
