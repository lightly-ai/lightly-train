#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
import pytest
from lightning_utilities.core.imports import RequirementCache

from lightly_train._transforms.random_zoom_out import RandomZoomOut

ALBUMENTATIONS_GEQ_1_4_15 = RequirementCache("albumentations>=1.4.15")


class TestRandomZoomOut:
    @pytest.mark.parametrize(
        "side_range, fill, error",
        [
            ((0.5, 2.0), 0.0, "side_range must be"),
            ((2.0, 1.0), 0.0, "side_range must be"),
        ],
    )
    def test__init__errors(
        self,
        side_range: tuple[float, float],
        fill: float,
        error: str,
    ) -> None:
        with pytest.raises(ValueError, match=error):
            RandomZoomOut(side_range=side_range, fill=fill, p=1.0)

    def test__call__check_return_shapes(self) -> None:
        img = np.ones((16, 16, 3), dtype=np.uint8) * 127
        mask = np.ones((16, 16), dtype=np.uint8) * 127
        bboxes = np.array([[4, 4, 12, 12]], dtype=np.float32)
        class_labels = np.array([1], dtype=np.int32)

        transform = RandomZoomOut(side_range=(1.0, 2.0), fill=0, p=1.0)
        out = transform(image=img, mask=mask, bboxes=bboxes, class_labels=class_labels)

        # Must be bigger since we zoomed out.
        assert out["image"].shape[0] >= img.shape[0]
        assert out["image"].shape[1] >= img.shape[1]

        # Check that mask was equally padded.
        assert out["mask"].shape[0] == out["image"].shape[0]
        assert out["mask"].shape[1] == out["image"].shape[1]

        if ALBUMENTATIONS_GEQ_1_4_15:
            assert out["bboxes"].shape == bboxes.shape
            assert out["class_labels"].shape == class_labels.shape
        else:
            assert all(len(elem) == 4 for elem in out["bboxes"])
            assert len(out["class_labels"]) == len(out["bboxes"])

    def test__call__no_transform_when_p0(self) -> None:
        img = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
        mask = np.random.randint(0, 255, size=(8, 8), dtype=np.uint8)
        bboxes = np.array([[1, 1, 2, 2]], dtype=np.float32)
        class_labels = np.array([1], dtype=np.int32)

        transform = RandomZoomOut(side_range=(1.0, 2.0), fill=0, p=0.0)
        out = transform(image=img, mask=mask, bboxes=bboxes, class_labels=class_labels)

        np.testing.assert_array_equal(out["image"], img)
        np.testing.assert_array_equal(out["mask"], mask)
        np.testing.assert_array_equal(out["bboxes"], bboxes)
        np.testing.assert_array_equal(out["class_labels"], class_labels)

    def test__call__always_transform_when_p1(self) -> None:
        img = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
        mask = np.random.randint(0, 255, size=(8, 8), dtype=np.uint8)
        bboxes = np.array([[1, 1, 2, 2]], dtype=np.float32)
        class_labels = np.array([1], dtype=np.int32)

        transform = RandomZoomOut(side_range=(1.5, 2.0), fill=0, p=1.0)
        out = transform(image=img, mask=mask, bboxes=bboxes, class_labels=class_labels)

        # Should be larger than or equal to input size, and not exactly equal
        assert not np.array_equal(out["image"], img)
        assert not np.array_equal(out["mask"], mask)
        assert not np.array_equal(out["bboxes"], bboxes)
        # Class labels will stay, since we don't drop any boxes when zooming out.
        assert np.array_equal(out["class_labels"], class_labels)
