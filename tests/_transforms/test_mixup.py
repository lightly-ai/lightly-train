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

from lightly_train._transforms.mixup import MixUp


def _make_item(
    *,
    image: NDArray[np.uint8],
    bboxes: NDArray[np.float64],
    class_labels: NDArray[np.int64],
    masks: NDArray[np.uint8] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "image": image,
        "bboxes": bboxes,
        "class_labels": class_labels,
    }
    if masks is not None:
        item["masks"] = masks
    return item


class TestMixUp:
    def test__call__returns_input_unchanged_when_batch_too_small(self) -> None:
        transform = MixUp()

        empty_batch: list[dict[str, Any]] = []
        assert transform(batch=empty_batch) is empty_batch

        single_batch = [
            _make_item(
                image=np.zeros((32, 32, 3), dtype=np.uint8),
                bboxes=np.zeros((0, 4), dtype=np.float64),
                class_labels=np.zeros((0,), dtype=np.int64),
            )
        ]
        assert transform(batch=single_batch) is single_batch

    def test__call__preserves_shape_and_length(self) -> None:
        np.random.seed(0)
        image_shape = (32, 32, 3)
        item_a = _make_item(
            image=np.full(image_shape, 10, dtype=np.uint8),
            bboxes=np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
            class_labels=np.array([0], dtype=np.int64),
        )
        item_b = _make_item(
            image=np.full(image_shape, 200, dtype=np.uint8),
            bboxes=np.array(
                [[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.1, 0.1]], dtype=np.float64
            ),
            class_labels=np.array([1, 2], dtype=np.int64),
        )
        batch = [item_a, item_b]

        transform = MixUp()
        result = transform(batch=batch)

        assert len(result) == 2
        # Item 0 gets its own bboxes + item 1's bboxes (shifted_batch starts with last).
        assert result[0]["image"].shape == image_shape
        assert result[0]["bboxes"].shape == (1 + 2, 4)
        assert result[0]["class_labels"].shape == (1 + 2,)
        # Item 1 gets its own bboxes + item 0's bboxes.
        assert result[1]["image"].shape == image_shape
        assert result[1]["bboxes"].shape == (2 + 1, 4)
        assert result[1]["class_labels"].shape == (2 + 1,)

    def test__call__concatenates_correct_neighbor(self) -> None:
        np.random.seed(0)
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        item_a = _make_item(
            image=image.copy(),
            bboxes=np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
            class_labels=np.array([0], dtype=np.int64),
        )
        item_b = _make_item(
            image=image.copy(),
            bboxes=np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float64),
            class_labels=np.array([1], dtype=np.int64),
        )
        batch = [item_a, item_b]

        transform = MixUp()
        result = transform(batch=batch)

        # shifted_batch = batch[-1:] + batch[:-1] = [item_b, item_a]
        # so result[0] = own (item_a) + shifted[0] (item_b) -> labels [0, 1]
        # and result[1] = own (item_b) + shifted[1] (item_a) -> labels [1, 0]
        assert list(result[0]["class_labels"]) == [0, 1]
        assert list(result[1]["class_labels"]) == [1, 0]

    def test__call__omits_masks_when_absent(self) -> None:
        np.random.seed(0)
        image_shape = (32, 32, 3)
        batch = [
            _make_item(
                image=np.full(image_shape, 10, dtype=np.uint8),
                bboxes=np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
                class_labels=np.array([0], dtype=np.int64),
            ),
            _make_item(
                image=np.full(image_shape, 200, dtype=np.uint8),
                bboxes=np.array([[0.25, 0.25, 0.2, 0.2]], dtype=np.float64),
                class_labels=np.array([1], dtype=np.int64),
            ),
        ]

        transform = MixUp()
        result = transform(batch=batch)

        # Object-detection path: no masks key in, no masks key out.
        assert all("masks" not in item for item in result)
        assert set(result[0].keys()) == {"image", "bboxes", "class_labels"}

    def test__call__concatenates_masks_with_correct_neighbor(self) -> None:
        np.random.seed(0)
        image_shape = (8, 8, 3)
        mask_a = np.ones((1, 8, 8), dtype=np.uint8)
        mask_b = np.zeros((2, 8, 8), dtype=np.uint8)
        item_a = _make_item(
            image=np.full(image_shape, 10, dtype=np.uint8),
            bboxes=np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
            class_labels=np.array([0], dtype=np.int64),
            masks=mask_a,
        )
        item_b = _make_item(
            image=np.full(image_shape, 200, dtype=np.uint8),
            bboxes=np.array(
                [[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.1, 0.1]], dtype=np.float64
            ),
            class_labels=np.array([1, 2], dtype=np.int64),
            masks=mask_b,
        )
        batch = [item_a, item_b]

        transform = MixUp()
        result = transform(batch=batch)

        # shifted_batch = [item_b, item_a].
        # result[0] = own (item_a, 1 mask) + shifted (item_b, 2 masks) -> 3 masks.
        assert result[0]["masks"].shape == (1 + 2, 8, 8)
        # Masks stay index-aligned with the concatenated class_labels.
        assert result[0]["masks"].shape[0] == result[0]["class_labels"].shape[0]
        # Masks are unioned, not blended: own first, then neighbor.
        np.testing.assert_array_equal(
            result[0]["masks"], np.concatenate((mask_a, mask_b), axis=0)
        )

        # result[1] = own (item_b, 2 masks) + shifted (item_a, 1 mask) -> 3 masks.
        assert result[1]["masks"].shape == (2 + 1, 8, 8)
        np.testing.assert_array_equal(
            result[1]["masks"], np.concatenate((mask_b, mask_a), axis=0)
        )

    def test__call__handles_empty_masks(self) -> None:
        np.random.seed(0)
        image_shape = (8, 8, 3)
        empty_bboxes = np.zeros((0, 4), dtype=np.float64)
        empty_labels = np.zeros((0,), dtype=np.int64)
        empty_masks = np.zeros((0, 8, 8), dtype=np.uint8)
        batch = [
            _make_item(
                image=np.full(image_shape, 10, dtype=np.uint8),
                bboxes=empty_bboxes.copy(),
                class_labels=empty_labels.copy(),
                masks=empty_masks.copy(),
            ),
            _make_item(
                image=np.full(image_shape, 200, dtype=np.uint8),
                bboxes=empty_bboxes.copy(),
                class_labels=empty_labels.copy(),
                masks=empty_masks.copy(),
            ),
        ]

        transform = MixUp()
        result = transform(batch=batch)

        assert result[0]["masks"].shape == (0, 8, 8)
        assert result[1]["masks"].shape == (0, 8, 8)
