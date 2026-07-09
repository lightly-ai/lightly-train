#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import numpy as np

from lightly_train._transforms.ltdetr_transforms.utils import (
    filter_degenerate_yolo_boxes,
    normalize_bboxes_and_labels,
)


class TestFilterDegenerateYoloBoxes:
    def test_drops_zero_area_boxes(self) -> None:
        # YOLO format (cx, cy, w, h): rows 1 and 3 are degenerate (w or h == 0).
        bboxes = np.array(
            [
                [0.5, 0.5, 0.4, 0.4],
                [0.5, 0.5, 0.0, 0.4],
                [0.2, 0.2, 0.1, 0.1],
                [0.5, 0.5, 0.4, 0.0],
            ],
            dtype=np.float64,
        )
        class_labels = np.array([1, 2, 3, 4], dtype=np.int64)

        out_bboxes, out_labels, out_indices = filter_degenerate_yolo_boxes(
            bboxes=bboxes,
            class_labels=class_labels,
        )

        np.testing.assert_array_equal(out_bboxes, bboxes[[0, 2]])
        np.testing.assert_array_equal(out_labels, np.array([1, 3], dtype=np.int64))
        assert out_indices is None

    def test_filters_indices_in_lockstep(self) -> None:
        # The instance segmentation transform relies on the ``indices`` array being
        # filtered together with the boxes so masks stay aligned.
        bboxes = np.array(
            [
                [0.5, 0.5, 0.4, 0.4],
                [0.5, 0.5, 0.0, 0.4],
                [0.2, 0.2, 0.1, 0.1],
            ],
            dtype=np.float64,
        )
        class_labels = np.array([1, 2, 3], dtype=np.int64)
        indices = np.array([10, 11, 12], dtype=np.int64)

        out_bboxes, out_labels, out_indices = filter_degenerate_yolo_boxes(
            bboxes=bboxes,
            class_labels=class_labels,
            indices=indices,
        )

        np.testing.assert_array_equal(out_bboxes, bboxes[[0, 2]])
        np.testing.assert_array_equal(out_labels, np.array([1, 3], dtype=np.int64))
        assert out_indices is not None
        np.testing.assert_array_equal(out_indices, np.array([10, 12], dtype=np.int64))

    def test_empty_input(self) -> None:
        bboxes = np.zeros((0, 4), dtype=np.float64)
        class_labels = np.zeros((0,), dtype=np.int64)
        indices = np.zeros((0,), dtype=np.int64)

        out_bboxes, out_labels, out_indices = filter_degenerate_yolo_boxes(
            bboxes=bboxes,
            class_labels=class_labels,
            indices=indices,
        )

        assert out_bboxes.shape == (0, 4)
        assert out_labels.shape == (0,)
        assert out_indices is not None
        assert out_indices.shape == (0,)


class TestNormalizeBboxesAndLabels:
    def test_converts_lists_to_arrays(self) -> None:
        bboxes, class_labels = normalize_bboxes_and_labels([[0.5, 0.5, 0.4, 0.4]], [1])
        assert isinstance(bboxes, np.ndarray)
        assert isinstance(class_labels, np.ndarray)
        np.testing.assert_array_equal(
            bboxes, np.array([[0.5, 0.5, 0.4, 0.4]], dtype=np.float64)
        )
        np.testing.assert_array_equal(class_labels, np.array([1]))

    def test_passes_arrays_through(self) -> None:
        bboxes_in = np.array([[0.5, 0.5, 0.4, 0.4]], dtype=np.float64)
        class_labels_in = np.array([1], dtype=np.int64)

        bboxes, class_labels = normalize_bboxes_and_labels(bboxes_in, class_labels_in)

        assert bboxes is bboxes_in
        assert class_labels is class_labels_in
