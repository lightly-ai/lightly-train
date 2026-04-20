#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import random

import numpy as np

from lightly_train._transforms.mosaic import MosaicTransform


def _make_transform(
    *,
    output_size: int = 32,
    max_size: int | None = None,
    rotation_range: float = 0.0,
    translation_range: tuple[float, float] = (0.0, 0.0),
    scaling_range: tuple[float, float] = (1.0, 1.0),
    fill_value: int = 0,
    max_cached_images: int = 50,
    random_pop: bool = False,
) -> MosaicTransform:
    return MosaicTransform(
        output_size=output_size,
        max_size=max_size,
        rotation_range=rotation_range,
        translation_range=translation_range,
        scaling_range=scaling_range,
        fill_value=fill_value,
        max_cached_images=max_cached_images,
        random_pop=random_pop,
    )


class TestMosaicTransform:
    def test__call__output_shapes_and_dtypes(self) -> None:
        np.random.seed(0)
        random.seed(0)
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        bboxes = np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float64)
        labels = np.array([1], dtype=np.int64)

        transform = _make_transform()
        out_image, out_bboxes, out_labels = transform(image, bboxes, labels)

        assert isinstance(out_image, np.ndarray)
        assert out_image.dtype == np.uint8
        assert out_image.ndim == 3
        assert out_image.shape[2] == 3

        assert out_bboxes.dtype == np.float64
        assert out_bboxes.ndim == 2
        assert out_bboxes.shape[1] == 4
        assert np.all(out_bboxes >= 0.0)
        assert np.all(out_bboxes <= 1.0)

        assert out_labels.dtype == np.int64

    def test__call__handles_empty_bboxes(self) -> None:
        np.random.seed(0)
        random.seed(0)
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        bboxes = np.zeros((0, 4), dtype=np.float64)
        labels = np.zeros((0,), dtype=np.int64)

        transform = _make_transform()
        out_image, out_bboxes, out_labels = transform(image, bboxes, labels)

        assert out_image.dtype == np.uint8
        assert out_image.ndim == 3 and out_image.shape[2] == 3
        assert out_bboxes.shape == (0, 4)
        assert out_labels.shape == (0,)

    def test__call__float32_input(self) -> None:
        np.random.seed(0)
        random.seed(0)
        image = np.random.rand(64, 64, 3).astype(np.float32)
        bboxes = np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float64)
        labels = np.array([0], dtype=np.int64)

        transform = _make_transform()
        out_image, out_bboxes, out_labels = transform(image, bboxes, labels)

        assert out_image.dtype == np.float32
        assert out_image.ndim == 3 and out_image.shape[2] == 3
        assert out_bboxes.dtype == np.float64
        assert out_labels.dtype == np.int64

    def test__call__cache_is_populated_and_capped(self) -> None:
        np.random.seed(0)
        random.seed(0)
        bboxes = np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float64)
        labels = np.array([0], dtype=np.int64)

        transform = _make_transform(max_cached_images=2, random_pop=False)

        # Each call appends a resized image to the cache. With max_cached_images=2 and
        # random_pop=False (FIFO), the cache should never exceed 2 entries, and the
        # oldest entry gets evicted first.
        first_image = np.full((64, 64, 3), 10, dtype=np.uint8)
        transform(first_image, bboxes, labels)
        assert len(transform._mosaic_cache) <= 2

        transform(np.full((64, 64, 3), 50, dtype=np.uint8), bboxes, labels)
        assert len(transform._mosaic_cache) <= 2

        transform(np.full((64, 64, 3), 100, dtype=np.uint8), bboxes, labels)
        assert len(transform._mosaic_cache) <= 2

        transform(np.full((64, 64, 3), 200, dtype=np.uint8), bboxes, labels)
        assert len(transform._mosaic_cache) <= 2

        # The first item should have been evicted — none of the remaining cache items
        # should still have the pixel value of the first image.
        for cached in transform._mosaic_cache:
            assert not np.all(cached["img"] == 10)

    def test__call__identity_affine_preserves_boxes(self) -> None:
        np.random.seed(0)
        random.seed(0)
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        bboxes = np.array([[0.5, 0.5, 0.25, 0.25]], dtype=np.float64)
        labels = np.array([7], dtype=np.int64)

        transform = _make_transform(
            rotation_range=0.0,
            translation_range=(0.0, 0.0),
            scaling_range=(1.0, 1.0),
        )
        out_image, out_bboxes, out_labels = transform(image, bboxes, labels)

        # The first call has an empty cache, so all four mosaic slots are filled
        # with copies of the single input sample (one bbox each).
        assert out_bboxes.shape == (4, 4)
        assert out_bboxes.dtype == np.float64
        assert np.all(out_bboxes >= 0.0)
        assert np.all(out_bboxes <= 1.0)
        assert out_labels.shape == (4,)
        assert out_labels.dtype == np.int64
        assert np.all(out_labels == 7)
        assert out_image.dtype == np.uint8
