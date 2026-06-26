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

from lightly_train._task_models.depth_estimation import transforms
from lightly_train._task_models.depth_estimation.transforms import (
    DepthEstimationTrainTransform,
    DepthEstimationTrainTransformArgs,
)
from lightly_train._transforms.transform import RandomCropArgs, RandomFlipArgs


def _build_transform(
    *,
    image_size: tuple[int, int] = (28, 28),
    random_flip: RandomFlipArgs | None = None,
    random_crop: RandomCropArgs | None = None,
) -> DepthEstimationTrainTransform:
    args = DepthEstimationTrainTransformArgs(
        image_size=image_size,
        random_flip=random_flip,
        random_crop=random_crop,
    )
    args.resolve_auto(model_init_args={})
    args.resolve_incompatible()
    return DepthEstimationTrainTransform(transform_args=args)


class TestDepthEstimationTransform:
    def test___call___hflip_syncs_image_depth_sky(self) -> None:
        # With a guaranteed horizontal flip, image, depth and sky must all be flipped.
        transform = _build_transform(
            image_size=(28, 28),
            random_flip=RandomFlipArgs(horizontal_prob=1.0, vertical_prob=0.0),
        )
        image = (np.random.rand(28, 28, 3) * 255).astype(np.uint8)
        depth = np.arange(28 * 28, dtype=np.float32).reshape(28, 28)
        sky = (np.arange(28 * 28, dtype=np.float32).reshape(28, 28) / (28 * 28)).astype(
            np.float32
        )

        out = transform({"image": image, "depth": depth, "sky": sky})

        expected_depth = np.ascontiguousarray(depth[:, ::-1])
        expected_sky = np.ascontiguousarray(sky[:, ::-1])
        assert np.allclose(out["depth"][0].numpy(), expected_depth)
        assert np.allclose(out["sky"][0].numpy(), expected_sky)

    def test___call___random_crop_syncs_image_depth_sky(self) -> None:
        transform = _build_transform(
            image_size=(28, 28),
            random_crop=RandomCropArgs(
                height=14,
                width=14,
                pad_position="center",
                pad_if_needed=True,
                fill=0.0,
                prob=1.0,
            ),
        )
        image = (np.random.rand(28, 28, 3) * 255).astype(np.uint8)
        depth = np.random.rand(28, 28).astype(np.float32) + 0.5
        sky = np.random.rand(28, 28).astype(np.float32)

        out = transform({"image": image, "depth": depth, "sky": sky})

        assert out["image"].shape == (3, 14, 14)
        assert out["depth"].shape == (1, 14, 14)
        assert out["sky"].shape == (1, 14, 14)

    def test___call___depth_and_sky_not_normalized(self) -> None:
        # Normalize only applies to the image; depth and sky pass through unchanged
        # (up to the resize, which here is a no-op since input already matches).
        transform = _build_transform(image_size=(28, 28))
        image = np.full((28, 28, 3), 128, dtype=np.uint8)
        depth = np.full((28, 28), 5.0, dtype=np.float32)
        sky = np.full((28, 28), 0.7, dtype=np.float32)

        out = transform({"image": image, "depth": depth, "sky": sky})

        # Depth and sky keep their original values.
        assert np.allclose(out["depth"].numpy(), 5.0)
        assert np.allclose(out["sky"].numpy(), 0.7)
        # The image is normalized, so its values are no longer the raw 128/255.
        assert not np.allclose(out["image"].numpy(), 128.0 / 255.0)


def test__resolve_depth_transform_auto__multiple_of_patch_size() -> None:
    # The base 504x504 and a non-square size from the paper both resolve without error.
    for image_size in [(504, 504), (504, 378)]:
        args = DepthEstimationTrainTransformArgs(image_size=image_size)
        transforms._resolve_depth_transform_auto(args, model_init_args={})
        assert args.image_size == image_size


def test__resolve_depth_transform_auto__rejects_non_multiple_of_patch_size() -> None:
    args = DepthEstimationTrainTransformArgs(image_size=(500, 500))
    with pytest.raises(ValueError, match="must be a multiple of the patch size 14"):
        transforms._resolve_depth_transform_auto(args, model_init_args={})
