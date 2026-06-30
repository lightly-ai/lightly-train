#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lightly_train._data.depth_estimation_dataset import (
    DepthEstimationDataArgs,
    DepthEstimationDataset,
    SplitArgs,
)
from lightly_train._task_models.depth_estimation.transforms import (
    DepthEstimationTrainTransform,
    DepthEstimationTrainTransformArgs,
)


def _make_dataset(tmp_path: Path, *, height: int, width: int) -> DepthEstimationDataset:
    image_dir = tmp_path / "images"
    depth_dir = tmp_path / "depth"
    sky_dir = tmp_path / "sky"
    for d in (image_dir, depth_dir, sky_dir):
        d.mkdir()
    Image.fromarray((np.random.rand(height, width, 3) * 255).astype(np.uint8)).save(
        image_dir / "a.png"
    )
    np.save(depth_dir / "a.npy", np.random.rand(height, width).astype(np.float32) + 0.5)
    Image.fromarray(
        (np.random.rand(height, width) > 0.5).astype(np.uint8) * 255, mode="L"
    ).save(sky_dir / "a.png")

    data_args = DepthEstimationDataArgs(
        train=SplitArgs(images=image_dir, depth=depth_dir, sky=sky_dir),
        val=SplitArgs(images=image_dir, depth=depth_dir, sky=sky_dir),
    )
    dataset_args = data_args.get_train_args()
    image_info = list(dataset_args.list_image_info())

    transform_args = DepthEstimationTrainTransformArgs(image_size=(28, 28))
    transform_args.resolve_auto(model_init_args={})
    transform_args.resolve_incompatible()
    return DepthEstimationDataset(
        dataset_args=dataset_args,
        image_info=image_info,
        transform=DepthEstimationTrainTransform(transform_args=transform_args),
    )


class TestDepthEstimationDataset:
    def test___getitem__(self, tmp_path: Path) -> None:
        dataset = _make_dataset(tmp_path, height=40, width=50)

        item = dataset[0]

        assert item["image"].shape == (3, 28, 28)
        assert item["depth"].shape == (1, 28, 28)
        assert item["sky"].shape == (1, 28, 28)
        assert item["image_path"].endswith("a.png")


class TestDepthEstimationDataArgs:
    def test_included_classes__empty(self, tmp_path: Path) -> None:
        data_args = DepthEstimationDataArgs(
            train=SplitArgs(images=tmp_path, depth=tmp_path, sky=tmp_path),
            val=SplitArgs(images=tmp_path, depth=tmp_path, sky=tmp_path),
        )

        assert data_args.included_classes == {}

    def test_list_image_info__skips_unpaired_images(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        depth_dir = tmp_path / "depth"
        sky_dir = tmp_path / "sky"
        for d in (image_dir, depth_dir, sky_dir):
            d.mkdir()
        # Two images, but only one has matching depth and sky labels.
        for name in ("a.png", "b.png"):
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / name)
        np.save(depth_dir / "a.npy", np.ones((8, 8), dtype=np.float32))
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(
            sky_dir / "a.png"
        )

        data_args = DepthEstimationDataArgs(
            train=SplitArgs(images=image_dir, depth=depth_dir, sky=sky_dir),
            val=SplitArgs(images=image_dir, depth=depth_dir, sky=sky_dir),
        )
        info = list(data_args.get_train_args().list_image_info())

        assert len(info) == 1
        assert info[0]["image_filepaths"].endswith("a.png")


@pytest.mark.parametrize("missing", ["depth", "sky"])
def test_depth_estimation_dataset___getitem___shape_mismatch(
    tmp_path: Path, missing: str
) -> None:
    image_dir = tmp_path / "images"
    depth_dir = tmp_path / "depth"
    sky_dir = tmp_path / "sky"
    for d in (image_dir, depth_dir, sky_dir):
        d.mkdir()
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_dir / "a.png")
    # Write a mismatched label for the parametrized target.
    depth_shape = (8, 8) if missing == "depth" else (16, 16)
    sky_shape = (8, 8) if missing == "sky" else (16, 16)
    np.save(depth_dir / "a.npy", np.ones(depth_shape, dtype=np.float32))
    Image.fromarray(np.zeros(sky_shape, dtype=np.uint8), mode="L").save(
        sky_dir / "a.png"
    )

    data_args = DepthEstimationDataArgs(
        train=SplitArgs(images=image_dir, depth=depth_dir, sky=sky_dir),
        val=SplitArgs(images=image_dir, depth=depth_dir, sky=sky_dir),
    )
    dataset_args = data_args.get_train_args()
    transform_args = DepthEstimationTrainTransformArgs(image_size=(16, 16))
    transform_args.resolve_auto(model_init_args={})
    transform_args.resolve_incompatible()
    dataset = DepthEstimationDataset(
        dataset_args=dataset_args,
        image_info=list(dataset_args.list_image_info()),
        transform=DepthEstimationTrainTransform(transform_args=transform_args),
    )

    with pytest.raises(ValueError, match="Shape mismatch"):
        dataset[0]
