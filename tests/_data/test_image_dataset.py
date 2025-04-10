#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path
from typing import Iterable

import pytest
from torch import Tensor
from torch.utils.data import DataLoader

from lightly_train._data import image_dataset
from lightly_train._data.image_dataset import ImageDataset
from lightly_train.types import (
    Batch,
    DatasetItem,
    ImageFilename,
)

from .. import helpers
from ..helpers import DummyMethodTransform


@pytest.fixture
def flat_image_dir(tmp_path: Path) -> Path:
    return _create_images(tmp_path=tmp_path, filenames=["image1.jpg", "image2.jpg"])


@pytest.fixture
def image_dir_being_symlink(tmp_path: Path) -> Path:
    """Returns a path being a symlink to a normal image directory."""
    path_original = _create_images(
        tmp_path=tmp_path, filenames=["image1.jpg", "image2.jpg"]
    )
    path_symlink = tmp_path / "contains_symlinks"
    path_symlink.symlink_to(path_original)
    return path_symlink


@pytest.fixture
def image_dir_containing_symlinks(tmp_path: Path) -> Path:
    """
    Returns a path not being a symlink but containing a symlink to a normal image
    directory.
    """
    path_original = _create_images(
        tmp_path=tmp_path, filenames=["image1.jpg", "image2.jpg"]
    )
    path_containing_symlink = tmp_path / "contains_symlinks"
    path_containing_symlink.mkdir(parents=True, exist_ok=True)
    link_source = path_containing_symlink / "link"
    link_source.symlink_to(path_original)
    return path_containing_symlink


@pytest.fixture
def nested_image_dir(tmp_path: Path) -> Path:
    return _create_images(
        tmp_path=tmp_path, filenames=["class1/image1.jpg", "class2/image2.jpg"]
    )


class TestImageDataset:
    def test___getitem__(self, flat_image_dir: Path) -> None:
        filenames = [ImageFilename("image1.jpg"), ImageFilename("image2.jpg")]
        dataset = ImageDataset(
            image_dir=flat_image_dir,
            image_filenames=filenames,
            transform=DummyMethodTransform(),
        )
        assert len(dataset) == 2
        for i in range(2):
            item: DatasetItem = dataset[i]
            assert isinstance(item, dict)
            assert item["filename"] == filenames[i]
            assert isinstance(item["views"], list)
            assert len(item["views"]) == 1
            assert isinstance(item["views"][0], Tensor)
            assert item["views"][0].shape == (3, 32, 32)
            assert "mask" not in item

    @pytest.mark.parametrize(
        "mode, extension",
        [
            ("1", "png"),  # Monochrome
            ("L", "png"),  # Grayscale
            ("P", "png"),  # Palette
            ("RGB", "jpeg"),  # True color
            ("RGBA", "png"),  # True color with transparency
            ("CMYK", "tiff"),  # CMYK color separation
            ("YCbCr", "jpeg"),  # YCbCr (JPEG)
            ("LAB", "tiff"),  # L*a*b* color space
            ("I", "tiff"),  # 32-bit signed integer
            ("F", "tiff"),  # 32-bit floating point
            ("LA", "png"),  # Grayscale with alpha
        ],
    )
    def test___getitem____mode(self, tmp_path: Path, mode: str, extension: str) -> None:
        filenames = [ImageFilename(f"image1.{extension}")]
        image_dir = _create_images(tmp_path=tmp_path, filenames=filenames, mode=mode)
        dataset = ImageDataset(
            image_dir=image_dir,
            image_filenames=filenames,
            transform=DummyMethodTransform(),
        )
        image = dataset[0]["views"][0]
        assert isinstance(image, Tensor)
        assert image.shape == (3, 32, 32)

    def test___getitem____truncated(self, tmp_path: Path) -> None:
        filenames = [ImageFilename("image1.jpg")]
        image_dir = _create_images(tmp_path=tmp_path, filenames=filenames)

        # Truncate the image by 10 bytes.
        first_image_path = next(image_dir.glob("*.jpg"))
        with open(first_image_path, "rb") as f:
            first_image = f.read()
        with open(first_image_path, "wb") as f:
            f.write(first_image[:-10])

        dataset = ImageDataset(
            image_dir=image_dir,
            image_filenames=filenames,
            transform=DummyMethodTransform(),
        )
        image = dataset[0]["views"][0]
        assert isinstance(image, Tensor)
        assert image.shape == (3, 32, 32)

    def test___getitem____masks(self, tmp_path: Path) -> None:
        img_filenames = [ImageFilename("image1.jpg")]
        mask_filenames = [ImageFilename("image1.png")]
        image_dir = _create_images(
            tmp_path=tmp_path / "images", filenames=img_filenames
        )
        mask_dir = _create_images(
            tmp_path=tmp_path / "masks", filenames=mask_filenames, mode="L"
        )
        dataset = ImageDataset(
            image_dir=image_dir,
            image_filenames=img_filenames,
            mask_dir=mask_dir,
            transform=DummyMethodTransform(),
        )
        item: DatasetItem = dataset[0]
        print(f"{item=}")
        assert isinstance(item, dict)
        assert isinstance(item["views"], list)
        assert len(item["views"]) == 1
        view1 = item["views"][0]
        assert isinstance(view1, Tensor)
        assert view1.shape == (3, 32, 32)
        assert isinstance(item["masks"], list)
        assert len(item["masks"]) == 1
        mask1 = item["masks"][0]
        assert isinstance(mask1, Tensor)
        assert mask1.shape == (32, 32)

    def test_dataloader(self, flat_image_dir: Path) -> None:
        filenames = [ImageFilename("image1.jpg"), ImageFilename("image2.jpg")]
        dataset = ImageDataset(
            image_dir=flat_image_dir,
            image_filenames=filenames,
            transform=DummyMethodTransform(),
        )
        assert len(dataset) == 2
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=2,
            num_workers=0,
            shuffle=False,
        )
        batch: Batch
        for batch in dataloader:
            assert isinstance(batch["views"], list)
            assert len(batch["views"]) == 1
            view_1 = batch["views"][0]
            assert isinstance(view_1, Tensor)
            assert view_1.shape == (2, 3, 32, 32)

            assert batch["filename"] == filenames


def test_list_image_filenames__flat(flat_image_dir: Path) -> None:
    filenames = image_dataset.list_image_filenames(image_dir=flat_image_dir)
    assert sorted(list(filenames)) == [
        "image1.jpg",
        "image2.jpg",
    ]


def test_list_image_filenames__dir_is_symlink(image_dir_being_symlink: Path) -> None:
    filenames = image_dataset.list_image_filenames(image_dir=image_dir_being_symlink)
    assert sorted(list(filenames)) == [
        "image1.jpg",
        "image2.jpg",
    ]


def test_list_image_filenames__dir_contains_symlinks(
    image_dir_containing_symlinks: Path,
) -> None:
    filenames = image_dataset.list_image_filenames(
        image_dir=image_dir_containing_symlinks
    )
    assert sorted(list(filenames)) == [
        "link/image1.jpg",
        "link/image2.jpg",
    ]


def test_list_image_filenames__nested(nested_image_dir: Path) -> None:
    filenames = image_dataset.list_image_filenames(image_dir=nested_image_dir)
    assert sorted(list(filenames)) == [
        "class1/image1.jpg",
        "class2/image2.jpg",
    ]


@pytest.mark.parametrize(
    "extension",
    # The list below is not exhaustive. Some image types require extra handlers installed.
    [
        ".bmp",
        ".dib",
        ".pcx",
        ".dds",
        ".ps",
        ".eps",
        ".gif",
        ".png",
        ".apng",
        ".jp2",
        ".j2k",
        ".jpc",
        ".jpf",
        ".jpx",
        ".j2c",
        ".icns",
        ".ico",
        ".im",
        ".jfif",
        ".jpe",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".pbm",
        ".pgm",
        ".ppm",
        ".pnm",
        ".bw",
        ".rgb",
        ".rgba",
        ".sgi",
        ".tga",
        ".icb",
        ".vda",
        ".vst",
        ".webp",
    ],
)
def test_list_image_filenames__extensions(extension: str, tmp_path: Path) -> None:
    image_dir = _create_images(
        tmp_path=tmp_path, filenames=[f"image{i}{extension}" for i in range(1, 3)]
    )
    filenames = image_dataset.list_image_filenames(image_dir=image_dir)
    assert sorted(list(filenames)) == [
        f"image1{extension}",
        f"image2{extension}",
    ]


def _create_images(tmp_path: Path, filenames: Iterable[str], mode: str = "RGB") -> Path:
    image_dir = tmp_path / "images"
    helpers.create_images(
        image_dir=image_dir, files=filenames, height=32, width=32, mode=mode
    )
    return image_dir
