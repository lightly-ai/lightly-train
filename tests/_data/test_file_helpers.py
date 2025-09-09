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
from numpy.typing import DTypeLike
from pytest import LogCaptureFixture, MonkeyPatch
from pytest_mock import MockerFixture

from lightly_train._data import file_helpers
from lightly_train._data.file_helpers import ImageMode

from .. import helpers


def test_list_image_filenames_from_iterable(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # Change current working directory to allow relative paths to tmp_path.
    monkeypatch.chdir(tmp_path)

    helpers.create_images(
        image_dir=tmp_path,
        files=[
            "image1.jpg",
            "class1/image1.jpg",
            "class2/image2.jpg",
        ],
    )
    (tmp_path / "not_an_image.txt").touch()
    (tmp_path / "class2" / "not_an_image").touch()
    filenames = file_helpers.list_image_filenames_from_iterable(
        imgs_and_dirs=[
            "image1.jpg",  # relative image path
            tmp_path / "image2.jpg",  # absolute image path
            "class1",  # relative dir path
            tmp_path / "class2",  # absolute dir path
        ]
    )
    assert sorted(filenames) == sorted(
        [
            "image1.jpg",
            str(tmp_path / "image2.jpg"),
            str(Path("class1") / "image1.jpg"),
            str(tmp_path / "class2" / "image2.jpg"),
        ]
    )


@pytest.mark.parametrize("extension", helpers.SUPPORTED_IMAGE_EXTENSIONS)
def test_list_image_filenames_from_iterable__extensions(
    tmp_path: Path, extension: str
) -> None:
    helpers.create_images(image_dir=tmp_path, files=[f"image{extension}"])
    filenames = file_helpers.list_image_filenames_from_iterable(
        imgs_and_dirs=[tmp_path / f"image{extension}"]
    )
    assert list(filenames) == [str(tmp_path / f"image{extension}")]


def test_list_image_filenames_from_iterable__symlink(tmp_path: Path) -> None:
    helpers.create_images(
        image_dir=tmp_path / "symlinktarget",
        files=["image1.jpg", "class1/image1.jpg"],
    )
    helpers.create_images(
        image_dir=tmp_path / "symlinktarget2",
        files=["image2.jpg", "class2/image2.jpg"],
    )
    data_dir = tmp_path / "data"
    data_dir.symlink_to(tmp_path / "symlinktarget", target_is_directory=True)
    (data_dir / "image2.jpg").symlink_to(tmp_path / "symlinktarget2" / "image2.jpg")
    (data_dir / "class2").symlink_to(
        tmp_path / "symlinktarget2" / "class2", target_is_directory=True
    )
    filenames = file_helpers.list_image_filenames_from_iterable(
        imgs_and_dirs=[
            data_dir / "image1.jpg",
            data_dir / "class1",
            data_dir / "image2.jpg",
            data_dir / "class2",
        ]
    )
    assert sorted(filenames) == sorted(
        [
            str(data_dir / "image1.jpg"),
            str(data_dir / "class1" / "image1.jpg"),
            str(data_dir / "image2.jpg"),
            str(data_dir / "class2" / "image2.jpg"),
        ]
    )


def test_list_image_filenames_from_iterable__empty_dir(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with caplog.at_level(level="WARNING"):
        list(file_helpers.list_image_filenames_from_iterable(imgs_and_dirs=[empty_dir]))
    assert f"The directory '{empty_dir}' does not contain any images." in caplog.text


def test_list_image_filenames_from_iterable__invalid_path(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid_path"
    with pytest.raises(
        ValueError,
        match="Invalid path: '.*invalid_path'.",
    ):
        list(
            file_helpers.list_image_filenames_from_iterable(
                imgs_and_dirs=[invalid_path]
            )
        )


def test_list_image_filenames_from_dir(tmp_path: Path) -> None:
    helpers.create_images(
        image_dir=tmp_path,
        files=[
            "image1.jpg",
            "class1/image1.jpg",
            "class2/image2.jpg",
        ],
    )
    (tmp_path / "not_an_image.txt").touch()
    (tmp_path / "class2" / "not_an_image").touch()
    filenames = file_helpers.list_image_filenames_from_dir(image_dir=tmp_path)
    assert sorted(filenames) == sorted(
        [
            "image1.jpg",
            str(Path("class1") / "image1.jpg"),
            str(Path("class2") / "image2.jpg"),
        ]
    )


@pytest.mark.parametrize("extension", helpers.SUPPORTED_IMAGE_EXTENSIONS)
def test_list_image_filenames_from_dir__extensions(
    tmp_path: Path, extension: str
) -> None:
    helpers.create_images(image_dir=tmp_path, files=[f"image{extension}"])
    filenames = file_helpers.list_image_filenames_from_dir(image_dir=tmp_path)
    assert list(filenames) == [f"image{extension}"]


def test_list_image_filenames__symlink(tmp_path: Path) -> None:
    helpers.create_images(
        image_dir=tmp_path / "symlinktarget",
        files=["image1.jpg", "class1/image1.jpg"],
    )
    helpers.create_images(
        image_dir=tmp_path / "symlinktarget2",
        files=["image2.jpg", "class2/image2.jpg"],
    )
    data_dir = tmp_path / "data"
    data_dir.symlink_to(tmp_path / "symlinktarget", target_is_directory=True)
    (data_dir / "image2.jpg").symlink_to(tmp_path / "symlinktarget2" / "image2.jpg")
    (data_dir / "class2").symlink_to(
        tmp_path / "symlinktarget2" / "class2", target_is_directory=True
    )
    filenames = file_helpers.list_image_filenames_from_dir(image_dir=data_dir)
    assert sorted(filenames) == sorted(
        [
            "image1.jpg",
            str(Path("class1") / "image1.jpg"),
            "image2.jpg",
            str(Path("class2") / "image2.jpg"),
        ]
    )


@pytest.mark.parametrize(
    "extension, expected_backend",
    [
        (".jpg", "torch"),
        (".jpeg", "torch"),
        (".png", "torch"),
        (".bmp", "pil"),
        (".gif", "pil"),
        (".tiff", "pil"),
        (".webp", "pil"),
    ],
)
def test_open_image_numpy(
    tmp_path: Path, extension: str, expected_backend: str, mocker: MockerFixture
) -> None:
    image_path = tmp_path / f"image{extension}"
    helpers.create_image(path=image_path, height=32, width=32)

    torch_spy = mocker.spy(file_helpers, "_open_image_numpy__with_torch")
    pil_spy = mocker.spy(file_helpers, "_open_image_numpy__with_pil")

    result = file_helpers.open_image_numpy(image_path=image_path, mode=ImageMode.RGB)
    assert isinstance(result, np.ndarray)
    assert result.shape == (32, 32, 3)

    if expected_backend == "torch":
        torch_spy.assert_called_once()
        pil_spy.assert_not_called()
    else:
        pil_spy.assert_called_once()
        torch_spy.assert_not_called()


@pytest.mark.parametrize(
    "dtype, expected_dtype, mode, max_value",
    [(np.uint8, np.uint8, "L", 255), (np.uint16, np.int32, "I;16", 65535)],
)
def test_open_image_numpy__mask(
    tmp_path: Path,
    dtype: DTypeLike,
    expected_dtype: DTypeLike,
    mode: str,
    max_value: int,
) -> None:
    image_path = tmp_path / "image.png"
    helpers.create_image(
        path=image_path,
        height=32,
        width=32,
        mode=mode,
        max_value=max_value,
        dtype=dtype,
        num_channels=0,
    )

    result = file_helpers.open_image_numpy(image_path=image_path, mode=ImageMode.MASK)
    assert isinstance(result, np.ndarray)
    assert result.shape == (32, 32)
    assert result.dtype == expected_dtype
