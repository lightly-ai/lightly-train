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
    ("extension", "expected_backend", "dtype", "num_channels", "pil_mode"),
    [
        (".jpg", "torch", np.uint8, 3, "RGB"),
        (".jpeg", "torch", np.uint8, 3, "RGB"),
        (".png", "torch", np.uint8, 3, "RGB"),
        (".png", "torch", np.uint8, 4, "RGBA"),
        (".bmp", "pil", np.uint8, 3, "RGB"),
        (".gif", "pil", np.uint8, 3, "RGB"),
        (".webp", "pil", np.uint8, 4, "RGBA"),
        (".tiff", "pil", np.int32, 0, "I"),
        (".tiff", "pil", np.uint16, 0, "I;16"),
        (".tiff", "pil", np.float32, 0, "F"),
    ],
)
def test_open_image_numpy(
    tmp_path: Path,
    extension: str,
    expected_backend: str,
    dtype: DTypeLike,
    num_channels: int,
    pil_mode: str,
    mocker: MockerFixture,
) -> None:
    image_path = tmp_path / f"image{extension}"

    max_value = int(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1
    helpers.create_image(
        path=image_path,
        height=32,
        width=32,
        mode=pil_mode,
        dtype=dtype,
        max_value=max_value,
        num_channels=num_channels,
    )

    open_mode = (
        ImageMode.RGB
        if pil_mode == "RGB" and dtype == np.uint8
        else ImageMode.UNCHANGED
    )
    torch_spy = mocker.spy(file_helpers, "_open_image_numpy__with_torch")
    pil_spy = mocker.spy(file_helpers, "_open_image_numpy__with_pil")
    result = file_helpers.open_image_numpy(image_path=image_path, mode=open_mode)
    assert isinstance(result, np.ndarray)

    expected_shape = (32, 32) if num_channels == 0 else (32, 32, num_channels)
    expected_dtype = np.uint8 if dtype == np.uint8 else np.float32
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype

    if expected_backend == "torch":
        torch_spy.assert_called_once()
        pil_spy.assert_not_called()
    else:
        torch_spy.assert_not_called()
        pil_spy.assert_called_once()


@pytest.mark.parametrize(
    ("extension", "expected_backend", "dtype", "num_channels", "pil_mode"),
    [
        (".png", "torch", np.uint8, 0, "L"),
        (".png", "torch", np.uint8, 3, "RGB"),
        (".png", "torch", np.uint16, 0, "I;16"),
        (".bmp", "pil", np.uint8, 0, "L"),
    ],
)
def test_open_mask_numpy(
    tmp_path: Path,
    extension: str,
    expected_backend: str,
    dtype: DTypeLike,
    num_channels: int,
    pil_mode: str,
    mocker: MockerFixture,
) -> None:
    mask_path = tmp_path / f"mask{extension}"

    max_value = int(np.iinfo(dtype).max)
    helpers.create_image(
        path=mask_path,
        height=32,
        width=32,
        mode=pil_mode,
        dtype=dtype,
        max_value=max_value,
        num_channels=num_channels,
    )

    torch_spy = mocker.spy(file_helpers, "_open_mask_numpy__with_torch")
    pil_spy = mocker.spy(file_helpers, "_open_mask_numpy__with_pil")
    result = file_helpers.open_mask_numpy(mask_path=mask_path)
    assert isinstance(result, np.ndarray)

    expected_shape = (32, 32) if num_channels in (0, 1) else (32, 32, num_channels)
    assert result.shape == expected_shape
    assert result.dtype == dtype

    if expected_backend == "torch":
        torch_spy.assert_called_once()
        pil_spy.assert_not_called()
    else:
        torch_spy.assert_not_called()
        pil_spy.assert_called_once()


def test_open_yolo_object_detection_label_numpy(tmp_path: Path) -> None:
    with open(tmp_path / "label.txt", "w") as f:
        f.write("2 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.4 0.4\n")
    bboxes, class_labels = file_helpers.open_yolo_object_detection_label_numpy(
        label_path=tmp_path / "label.txt"
    )
    np.testing.assert_array_almost_equal(
        bboxes, np.array([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]])
    )
    np.testing.assert_array_equal(class_labels, np.array([2, 1]))


def test_open_yolo_object_detection_label_numpy__empty(
    tmp_path: Path,
) -> None:
    with open(tmp_path / "label.txt", "w") as f:
        f.write("")
    bboxes, class_labels = file_helpers.open_yolo_object_detection_label_numpy(
        label_path=tmp_path / "label.txt"
    )
    assert bboxes.shape == (0, 4)
    assert class_labels.shape == (0,)


def test_open_yolo_instance_segmentation_label_numpy(tmp_path: Path) -> None:
    with open(tmp_path / "label.txt", "w") as f:
        f.write("2 0.1 0.1 0.1 0.2 0.2 0.2\n1 0.3 0.3 0.3 0.4 0.4 0.4\n")
    polygons, bboxes, class_labels = (
        file_helpers.open_yolo_instance_segmentation_label_numpy(
            label_path=tmp_path / "label.txt"
        )
    )
    assert len(polygons) == 2
    np.testing.assert_array_almost_equal(
        polygons[0], np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    )
    np.testing.assert_array_almost_equal(
        polygons[1], np.array([0.3, 0.3, 0.3, 0.4, 0.4, 0.4])
    )
    np.testing.assert_array_almost_equal(
        bboxes, np.array([[0.15, 0.15, 0.1, 0.1], [0.35, 0.35, 0.1, 0.1]])
    )
    np.testing.assert_array_equal(class_labels, np.array([2, 1]))


def test_open_yolo_instance_segmentation_label_numpy__empty(
    tmp_path: Path,
) -> None:
    with open(tmp_path / "label.txt", "w") as f:
        f.write("")
    polygons, bboxes, class_labels = (
        file_helpers.open_yolo_instance_segmentation_label_numpy(
            label_path=tmp_path / "label.txt"
        )
    )
    assert len(polygons) == 0
    assert bboxes.shape == (0, 4)
    assert class_labels.shape == (0,)
