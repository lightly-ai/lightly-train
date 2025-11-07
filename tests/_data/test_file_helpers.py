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
import torch
from numpy.typing import DTypeLike
from pytest import LogCaptureFixture, MonkeyPatch
from pytest_mock import MockerFixture
from torch import Tensor
from torchvision.transforms.v2 import functional as F

from lightly_train._data import file_helpers
from lightly_train._data.file_helpers import (
    TORCHVISION_GEQ_0_20_0,
    ImageMode,
)

from .. import helpers

# Convert uint16, uint32, uint64 to signed integer type because torch <2.3.0 has only
# limited support for these types.
numpy_to_torch_dtype_dict = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.uint16: torch.int32,
    np.uint32: torch.int64,
    np.uint64: torch.int64,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}


try:
    import pydicom
except ImportError:
    pydicom = None  # type: ignore[assignment]


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
    if extension == ".dcm":
        pytest.skip("Skipping DICOM in this test.")
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
    if extension == ".dcm":
        pytest.skip("Skipping DICOM in this test.")
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
        (".gif", "pil", np.uint8, 0, "P"),
        (".webp", "pil", np.uint8, 4, "RGBA"),
        (".tiff", "pil", np.int32, 0, "I"),
        (".tiff", "pil", np.float32, 0, "F"),
        (".dcm", "pydicom", np.float32, 0, ""),
    ],
)
def test_open_image_tensor(
    tmp_path: Path,
    extension: str,
    expected_backend: str,
    dtype: DTypeLike,
    num_channels: int,
    pil_mode: str,
    mocker: MockerFixture,
) -> None:
    if extension == ".dcm":
        pydicom_examples = pytest.importorskip(
            "pydicom.examples",
            reason="pydicom examples not supported",
        )
        image_path: Path = pydicom_examples.get_path("mr")
    else:
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

    torch_spy = mocker.spy(file_helpers, "load_image")
    pil_spy = mocker.spy(F, "pil_to_tensor")
    pydicom_spy = mocker.spy(file_helpers, "_open_image_numpy__with_pydicom")

    result = file_helpers.open_image_tensor(image_path=image_path)
    assert isinstance(result, Tensor)

    expected_shape = (1, 32, 32) if num_channels == 0 else (num_channels, 32, 32)
    expected_dtype = (
        torch.float32 if dtype == np.float32 else numpy_to_torch_dtype_dict[dtype]  # type: ignore[index]
    )
    if extension != ".dcm":
        assert result.shape == expected_shape
    else:
        assert result.shape == (1, 64, 64)
    assert result.dtype == expected_dtype

    if expected_dtype == torch.float32:
        assert result.min() >= 0.0 and result.max() <= 1.0

    if expected_backend == "torch":
        torch_spy.assert_called_once()
        pil_spy.assert_not_called()
        pydicom_spy.assert_not_called()
    elif expected_backend == "pil":
        torch_spy.assert_not_called()
        pil_spy.assert_called_once()
        pydicom_spy.assert_not_called()
    else:
        torch_spy.assert_not_called()
        pil_spy.assert_not_called()
        pydicom_spy.assert_called_once()


@pytest.mark.parametrize(
    ("extension", "expected_backend", "dtype", "num_channels", "pil_mode"),
    [
        (".jpg", "torch", np.uint8, 3, "RGB"),
        (".jpeg", "torch", np.uint8, 3, "RGB"),
        (".png", "torch", np.uint8, 3, "RGB"),
        (".png", "torch", np.uint8, 4, "RGBA"),
        (".bmp", "pil", np.uint8, 3, "RGB"),
        (".gif", "pil", np.uint8, 0, "P"),
        (".webp", "pil", np.uint8, 4, "RGBA"),
        (".tiff", "pil", np.int32, 0, "I"),
        (".tiff", "pil", np.uint16, 0, "I;16"),
        (".tiff", "pil", np.float32, 0, "F"),
        (".dcm", "pydicom", np.float32, 0, ""),
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
    if extension == ".dcm":
        pydicom_examples = pytest.importorskip(
            "pydicom.examples",
            reason="pydicom examples not supported",
        )
        image_path: Path = pydicom_examples.get_path("mr")
    else:
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
    pydicom_spy = mocker.spy(file_helpers, "_open_image_numpy__with_pydicom")

    result = file_helpers.open_image_numpy(image_path=image_path, mode=open_mode)
    assert isinstance(result, np.ndarray)

    expected_shape = (32, 32) if num_channels == 0 else (32, 32, num_channels)
    expected_dtype = np.uint8 if dtype == np.uint8 else np.float32
    if extension != ".dcm":
        assert result.shape == expected_shape
    else:
        assert result.shape == (64, 64)  # Shape of the example DICOM file
    assert result.dtype == expected_dtype

    if expected_dtype == np.float32:
        assert result.min() >= 0.0 and result.max() <= 1.0

    if expected_backend == "torch":
        torch_spy.assert_called_once()
        pil_spy.assert_not_called()
        pydicom_spy.assert_not_called()
    elif expected_backend == "pil":
        torch_spy.assert_not_called()
        pil_spy.assert_called_once()
        pydicom_spy.assert_not_called()
    else:
        torch_spy.assert_not_called()
        pil_spy.assert_not_called()
        pydicom_spy.assert_called_once()


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
    if (
        (not TORCHVISION_GEQ_0_20_0)
        and expected_backend == "torch"
        and dtype == np.uint16
    ):
        pytest.skip(
            "torchvision<0.20.0 does not support uint16 masks with torchvision."
        )

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


@pytest.mark.parametrize(
    ("data_format", "expected_shape", "expected_dtype"),
    [
        ("mr", (64, 64), np.float32),
        ("ct", (128, 128), np.float32),
        ("overlay", (300, 484), np.float32),
        (
            "rgb_color",
            (240, 320, 3),
            np.uint8,
        ),  # skipped test for palette_color as it behaves differently in different python versions
        ("jpeg2k", (480, 640, 3), np.uint8),
    ],
)
def test__open_image_numpy__with_pydicom(
    data_format: str,
    expected_shape: tuple[int, ...],
    expected_dtype: DTypeLike,
) -> None:
    pydicom_examples = pytest.importorskip(
        "pydicom.examples",
        reason="pydicom examples not supported",
    )
    image_path: Path = pydicom_examples.get_path(data_format)

    result = file_helpers._open_image_numpy__with_pydicom(image_src=image_path)

    assert isinstance(result, np.ndarray)
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype
