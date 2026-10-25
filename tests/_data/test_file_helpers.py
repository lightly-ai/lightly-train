#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import warnings
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


def test_open_yolo_oriented_object_detection_label_numpy(tmp_path: Path) -> None:
    with open(tmp_path / "label.txt", "w") as f:
        f.write(
            "2 0.1 0.1 0.2 0.1 0.1 0.2 0.2 0.2\n1 0.3 0.3 0.4 0.3 0.3 0.4 0.4 0.4\n"
        )
    bboxes, class_labels = file_helpers.open_yolo_oriented_object_detection_label_numpy(
        label_path=tmp_path / "label.txt"
    )
    np.testing.assert_array_almost_equal(
        bboxes,
        np.array(
            [
                [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 0.4, 0.4],
            ]
        ),
    )
    np.testing.assert_array_equal(class_labels, np.array([2, 1]))


def test_open_yolo_oriented_object_detection_label_numpy__empty(
    tmp_path: Path,
) -> None:
    with open(tmp_path / "label.txt", "w") as f:
        f.write("")
    bboxes, class_labels = file_helpers.open_yolo_oriented_object_detection_label_numpy(
        label_path=tmp_path / "label.txt"
    )
    assert bboxes.shape == (0, 8)
    assert class_labels.shape == (0,)


def test_open_yolo_object_detection_label(tmp_path: Path) -> None:
    with open(tmp_path / "label.txt", "w") as f:
        f.write("2 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.4 0.4\n")
    bboxes, class_labels = file_helpers.open_yolo_object_detection_label(
        label_path=tmp_path / "label.txt"
    )
    assert bboxes == [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]
    assert class_labels == [2, 1]


def test_open_yolo_object_detection_label__empty(
    tmp_path: Path,
) -> None:
    with open(tmp_path / "label.txt", "w") as f:
        f.write("")
    bboxes, class_labels = file_helpers.open_yolo_object_detection_label(
        label_path=tmp_path / "label.txt"
    )
    assert bboxes == []
    assert class_labels == []


def test_open_yolo_instance_segmentation_label(tmp_path: Path) -> None:
    with open(tmp_path / "label.txt", "w") as f:
        f.write("2 0.1 0.1 0.1 0.2 0.2 0.2\n1 0.3 0.3 0.3 0.4 0.4 0.4\n")
    polygons, bboxes, class_labels = file_helpers.open_yolo_instance_segmentation_label(
        label_path=tmp_path / "label.txt"
    )
    assert len(polygons) == 2
    assert polygons[0] == [[0.1, 0.1, 0.1, 0.2, 0.2, 0.2]]
    assert polygons[1] == [[0.3, 0.3, 0.3, 0.4, 0.4, 0.4]]
    np.testing.assert_array_almost_equal(
        bboxes, np.array([[0.15, 0.15, 0.1, 0.1], [0.35, 0.35, 0.1, 0.1]])
    )
    np.testing.assert_array_equal(class_labels, np.array([2, 1]))


def test_open_yolo_instance_segmentation_label__empty(
    tmp_path: Path,
) -> None:
    with open(tmp_path / "label.txt", "w") as f:
        f.write("")
    polygons, bboxes, class_labels = file_helpers.open_yolo_instance_segmentation_label(
        label_path=tmp_path / "label.txt"
    )
    assert polygons == []
    assert bboxes == []
    assert class_labels == []


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


@pytest.mark.parametrize("data_format", ["ct", "mr", "overlay"])
def test__open_image_numpy__with_pydicom__value_range(data_format: str) -> None:
    pydicom_examples = pytest.importorskip(
        "pydicom.examples",
        reason="pydicom examples not supported",
    )
    image_path: Path = pydicom_examples.get_path(data_format)

    result = file_helpers._open_image_numpy__with_pydicom(image_src=image_path)

    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    # The image must keep its contrast instead of collapsing to a narrow band.
    assert result.std() > 0.05


DX_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.1.1"


@pytest.mark.parametrize("with_window", [False, True])
def test__open_image_numpy__with_pydicom__ct_hounsfield_units(
    tmp_path: Path, with_window: bool
) -> None:
    pydicom_examples = pytest.importorskip(
        "pydicom.examples",
        reason="pydicom examples not supported",
    )
    ds = pydicom.dcmread(pydicom_examples.get_path("ct"))
    if with_window:
        # The file window of a CT image is ignored.
        ds.WindowCenter = 40
        ds.WindowWidth = 400
    image_path = tmp_path / "ct.dcm"
    ds.save_as(image_path)
    hu = ds.pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    result = file_helpers._open_image_numpy__with_pydicom(image_src=image_path)

    expected = np.clip((hu + 1024.0) / 4095.0, 0.0, 1.0)
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("case", ["no_rescale", "rescale_type_not_hu"])
def test__open_image_numpy__with_pydicom__ct_without_hounsfield_units(
    tmp_path: Path, case: str
) -> None:
    pydicom_examples = pytest.importorskip(
        "pydicom.examples",
        reason="pydicom examples not supported",
    )
    ds = pydicom.dcmread(pydicom_examples.get_path("ct"))
    if case == "no_rescale":
        del ds.RescaleSlope
        del ds.RescaleIntercept
        values = ds.pixel_array.astype(np.float64)
    else:
        ds.RescaleType = "US"
        values = ds.pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    image_path = tmp_path / "ct.dcm"
    ds.save_as(image_path)

    with pytest.warns(UserWarning, match="CT images without Hounsfield units"):
        result = file_helpers._open_image_numpy__with_pydicom(image_src=image_path)

    # Values that are not Hounsfield units fall back to per-image percentiles.
    low, high = np.percentile(values, [0.5, 99.5])
    expected = np.clip((values - low) / (high - low), 0.0, 1.0)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test__open_image_numpy__with_pydicom__xray_window(tmp_path: Path) -> None:
    pydicom_examples = pytest.importorskip(
        "pydicom.examples",
        reason="pydicom examples not supported",
    )
    ds = pydicom.dcmread(pydicom_examples.get_path("mr"))
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.UID(DX_SOP_CLASS_UID)
    image_path = tmp_path / "dx.dcm"
    ds.save_as(image_path)
    values = ds.pixel_array.astype(np.float64)
    center, width = float(ds.WindowCenter), float(ds.WindowWidth)

    result = file_helpers._open_image_numpy__with_pydicom(image_src=image_path)

    # LINEAR window function of DICOM PS3.3 C.11.2.1.2.1.
    expected = np.clip((values - (center - 0.5)) / (width - 1) + 0.5, 0.0, 1.0)
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("sop_class_uid", [None, DX_SOP_CLASS_UID])
def test__open_image_numpy__with_pydicom__percentiles(
    tmp_path: Path, sop_class_uid: str | None
) -> None:
    pydicom_examples = pytest.importorskip(
        "pydicom.examples",
        reason="pydicom examples not supported",
    )
    ds = pydicom.dcmread(pydicom_examples.get_path("mr"))
    if sop_class_uid is None:
        # An MR image uses percentiles, also when it has a window.
        assert "WindowCenter" in ds
    else:
        # An X-ray image without a window or VOI LUT uses percentiles.
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.UID(sop_class_uid)
        del ds.WindowCenter
        del ds.WindowWidth
    image_path = tmp_path / "image.dcm"
    ds.save_as(image_path)
    values = ds.pixel_array.astype(np.float64)
    low, high = np.percentile(values, (0.5, 99.5))

    result = file_helpers._open_image_numpy__with_pydicom(image_src=image_path)

    expected = np.clip((values - low) / (high - low), 0.0, 1.0)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test__open_image_numpy__with_pydicom__percentiles_sparse(tmp_path: Path) -> None:
    pydicom_examples = pytest.importorskip(
        "pydicom.examples",
        reason="pydicom examples not supported",
    )
    ds = pydicom.dcmread(pydicom_examples.get_path("mr"))
    pixels = np.zeros_like(ds.pixel_array)
    pixels[:2, :10] = 500  # 20 of 4096 pixels, so both percentiles are 0.
    ds.PixelData = pixels.tobytes()
    image_path = tmp_path / "mr.dcm"
    ds.save_as(image_path)

    result = file_helpers._open_image_numpy__with_pydicom(image_src=image_path)

    np.testing.assert_allclose(result, pixels / 500.0, atol=1e-6)


def test__open_image_numpy__with_pydicom__enhanced_ct(tmp_path: Path) -> None:
    pydicom_examples = pytest.importorskip(
        "pydicom.examples",
        reason="pydicom examples not supported",
    )
    ds = pydicom.dcmread(pydicom_examples.get_path("ct"))
    # A single-frame Enhanced CT with the rescale only in the functional groups.
    transform = pydicom.Dataset()
    transform.RescaleSlope = ds.RescaleSlope
    transform.RescaleIntercept = ds.RescaleIntercept
    transform.RescaleType = "HU"
    group = pydicom.Dataset()
    group.PixelValueTransformationSequence = pydicom.Sequence([transform])
    ds.SharedFunctionalGroupsSequence = pydicom.Sequence([group])
    del ds.RescaleSlope
    del ds.RescaleIntercept
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.UID(
        "1.2.840.10008.5.1.4.1.1.2.1"
    )
    image_path = tmp_path / "enhanced_ct.dcm"
    ds.save_as(image_path)
    values = ds.pixel_array.astype(np.float64)
    low, high = np.percentile(values, (0.5, 99.5))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = file_helpers._open_image_numpy__with_pydicom(image_src=image_path)

    # The loader does not read functional groups, so it uses percentiles silently.
    expected = np.clip((values - low) / (high - low), 0.0, 1.0)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test__open_image_numpy__with_pydicom__monochrome1(tmp_path: Path) -> None:
    pydicom_examples = pytest.importorskip(
        "pydicom.examples",
        reason="pydicom examples not supported",
    )
    monochrome2_path = pydicom_examples.get_path("mr")
    ds = pydicom.dcmread(monochrome2_path)
    ds.PhotometricInterpretation = "MONOCHROME1"
    monochrome1_path = tmp_path / "mr.dcm"
    ds.save_as(monochrome1_path)

    monochrome1 = file_helpers._open_image_numpy__with_pydicom(
        image_src=monochrome1_path
    )
    monochrome2 = file_helpers._open_image_numpy__with_pydicom(
        image_src=monochrome2_path
    )

    np.testing.assert_allclose(monochrome1, 1.0 - monochrome2, atol=1e-6)
