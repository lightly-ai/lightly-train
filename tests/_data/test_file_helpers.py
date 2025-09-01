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
from pytest_mock import MockerFixture

from lightly_train._data import file_helpers
from lightly_train._data.file_helpers import ImageMode

from .. import helpers


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
