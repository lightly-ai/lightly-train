#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision import io
from torchvision.io import ImageReadMode
from torchvision.transforms.v2 import functional as F

from lightly_train.types import (
    ImageFilename,
    NDArrayBBoxes,
    NDArrayClasses,
    NDArrayImage,
    PathLike,
)


def list_image_files(imgs_and_dirs: Sequence[Path]) -> Iterable[Path]:
    """List image files recursively from the given list of image files and directories.

    Args:
        imgs_and_dirs: A list of (relative or absolute) paths to image files and
            directories that should be scanned for images.

    Returns:
        A list of absolute paths pointing to the image files.
    """
    for img_or_dir in imgs_and_dirs:
        if img_or_dir.is_file() and (
            img_or_dir.suffix.lower() in _pil_supported_image_extensions()
        ):
            yield img_or_dir.resolve()
        elif img_or_dir.is_dir():
            yield from _get_image_filepaths(img_or_dir)
        else:
            raise ValueError(f"Invalid path: {img_or_dir}")


def list_image_filenames(
    *, image_dir: Path | None = None, files: Iterable[Path] | None = None
) -> Iterable[ImageFilename]:
    """List image filenames relative to `image_dir` recursively.

    Args:
        image_dir:
            The root directory to scan for images.

    Returns:
        An iterable of image filenames relative to `image_dir` or absolute paths
        if `files` is provided.
    """
    if (image_dir is not None and files is not None) or (
        image_dir is None and files is None
    ):
        raise ValueError(
            "Either `image_dir` or `files` must be provided, but not both."
        )
    elif files is not None:
        # NOTE(Jonas 06/2025): drop resolve if complains about performance are raised.
        return (ImageFilename(str(fpath.resolve())) for fpath in files)
    elif image_dir is not None:
        return (
            ImageFilename(str(fpath.relative_to(image_dir)))
            for fpath in _get_image_filepaths(image_dir=image_dir)
        )
    else:
        raise ValueError("Either `image_dir` or `files` must be provided.")


def _pil_supported_image_extensions() -> set[str]:
    return {
        ex
        for ex, format in Image.registered_extensions().items()
        if format in Image.OPEN
    }


def _get_image_filepaths(image_dir: Path) -> Iterable[Path]:
    extensions = _pil_supported_image_extensions()
    for root, _, files in os.walk(image_dir, followlinks=True):
        root_path = Path(root)
        for file in files:
            fpath = root_path / file
            if fpath.suffix.lower() in extensions:
                yield fpath


def _torchvision_supported_image_extensions() -> set[str]:
    # See https://pytorch.org/vision/0.18/generated/torchvision.io.read_image.html
    return {"jpg", "jpeg", "png"}


def as_image_tensor(image: PathLike | PILImage | Tensor) -> Tensor:
    """Returns image as (C, H, W) tensor."""
    if isinstance(image, Tensor):
        return image
    elif isinstance(image, PILImage):
        image_tensor: Tensor = F.pil_to_tensor(image)
        return image_tensor
    else:
        return open_image_tensor(Path(image))


def open_image_tensor(image_path: Path) -> Tensor:
    """Returns image as (C, H, W) tensor."""
    image: Tensor
    if image_path.suffix.lower() in _torchvision_supported_image_extensions():
        image = io.read_image(str(image_path), mode=ImageReadMode.RGB)
        return image
    else:
        image = F.pil_to_tensor(Image.open(image_path).convert("RGB"))
        return image


def open_image_numpy(
    image_path: Path,
    mode: Literal["RGB", "L", "UNCHANGED", "MASK"] = "RGB",
) -> NDArrayImage:
    """Returns image as (H, W, C) numpy array."""
    image_np: NDArray[np.uint8]
    if image_path.suffix.lower() in _torchvision_supported_image_extensions():
        mode_torch = {
            "RGB": ImageReadMode.RGB,
            "L": ImageReadMode.GRAY,
            "UNCHANGED": ImageReadMode.UNCHANGED,
            "MASK": ImageReadMode.GRAY,
        }[mode]
        image_torch = io.read_image(str(image_path), mode=mode_torch)
        image_torch = image_torch.permute(1, 2, 0)
        image_np = image_torch.numpy()
    else:
        convert_mode = {
            "RGB": "RGB",
            "L": "L",
            "UNCHANGED": None,
            "MASK": None,
        }[mode]
        image = Image.open(image_path)
        if convert_mode is not None:
            image = image.convert(convert_mode)
        image_np = np.array(image)
    dtype = image_np.dtype
    if np.issubdtype(dtype, np.unsignedinteger) and dtype != np.uint8:
        # Convert uint16, uint32, uint64 to signed integer type because torch has only
        # limited support for these types.
        dtype_str = str(dtype)  # Str in case dtype is not supported on platform.
        target_dtype = {
            "uint16": np.int32,
            "uint32": np.int64,
            "uint64": np.int64,  # int128 is not supported by numpy and torch.
        }[dtype_str]
        image_np = image_np.astype(target_dtype)
    return image_np


def open_yolo_label_numpy(label_path: Path) -> tuple[NDArrayBBoxes, NDArrayClasses]:
    """Open a YOLO label file and return the bounding boxes and classes as numpy arrays."""
    bboxes = []
    classes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            # Skip empty lines.
            if not line:
                continue
            class_id, x_center, y_center, width, height = (
                float(x) for x in line.split()
            )
            bboxes.append([x_center, y_center, width, height])
            classes.append(int(class_id))
    return np.array(bboxes, dtype=np.float64), np.array(classes, dtype=np.int64)
