#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Set
from enum import Enum
from pathlib import Path

import numpy as np
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

logger = logging.getLogger(__name__)


class ImageMode(Enum):
    RGB = "RGB"
    UNCHANGED = "UNCHANGED"
    MASK = "MASK"


def list_image_filenames_from_iterable(
    imgs_and_dirs: Iterable[PathLike],
) -> Iterable[ImageFilename]:
    """List image files recursively from the given list of image files and directories.

    Assumes that all given paths exist.

    Args:
        imgs_and_dirs: A list of (relative or absolute) paths to image files and
            directories that should be scanned for images.

    Returns:
        An iterable of image filenames starting from the given paths. The given paths
        are always included in the output filenames.
    """
    supported_extensions = _pil_supported_image_extensions()
    for img_or_dir in imgs_and_dirs:
        _, ext = os.path.splitext(img_or_dir)
        # Only check image extension. This is faster than checking isfile() because it
        # does not require a system call.
        if ext.lower() in supported_extensions:
            yield ImageFilename(img_or_dir)
        # For dirs we have to make a system call.
        elif os.path.isdir(img_or_dir):
            contains_images = False
            dir_str = str(img_or_dir)
            for image_filename in _get_image_filenames(
                image_dir=dir_str, image_extensions=supported_extensions
            ):
                contains_images = True
                yield ImageFilename(os.path.join(dir_str, image_filename))
            if not contains_images:
                logger.warning(
                    f"The directory '{img_or_dir}' does not contain any images."
                )
        else:
            raise ValueError(
                f"Invalid path: '{img_or_dir}'. It is neither a valid image nor a "
                f"directory. Valid image extensions are: {supported_extensions}"
            )


def list_image_filenames_from_dir(image_dir: PathLike) -> Iterable[ImageFilename]:
    """List image filenames relative to `image_dir` recursively.

    Args:
        image_dir:
            The root directory to scan for images.

    Returns:
        An iterable of image filenames relative to `image_dir`.
    """
    for filename in _get_image_filenames(image_dir=image_dir):
        yield ImageFilename(filename)


def _pil_supported_image_extensions() -> set[str]:
    return {
        ex
        for ex, format in Image.registered_extensions().items()
        if format in Image.OPEN
    }


def _get_image_filenames(
    image_dir: PathLike, image_extensions: Set[str] | None = None
) -> Iterable[str]:
    """Returns image filenames relative to image_dir."""
    image_extensions = (
        _pil_supported_image_extensions()
        if image_extensions is None
        else image_extensions
    )
    for dirpath, _, filenames in os.walk(image_dir, followlinks=True):
        # Make paths relative to image_dir. `dirpath` is absolute.
        parent = os.path.relpath(dirpath, start=image_dir)
        parent = "" if parent == "." else parent
        for file in filenames:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                yield os.path.join(parent, file)


_TORCHVISION_SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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
    if image_path.suffix.lower() in _TORCHVISION_SUPPORTED_IMAGE_EXTENSIONS:
        image = io.read_image(str(image_path), mode=ImageReadMode.RGB)
        return image
    else:
        image = F.pil_to_tensor(Image.open(image_path).convert("RGB"))
        return image


def open_image_numpy(
    image_path: Path,
    mode: ImageMode = ImageMode.RGB,
) -> NDArrayImage:
    """Returns image as (H, W, C) or (H, W) numpy array."""
    image_np: NDArrayImage
    if image_path.suffix.lower() in _TORCHVISION_SUPPORTED_IMAGE_EXTENSIONS:
        try:
            image_np = _open_image_numpy__with_torch(image_path=image_path, mode=mode)
        except RuntimeError:
            # RuntimeError can happen for truncated images. Fall back to PIL.
            image_np = _open_image_numpy__with_pil(image_path=image_path, mode=mode)
    else:
        image_np = _open_image_numpy__with_pil(image_path=image_path, mode=mode)
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


def _open_image_numpy__with_torch(
    image_path: Path,
    mode: ImageMode = ImageMode.RGB,
) -> NDArrayImage:
    image_np: NDArrayImage
    mode_torch = {
        ImageMode.RGB: ImageReadMode.RGB,
        ImageMode.UNCHANGED: ImageReadMode.UNCHANGED,
        ImageMode.MASK: ImageReadMode.UNCHANGED,
    }[mode]
    image_torch = io.read_image(str(image_path), mode=mode_torch)
    image_torch = image_torch.permute(1, 2, 0)
    if image_torch.shape[2] == 1 and mode == ImageMode.RGB:
        # Convert single-channel grayscale to 3-channel RGB.
        # (H, W, 1) -> (H, W, 3)
        image_torch = image_torch.repeat(1, 1, 3)
    if image_torch.shape[2] == 1 and mode == ImageMode.MASK:
        # Squeeze channel dimension for single-channel masks.
        # (H, W, 1) -> (H, W)
        image_torch = image_torch.squeeze(2)
    image_np = image_torch.numpy()
    return image_np


def _open_image_numpy__with_pil(
    image_path: Path,
    mode: ImageMode = ImageMode.RGB,
) -> NDArrayImage:
    image_np: NDArrayImage
    convert_mode = {
        ImageMode.RGB: "RGB",
        ImageMode.UNCHANGED: None,
        ImageMode.MASK: None,
    }[mode]
    image = Image.open(image_path)
    if convert_mode is not None:
        image = image.convert(convert_mode)
    image_np = np.array(image)
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
