#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from torchvision import io
from torchvision.io import ImageReadMode

from lightly_train.types import ImageFilename, NDArrayImage


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


def open_image(
    image_path: Path, mode: Literal["RGB", "L", "UNCHANGED"] = "RGB"
) -> NDArrayImage:
    image_np: NDArray[np.uint8]
    if image_path.suffix.lower() in _torchvision_supported_image_extensions():
        mode_torch = {
            "RGB": ImageReadMode.RGB,
            "L": ImageReadMode.GRAY,
            "UNCHANGED": ImageReadMode.UNCHANGED,
        }[mode]
        image_torch = io.read_image(str(image_path), mode=mode_torch)
        image_torch = image_torch.permute(1, 2, 0)
        image_np = image_torch.numpy()
    else:
        image = Image.open(image_path)
        if mode != "UNCHANGED":
            image = image.convert(mode)
        image_np = np.array(image)
    return image_np
