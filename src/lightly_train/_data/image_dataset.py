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
import torchvision
from numpy.typing import NDArray
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode

from lightly_train._env import Env
from lightly_train.types import (
    DatasetItem,
    ImageFilename,
    NDArrayImage,
    PathLike,
    Transform,
    TransformInput,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset[DatasetItem]):
    def __init__(
        self,
        image_dir: Path | None,
        image_filenames: Sequence[ImageFilename],
        transform: Transform,
        mask_dir: Path | None = None,
    ):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.mask_dir = mask_dir
        self.transform = transform

        image_mode = Env.LIGHTLY_TRAIN_IMAGE_MODE.value
        if image_mode not in ("RGB", "UNCHANGED"):
            raise ValueError(
                f'Invalid image mode: {Env.LIGHTLY_TRAIN_IMAGE_MODE.name}="{image_mode}". '
                "Supported modes are 'RGB' and 'UNCHANGED'."
            )
        self.image_mode: Literal["RGB", "UNCHANGED"] = image_mode  # type: ignore[assignment]

    def __getitem__(self, idx: int) -> DatasetItem:
        filename = self.image_filenames[idx]
        if self.image_dir is None:
            image = _open_image(Path(filename), mode=self.image_mode)
        else:
            image = _open_image(self.image_dir / filename, mode=self.image_mode)

        input: TransformInput = {"image": image}

        if self.mask_dir:
            maskname = Path(filename).with_suffix(".png")
            mask = _open_image(self.mask_dir / maskname, mode="L")
            input["mask"] = mask

        transformed = self.transform(input)

        dataset_item: DatasetItem = {
            "filename": filename,
            "views": [view["image"] for view in transformed],
        }
        if self.mask_dir:
            dataset_item["masks"] = [view["mask"] for view in transformed]
        return dataset_item

    def __len__(self) -> int:
        return len(self.image_filenames)


def list_image_filenames(
    *, image_dir: Path | None = None, files: Iterable[PathLike] | None = None
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
        return (ImageFilename(fpath) for fpath in files)
    elif image_dir is not None:
        image_dir_str = str(image_dir)
        return (
            ImageFilename(os.path.relpath(fpath, start=image_dir_str))
            for fpath in _get_image_filepaths(image_dir=image_dir)
        )
    else:
        raise ValueError("Either `image_dir` or `files` must be provided.")


def list_image_files(imgs_and_dirs: Sequence[Path]) -> Iterable[PathLike]:
    """List image files recursively from the given list of image files and directories.

    Args:
        imgs_and_dirs: A list of (relative or absolute) paths to image files and
            directories that should be scanned for images.

    Returns:
        A list of absolute paths pointing to the image files.
    """
    extensions = _pil_supported_image_extensions()
    for img_or_dir in imgs_and_dirs:
        if img_or_dir.is_file() and (
            img_or_dir.suffix.lower() in extensions
        ):
            yield img_or_dir.resolve()
        elif img_or_dir.is_dir():
            yield from _get_image_filepaths(img_or_dir)
        else:
            raise ValueError(f"Invalid path: {img_or_dir}")


def _get_image_filepaths(image_dir: Path) -> Iterable[str]:
    # This function is inspired by os.path.walk: https://github.com/python/cpython/blob/008c8cafed3498b0e91766c5d94595f717f92425/Lib/os.py#L289
    extensions = _pil_supported_image_extensions()
    stack = [str(image_dir)]

    while stack:
        current = stack.pop()
    
        try:
            scandir_it = os.scandir(current)
        except OSError:
            continue

        with scandir_it:
            while True:
                try:
                    entry = next(scandir_it)
                except (OSError, StopIteration):
                    break

                try:
                    is_dir = entry.is_dir(follow_symlinks=True)
                except OSError:
                    is_dir = False

                if is_dir:
                    stack.append(entry.path)
                    continue

                try:
                    is_file = entry.is_file()
                except OSError:
                    is_file = False

                if is_file:
                    try:
                        file_extension = os.path.splitext(entry.name)[1].lower()
                    except (OSError, IndexError):
                        file_extension = ""

                    if file_extension in extensions:
                        yield entry.path


def _pil_supported_image_extensions() -> set[str]:
    return {
        ex
        for ex, format in Image.registered_extensions().items()
        if format in Image.OPEN
    }


def _torchvision_supported_image_extensions() -> set[str]:
    # See https://pytorch.org/vision/0.18/generated/torchvision.io.read_image.html
    return {"jpg", "jpeg", "png"}


def _open_image(
    image_path: Path, mode: Literal["RGB", "L", "UNCHANGED"] = "RGB"
) -> NDArrayImage:
    image_np: NDArray[np.uint8]
    if image_path.suffix.lower() in _torchvision_supported_image_extensions():
        mode_torch = {
            "RGB": ImageReadMode.RGB,
            "L": ImageReadMode.GRAY,
            "UNCHANGED": ImageReadMode.UNCHANGED,
        }[mode]
        image_torch = torchvision.io.read_image(str(image_path), mode=mode_torch)
        image_torch = image_torch.permute(1, 2, 0)
        image_np = image_torch.numpy()
    else:
        image = Image.open(image_path)
        if mode != "UNCHANGED":
            image = image.convert(mode)
        image_np = np.array(image)
    return image_np
