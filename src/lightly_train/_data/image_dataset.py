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

from lightly_train.types import (
    DatasetItem,
    ImageFilename,
    NDArrayImage,
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

    def __getitem__(self, idx: int) -> DatasetItem:
        filename = self.image_filenames[idx]
        if self.image_dir is None:
            image = _open_image(Path(filename))
        else:
            image = _open_image(self.image_dir / filename)

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
    *, image_dir: Path | None = None, files: Iterable[Path] | None = None
) -> Iterable[ImageFilename]:
    """List image filenames relative to `image_dir` recursively.

    Args:
        image_dir:
            The root directory to scan for images.

    Returns:
        An iterable of image filenames relative to `image_dir`.
    """
    if (image_dir is not None and files is not None) or (
        image_dir is None and files is None
    ):
        raise ValueError(
            "Either `image_dir` or `files` must be provided, but not both."
        )
    elif files is not None:
        return (ImageFilename(str(fpath.resolve())) for fpath in files)
    elif image_dir is not None:
        return (
            ImageFilename(str(fpath.relative_to(image_dir)))
            for fpath in _get_image_filepaths(image_dir=image_dir)
        )
    else:
        raise ValueError("Either `image_dir` or `files` must be provided.")


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
            img_or_dir.suffix in _pil_supported_image_extensions()
        ):
            yield img_or_dir.resolve()
        elif img_or_dir.is_dir():
            yield from _get_image_filepaths(img_or_dir)
        else:
            raise ValueError(f"Invalid path: {img_or_dir}")


def _get_image_filepaths(image_dir: Path) -> Iterable[Path]:
    extensions = _pil_supported_image_extensions()
    for root, _, files in os.walk(image_dir, followlinks=True):
        root_path = Path(root)
        for file in files:
            fpath = root_path / file
            if fpath.suffix.lower() in extensions:
                yield fpath


def _pil_supported_image_extensions() -> set[str]:
    return {
        ex
        for ex, format in Image.registered_extensions().items()
        if format in Image.OPEN
    }


def _torchvision_supported_image_extensions() -> set[str]:
    # See https://pytorch.org/vision/0.18/generated/torchvision.io.read_image.html
    return {"jpg", "jpeg", "png"}


def _open_image(image_path: Path, mode: Literal["RGB", "L"] = "RGB") -> NDArrayImage:
    image_np: NDArray[np.uint8]
    if image_path.suffix.lower() in _torchvision_supported_image_extensions():
        mode_torch = {"RGB": ImageReadMode.RGB, "L": ImageReadMode.GRAY}[mode]
        image_torch = torchvision.io.read_image(str(image_path), mode=mode_torch)
        image_torch = image_torch.permute(1, 2, 0)
        image_np = image_torch.numpy()
    else:
        image = Image.open(image_path).convert(mode)
        image_np = np.array(image)
    return image_np
