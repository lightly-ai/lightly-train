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
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, get_args

import fsspec
import numpy as np
import torch
from lightning_utilities.core.imports import RequirementCache
from PIL import Image, ImageFile
from PIL.Image import Image as PILImage
from torch import Tensor
from torchvision.io import ImageReadMode
from torchvision.transforms.v2 import functional as F

from lightly_train.types import (
    ImageDtypes,
    ImageFilename,
    NDArrayBBoxes,
    NDArrayClasses,
    NDArrayImage,
    NDArrayMask,
    NDArrayPolygon,
    PathLike,
)

logger = logging.getLogger(__name__)

TORCHVISION_GEQ_0_20_0 = RequirementCache("torchvision>=0.20.0")
if TORCHVISION_GEQ_0_20_0:
    # `read_image` is marked as obsolete in torchvision>=0.20.0 in favor of `decode_image`.
    # `decode_image` can additionally load uint16 masks, but can only accept str inputs in torchvision>=0.20.0
    # so we use `decode_image` for torchvision>=0.20.0 and fall back to `read_image` otherwise.
    from torchvision.io import decode_image as load_image
else:
    from torchvision.io import read_image as load_image  # type: ignore[no-redef]

PYDICOM_GEQ_3_0_0 = RequirementCache("pydicom>=3.0.0")


class ImageMode(Enum):
    RGB = "RGB"
    UNCHANGED = "UNCHANGED"


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
    supported_extensions = _supported_image_extensions()
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


def _supported_image_extensions() -> set[str]:
    return _pil_supported_image_extensions() | {".dcm"}


def _get_image_filenames(
    image_dir: PathLike, image_extensions: Set[str] | None = None
) -> Iterable[str]:
    """Returns image filenames relative to image_dir."""
    image_extensions = (
        _supported_image_extensions() if image_extensions is None else image_extensions
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


def _image_src_from_path(path: PathLike) -> PathLike | BinaryIO:
    protocol = fsspec.utils.get_protocol(str(path))
    if protocol == "file":
        return path
    with fsspec.open(path, "rb") as file:
        return BytesIO(file.read())


def as_image_tensor(image: PathLike | PILImage | Tensor) -> Tensor:
    """Returns image as (C, H, W) tensor."""
    if isinstance(image, Tensor):
        return image
    elif isinstance(image, PILImage):
        image_tensor: Tensor = F.pil_to_tensor(image)
        return image_tensor
    else:
        return open_image_tensor(image)


def open_image_tensor(image_path: PathLike) -> Tensor:
    """Returns image as (C, H, W) tensor.

    Args:
        image_path: Path to the image file. Can be a local path or URL.
    """
    image: Tensor

    suffix = Path(image_path).suffix.lower()
    if suffix in _TORCHVISION_SUPPORTED_IMAGE_EXTENSIONS:
        try:
            # Fast path when loading local file with torch.
            image = load_image(str(image_path))
        except RuntimeError:
            # RuntimeError can happen for images that cannot be read by torch (e.g. URLs).
            pass
        else:
            return image
    image_src = _image_src_from_path(image_path)

    if suffix == ".dcm":
        image_np = _open_image_numpy__with_pydicom(image_src=image_src)
        if image_np.ndim == 2:
            # (H, W) -> (H, W, C)
            image_np = np.expand_dims(image_np, axis=2)
        # (H, W, C) -> (C, H, W)
        image_np = np.transpose(image_np, (2, 0, 1))
        image = Tensor(image_np)
        return image

    image = F.pil_to_tensor(Image.open(image_src))
    return image


def open_image_numpy(
    image_path: Path,
    mode: ImageMode = ImageMode.RGB,
) -> NDArrayImage:
    """Returns image as (H, W, C) or (H, W) numpy array."""
    image_np: NDArrayImage

    # Torchvision supported images
    suffix = image_path.suffix.lower()
    if suffix in _TORCHVISION_SUPPORTED_IMAGE_EXTENSIONS:
        try:
            image_np = _open_image_numpy__with_torch(image_path=image_path, mode=mode)
        except RuntimeError:
            # RuntimeError can happen for truncated images. Fall back to PIL.
            image_np = _open_image_numpy__with_pil(image_path=image_path, mode=mode)
    # DICOM images. ImageMode is not relevant here. It will always be loaded as is.
    # NOTE: We do not support loading DICOM images as segmentation masks.
    elif suffix == ".dcm":
        image_np = _open_image_numpy__with_pydicom(image_src=image_path)
    # Pillow images
    else:
        image_np = _open_image_numpy__with_pil(image_path=image_path, mode=mode)

    return image_np


def _open_image_numpy__with_torch(
    image_path: Path,
    mode: ImageMode = ImageMode.RGB,
) -> NDArrayImage:
    image_np: NDArrayImage

    mode_torch = {
        ImageMode.RGB: ImageReadMode.RGB,
        ImageMode.UNCHANGED: ImageReadMode.UNCHANGED,
    }[mode]
    image_torch = load_image(str(image_path), mode=mode_torch)
    image_torch = image_torch.permute(1, 2, 0)

    if image_torch.shape[2] == 1 and mode == ImageMode.RGB:
        # Convert single-channel grayscale to 3-channel RGB.
        # (H, W, 1) -> (H, W, 3)
        image_torch = image_torch.repeat(1, 1, 3)
    if image_torch.dtype != torch.uint8:
        # Convert to float32 image in [0, 1] range for non-uint8 types because albumentations only supports
        # np.float32 and np.uint8 types.
        image_torch = F.to_dtype(image_torch, torch.float32, scale=True)

    image_np = image_torch.numpy()
    return image_np


def _open_image_numpy__with_pydicom(
    image_src: PathLike | BinaryIO,
) -> NDArrayImage:
    if not RequirementCache("pydicom"):
        raise ImportError(
            "pydicom is required to read DICOM images. "
            "Please install it with 'pip install lightly-train[dicom]'."
        )
    from pydicom import Dataset

    if PYDICOM_GEQ_3_0_0:
        from pydicom.pixels import (  # type: ignore[import-not-found]
            utils as pydicom_utils,
        )
        from pydicom.pixels.processing import (  # type: ignore[import-not-found]
            apply_color_lut,
            apply_modality_lut,
            convert_color_space,
        )
    else:
        import pydicom.pixel_data_handlers.util as pydicom_utils  # type: ignore[no-redef]
        from pydicom.pixel_data_handlers.util import (  # type: ignore[no-redef]
            apply_color_lut,
            apply_modality_lut,
            convert_color_space,
        )

    image_np: NDArrayImage

    dataset = Dataset()
    pixel_array = pydicom_utils.pixel_array(image_src, ds_out=dataset)

    num_frames = pydicom_utils.get_nr_frames(dataset)
    if num_frames > 1:
        raise ValueError("Multi-frame DICOM images are not supported.")

    pm = dataset.PhotometricInterpretation
    if (
        pixel_array.shape[-1] == 3
        and np.issubdtype(pixel_array.dtype, np.uint8)
        and "YBR_FULL" in pm
    ):
        pixel_array = convert_color_space(pixel_array, pm, "RGB")

    if pm == "PALETTE COLOR":
        pixel_array = apply_color_lut(pixel_array, dataset)

    rescaled_array = apply_modality_lut(pixel_array, dataset)

    image_np = (
        rescaled_array.astype(np.float32)
        if not np.issubdtype(rescaled_array.dtype, np.integer)
        else rescaled_array
    )
    original_dtype = image_np.dtype
    if not any(
        np.issubdtype(original_dtype, allowed) for allowed in get_args(ImageDtypes)
    ):
        # Convert to float32 image in [0, 1] range for non-uint8 types because albumentations only supports
        # np.float32 and np.uint8 types.
        image_np = image_np.astype(np.float32)

        info = np.iinfo(original_dtype)  # type: ignore[type-var]
        image_np = (image_np - float(info.min)) / float(info.max - info.min)

    return image_np


def _open_image_numpy__with_pil(
    image_path: Path,
    mode: ImageMode = ImageMode.RGB,
) -> NDArrayImage:
    image_np: NDArrayImage

    image: PILImage | ImageFile.ImageFile = Image.open(image_path)
    if mode != ImageMode.UNCHANGED:
        image = image.convert(mode.value)

    image_np = np.array(image)
    original_dtype = image_np.dtype
    if not any(
        np.issubdtype(original_dtype, allowed) for allowed in get_args(ImageDtypes)
    ):
        # Convert to float32 image in [0, 1] range for non-uint8 types because albumentations only supports
        # np.float32 and np.uint8 types.
        image_np = image_np.astype(np.float32)

        # Here we assume that all the other images uses integer types.
        # See https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes for more details.
        info = np.iinfo(original_dtype)  # type: ignore[type-var]
        image_np = (image_np - float(info.min)) / float(info.max - info.min)

    return image_np


def open_mask_numpy(
    mask_path: Path,
) -> NDArrayMask:
    """Returns mask as (H, W, C) or (H, W) numpy array."""
    mask_np: NDArrayMask
    if mask_path.suffix.lower() in _TORCHVISION_SUPPORTED_IMAGE_EXTENSIONS:
        try:
            mask_np = _open_mask_numpy__with_torch(mask_path=mask_path)
        except RuntimeError:
            # RuntimeError can happen for truncated images. Fall back to PIL.
            mask_np = _open_mask_numpy__with_pil(mask_path=mask_path)
    else:
        mask_np = _open_mask_numpy__with_pil(mask_path=mask_path)

    return mask_np


def _open_mask_numpy__with_torch(
    mask_path: Path,
) -> NDArrayMask:
    mask_np: NDArrayMask

    mask_torch = load_image(str(mask_path))
    mask_torch = mask_torch.permute(1, 2, 0)

    if mask_torch.shape[2] == 1:
        # Squeeze channel dimension for single-channel masks.
        # (H, W, 1) -> (H, W)
        mask_torch = mask_torch.squeeze(2)

    mask_np = mask_torch.numpy()
    return mask_np


def _open_mask_numpy__with_pil(
    mask_path: Path,
) -> NDArrayMask:
    mask_np: NDArrayMask

    mask: PILImage | ImageFile.ImageFile = Image.open(mask_path)
    mask_np = np.array(mask)

    return mask_np


def open_yolo_object_detection_label_numpy(
    label_path: Path,
) -> tuple[NDArrayBBoxes, NDArrayClasses]:
    """Open a YOLO label file and return the bounding boxes and classes as numpy arrays.

    Returns:
        (bboxes, classes) tuple. All values are in normalized coordinates
        between [0, 1]. Bboxes are formatted as (x_center, y_center, width, height).
    """
    bboxes = []
    classes = []
    for line in _iter_yolo_label_lines(label_path=label_path):
        parts = [float(x) for x in line.split()]
        class_id = parts[0]
        x_center = parts[1]
        y_center = parts[2]
        width = parts[3]
        height = parts[4]
        bboxes.append([x_center, y_center, width, height])
        classes.append(int(class_id))
    bboxes_np = np.array(bboxes) if bboxes else np.zeros((0, 4), dtype=np.float64)
    classes_np = np.array(classes, dtype=np.int64)
    return bboxes_np, classes_np


def open_yolo_instance_segmentation_label_numpy(
    label_path: Path,
) -> tuple[list[NDArrayPolygon], NDArrayBBoxes, NDArrayClasses]:
    """Open a YOLO label file and return the polygons, bboxes, and classes as numpy
    arrays.

    Returns:
        (polygons, bboxes, classes) tuple. All values are in normalized coordinates
        between [0, 1]. Polygons are list of numpy arrays of shape (n_points*2,) and
        each array is a sequence of x0, y0, x1, y1, ... coordinates.
        Bboxes are formatted as (x_center, y_center, width, height).
    """
    classes = []
    polygons = []
    bboxes = []
    for line in _iter_yolo_label_lines(label_path=label_path):
        parts = [float(x) for x in line.split()]
        class_id = parts[0]
        polygon = np.array(parts[1:], dtype=np.float64)
        classes.append(int(class_id))
        polygons.append(polygon)
        bboxes.append(_bbox_from_polygon(polygon))
    classes_np = np.array(classes, dtype=np.int64)
    bboxes_np = np.stack(bboxes) if bboxes else np.zeros((0, 4), dtype=np.float64)
    return polygons, bboxes_np, classes_np


def _bbox_from_polygon(polygon: NDArrayPolygon) -> NDArrayBBoxes:
    xs = polygon[0::2]
    ys = polygon[1::2]
    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    bbox = np.array([x_center, y_center, width, height], dtype=np.float64)
    return bbox


def _iter_yolo_label_lines(label_path: Path) -> Iterable[str]:
    """Yield lines from a YOLO label file.

    Skips empty and duplicate lines.
    """
    lines = set()
    with open(label_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            # Skip empty lines.
            if not line:
                continue
            # Skip duplicate lines.
            if line in lines:
                continue
            lines.add(line)
            yield line
