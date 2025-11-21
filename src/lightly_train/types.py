#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, TypedDict, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BeforeValidator, Field
from torch import Tensor
from typing_extensions import Annotated, NotRequired

# Underlying model type of the packages. Most of the time this is a torch.nn.Module
# however, for in some instances they can be custom classes with nn.Modules only in the
# attributes.
PackageModel = Any

# Types for the new transforms.
ImageDtypes = Union[np.uint8, np.float32]
NDArrayImage = NDArray[ImageDtypes]  # (H, W) or (H, W, C)
NDArrayMask = NDArray[Union[np.uint8, np.uint16, np.int_]]  # (H, W) or (H, W, C)
NDArrayBBoxes = NDArray[np.float64]  # (n_boxes, 4)
NDArrayClasses = NDArray[np.int64]  # (n_boxes,)
# Array with x0, y0, x1, y1, x2, y2, ... coordinates of the polygon points. Coordinates
# are in [0, 1].
NDArrayPolygon = NDArray[np.float64]  # (n_points*2,)
NDArrayBinaryMask = NDArray[np.bool_]  # (H, W)
NDArrayBinaryMasks = NDArray[np.bool_]  # (n_instances, H, W)
# Binary masks as integers for compatibility with albumentations as it doesn't support
# boolean masks.
NDArrayBinaryMasksInt = NDArray[np.int_]  # (n_instances, H, W)


class TransformInput(TypedDict):
    image: NDArrayImage
    mask: NotRequired[NDArrayMask]
    # TODO: bbox: NDArray[np.float64] | None


class TransformOutputSingleView(TypedDict):
    image: Tensor
    mask: NotRequired[Tensor]  # | None
    # TODO: bbox: Tensor | None


TransformOutput = List[TransformOutputSingleView]
Transform = Callable[[TransformInput], TransformOutput]


# Types for the dataset items (input to the dataloader collate) and the
# Batch (output of the dataloader collate).
ImageFilename = str


class DatasetItem(TypedDict):
    filename: ImageFilename
    views: list[Tensor]  # One tensor per view, of shape (3, H, W) each.
    masks: NotRequired[list[Tensor]]  # One tensor per view, of shape (H, W) each


# The type and variable names of the Batch is fully determined by the type and
# variable names of the DatasetItem by the dataloader collate function.
class Batch(TypedDict):
    filename: list[ImageFilename]  # length==batch_size
    views: list[Tensor]  # One tensor per view, of shape (batch_size, 3, H, W) each.
    masks: NotRequired[
        list[Tensor]
    ]  # One tensor per view, of shape (batch_size, H, W) each.


class TaskDatasetItem(TypedDict):
    pass


class TaskBatch(TypedDict):
    pass


class BinaryMasksDict(TypedDict):
    # Boolean tensor with shape (num_classes_in_image, H, W).
    masks: Tensor
    # Class labels corresponding to the boolean masks. Tensor with shape
    # (num_classes_in_image,)
    labels: Tensor


class PanopticBinaryMasksDict(TypedDict):
    # Boolean tensor with shape (num_instances, H, W).
    masks: Tensor
    # Class labels corresponding to the boolean masks. Tensor with shape
    # (num_instances,)
    labels: Tensor
    iscrowd: Tensor  # Tensor with shape (num_instances,)


class MaskSemanticSegmentationDatasetItem(TaskDatasetItem):
    image_path: ImageFilename
    image: Tensor
    mask: Tensor
    binary_masks: BinaryMasksDict


class MaskSemanticSegmentationBatch(TypedDict):
    image_path: list[ImageFilename]  # length==batch_size
    # Tensor with shape (batch_size, 3, H, W) or list of Tensors with shape (3, H, W).
    image: Tensor | list[Tensor]
    # Tensor with shape (batch_size, H, W) or list of Tensors with shape (H, W).
    mask: Tensor | list[Tensor]
    binary_masks: list[BinaryMasksDict]  # On dict per image.


class ObjectDetectionDatasetItem(TypedDict):
    image_path: ImageFilename
    image: Tensor
    bboxes: Tensor  # Of shape (n_boxes, 4) with (x_center, y_center, w, h) coordinates.
    classes: Tensor  # Of shape (n_boxes,) with class labels.
    original_size: tuple[int, int]  # (width, height) of the original image.


class ObjectDetectionBatch(TypedDict):
    image_path: list[ImageFilename]  # length==batch_size
    image: Tensor  # Tensor with shape (batch_size, 3, H, W).
    bboxes: list[Tensor]  # One tensor per image, each of shape (n_boxes, 4).
    classes: list[Tensor]  # One tensor per image, each of shape (n_boxes,).
    original_size: list[tuple[int, int]]  # One (width, height) per image.


class InstanceSegmentationDatasetItem(TaskDatasetItem):
    image_path: ImageFilename
    image: Tensor
    binary_masks: BinaryMasksDict  # Dict with (n_instances,) masks and labels.
    bboxes: (
        Tensor  # Of shape (n_instances, 4) with (x_center, y_center, w, h) coordinates.
    )
    classes: Tensor  # Of shape (n_instances,) with class labels.


class InstanceSegmentationBatch(TypedDict):
    image_path: list[ImageFilename]  # length==batch_size
    # Tensor with shape (batch_size, C, H, W) or list of Tensors with shape (C, H, W).
    image: Tensor | list[Tensor]
    # One dict per image, each dict contains (n_instances,) masks and labels.
    binary_masks: list[BinaryMasksDict]
    bboxes: list[Tensor]  # One tensor per image, each of shape (n_instances, 4).
    classes: list[Tensor]  # One tensor per image, each of shape (n_instances,).


class MaskPanopticSegmentationDatasetItem(TaskDatasetItem):
    image_path: ImageFilename
    image: Tensor
    # Tensor with shape (H, W, 2) where the last dimension contains (label, segment_id).
    masks: Tensor
    binary_masks: PanopticBinaryMasksDict  # Dict with (n_instances,) masks and labels.


class MaskPanopticSegmentationBatch(TypedDict):
    image_path: list[ImageFilename]  # length==batch_size
    # Tensor with shape (batch_size, 3, H, W) or list of Tensors with shape (3, H, W).
    image: Tensor | list[Tensor]
    # Tensor with shape (batch_size, H, W, 2) or list of Tensors with shape (H, W, 2).
    masks: Tensor | list[Tensor]
    binary_masks: list[PanopticBinaryMasksDict]  # One dict per image.


# Replaces torch.optim.optimizer.ParamsT
# as it is only available in torch>=v2.2.
# Importing it conditionally cannot make typing work for both older
# and newer versions of torch.
ParamsT = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

PathLike = Union[str, Path]


def _try_convert_to_tuple(value: Any) -> Any:
    # Convert to tuple if possible. Otherwise return value. Pydantic will raise the
    # appropriate error on validation.
    if isinstance(value, Iterable):
        return tuple(value)
    return value


# Strict=False to allow pydantic to automatically convert lists or other iterables to
# tuples. This happens only on model initialization and not on assignment.
# The BeforeValidator is required because strict=False doesn't work with older Pydantic
# versions when the tuple is used in a union, see: https://github.com/lightly-ai/lightly-train/pull/444
ImageSizeTuple = Annotated[
    Tuple[int, int], Field(strict=False), BeforeValidator(_try_convert_to_tuple)
]
