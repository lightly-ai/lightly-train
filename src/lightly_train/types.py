#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, TypedDict, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from typing_extensions import NotRequired

# Underlying model type of the packages. Most of the time this is a torch.nn.Module
# however, for in some instances they can be custom classes with nn.Modules only in the
# attributes.
PackageModel = Any

# Types for the new transforms.
NDArrayImage = NDArray[np.uint8]
NDArrayBBoxes = NDArray[np.float64]  # (n_boxes, 4)
NDArrayClasses = NDArray[np.int64]  # (n_boxes,)


class TransformInput(TypedDict):
    image: NDArrayImage
    mask: NotRequired[NDArrayImage]
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


# Replaces torch.optim.optimizer.ParamsT
# as it is only available in torch>=v2.2.
# Importing it conditionally cannot make typing work for both older
# and newer versions of torch.
ParamsT = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

PathLike = Union[str, Path]
