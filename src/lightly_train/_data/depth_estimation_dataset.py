#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import data_helpers, file_helpers
from lightly_train._data.file_helpers import ImageMode
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.task_dataset import TaskDataset, TaskDatasetArgs
from lightly_train._transforms.depth_estimation_transform import (
    DepthEstimationCollateFunction,
    DepthEstimationTransform,
)
from lightly_train._transforms.task_transform import TaskCollateFunction
from lightly_train.types import (
    DepthEstimationDatasetItem,
    NDArrayDepth,
    PathLike,
)


class DepthEstimationDataset(TaskDataset):
    # Narrow the type of dataset_args.
    dataset_args: DepthEstimationDatasetArgs

    batch_collate_fn_cls: ClassVar[type[TaskCollateFunction]] = (
        DepthEstimationCollateFunction
    )

    def __init__(
        self,
        dataset_args: DepthEstimationDatasetArgs,
        image_info: Sequence[dict[str, str]],
        transform: DepthEstimationTransform | None = None,
    ) -> None:
        super().__init__(
            transform=transform, dataset_args=dataset_args, image_info=image_info
        )

    def __getitem__(self, index: int) -> DepthEstimationDatasetItem:
        row = self.image_info[index]
        image_path = row["image_filepaths"]
        depth_path = row["depth_filepaths"]
        sky_path = row["sky_filepaths"]

        image = file_helpers.open_image_numpy(
            image_path=Path(image_path), mode=ImageMode.RGB
        )
        depth = _load_npy(Path(depth_path))
        sky = _load_sky_png(Path(sky_path))

        if image.shape[:2] != depth.shape[:2] or image.shape[:2] != sky.shape[:2]:
            raise ValueError(
                f"Shape mismatch for '{image_path}': image (height, width) is "
                f"{image.shape[:2]}, depth is {depth.shape[:2]}, sky is "
                f"{sky.shape[:2]}."
            )

        transformed = self.transform({"image": image, "depth": depth, "sky": sky})
        return DepthEstimationDatasetItem(
            image_path=str(image_path),  # Str for torch dataloader compatibility.
            image=transformed["image"],
            depth=transformed["depth"],
            sky=transformed["sky"],
        )


class DepthEstimationDatasetArgs(TaskDatasetArgs):
    image_dir: Path
    depth_dir: Path
    sky_dir: Path

    def list_image_info(self) -> Iterable[dict[str, str]]:
        for image_filename in file_helpers.list_image_filenames_from_dir(
            image_dir=self.image_dir
        ):
            image_filepath = self.image_dir / image_filename
            stem = Path(image_filename).stem
            depth_filepath = (self.depth_dir / stem).with_suffix(".npy")
            sky_filepath = (self.sky_dir / stem).with_suffix(".png")
            if depth_filepath.exists() and sky_filepath.exists():
                yield {
                    "image_filepaths": str(image_filepath),
                    "depth_filepaths": str(depth_filepath),
                    "sky_filepaths": str(sky_filepath),
                }

    @staticmethod
    def get_dataset_cls() -> type[DepthEstimationDataset]:
        return DepthEstimationDataset


class SplitArgs(PydanticConfig):
    images: PathLike
    depth: PathLike
    sky: PathLike


class DepthEstimationDataArgs(TaskDataArgs):
    train: SplitArgs
    val: SplitArgs

    def resolve_data_paths(self, base_dir: Path) -> None:
        self.train.images = data_helpers.resolve_path(
            self.train.images, base_dir=base_dir
        )
        self.train.depth = data_helpers.resolve_path(
            self.train.depth, base_dir=base_dir
        )
        self.train.sky = data_helpers.resolve_path(self.train.sky, base_dir=base_dir)
        self.val.images = data_helpers.resolve_path(self.val.images, base_dir=base_dir)
        self.val.depth = data_helpers.resolve_path(self.val.depth, base_dir=base_dir)
        self.val.sky = data_helpers.resolve_path(self.val.sky, base_dir=base_dir)

    @property
    def included_classes(self) -> dict[int, str]:
        # Depth estimation is a dense regression task and has no classes.
        return {}

    def train_data_mmap_hash(self) -> str:
        return str(
            (
                Path(self.train.images).resolve(),
                Path(self.train.depth).resolve(),
                Path(self.train.sky).resolve(),
            )
        )

    def val_data_mmap_hash(self) -> str:
        return str(
            (
                Path(self.val.images).resolve(),
                Path(self.val.depth).resolve(),
                Path(self.val.sky).resolve(),
            )
        )

    def get_train_args(self) -> DepthEstimationDatasetArgs:
        return DepthEstimationDatasetArgs(
            image_dir=Path(self.train.images),
            depth_dir=Path(self.train.depth),
            sky_dir=Path(self.train.sky),
        )

    def get_val_args(self) -> DepthEstimationDatasetArgs:
        return DepthEstimationDatasetArgs(
            image_dir=Path(self.val.images),
            depth_dir=Path(self.val.depth),
            sky_dir=Path(self.val.sky),
        )


def _load_npy(path: Path) -> NDArrayDepth:
    """Loads a (H, W) float depth pseudo-label from a ``.npy`` file."""
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D (height, width) array in '{path}', got shape {array.shape}."
        )
    result: NDArrayDepth = array.astype(np.float32)
    return result


def _load_sky_png(path: Path) -> NDArrayDepth:
    """Loads a (H, W) sky pseudo-label from an 8-bit grayscale ``.png`` mask.

    The mask is stored as a single-channel ``{0, 255}`` image (white = sky) and is
    scaled to a ``{0.0, 1.0}`` float probability map for the BCE sky-distillation loss.
    """
    array = file_helpers.open_image_numpy(image_path=path, mode=ImageMode.UNCHANGED)
    # An 8-bit grayscale PNG decodes to (H, W) or (H, W, 1); drop a singleton channel.
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D (height, width) sky mask in '{path}', got shape "
            f"{array.shape}."
        )
    result: NDArrayDepth = array.astype(np.float32) / 255.0
    return result
