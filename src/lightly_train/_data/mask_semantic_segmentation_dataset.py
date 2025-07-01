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

from torch.utils.data import Dataset

from lightly_train._configs.config import PydanticConfig
from lightly_train._data import file_helpers
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._transforms.task_transform import TaskTransform
from lightly_train.types import (
    ImageFilename,
    MaskSemanticSegmentationDatasetItem,
    PathLike,
)


class MaskSemanticSegmentationDataset(Dataset[MaskSemanticSegmentationDatasetItem]):
    def __init__(
        self,
        dataset_args: MaskSemanticSegmentationDatasetArgs,
        image_filenames: Sequence[ImageFilename],
        transform: TaskTransform,
    ):
        self.args = dataset_args
        self.image_filenames = image_filenames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> MaskSemanticSegmentationDatasetItem:
        image_filename = self.image_filenames[index]
        image_path = self.args.image_dir / image_filename
        mask_path = (self.args.mask_dir / image_filename).with_suffix(".png")

        image = file_helpers.open_image(image_path=image_path, mode="RGB")
        mask = file_helpers.open_image(image_path=mask_path, mode="L")

        transformed = self.transform({"image": image, "mask": mask})
        return {
            "image_path": str(image_path),  # Str for torch dataloader compatibility.
            "image": transformed["image"],
            "mask": transformed["mask"],
        }


class MaskSemanticSegmentationDatasetArgs(PydanticConfig):
    image_dir: Path
    mask_dir: Path
    classes: dict[int, str] | None = None

    def list_image_filenames(self) -> Iterable[ImageFilename]:
        for image_filename in file_helpers.list_image_filenames(
            image_dir=self.image_dir
        ):
            if (self.mask_dir / image_filename).with_suffix(".png").exists():
                yield image_filename


class MaskSemanticSegmentationDataArgs(TaskDataArgs):
    train: SplitArgs
    val: SplitArgs
    classes: dict[int, str]

    @staticmethod
    def get_dataset_cls() -> type[MaskSemanticSegmentationDataset]:
        return MaskSemanticSegmentationDataset

    def get_train_args(self) -> MaskSemanticSegmentationDatasetArgs:
        return MaskSemanticSegmentationDatasetArgs(
            image_dir=Path(self.train.images),
            mask_dir=Path(self.train.masks),
            classes=self.classes,
        )

    def get_val_args(self) -> MaskSemanticSegmentationDatasetArgs:
        return MaskSemanticSegmentationDatasetArgs(
            image_dir=Path(self.val.images),
            mask_dir=Path(self.val.masks),
            classes=self.classes,
        )


class SplitArgs(PydanticConfig):
    images: PathLike
    masks: PathLike
