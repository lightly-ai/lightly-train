#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from PIL import ImageFile
from torch.utils.data import Dataset

from lightly_train._data import file_helpers
from lightly_train._data.file_helpers import ImageMode
from lightly_train._env import Env
from lightly_train.types import DatasetItem, ImageFilename, Transform, TransformInput

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset[DatasetItem]):
    def __init__(
        self,
        image_dir: Path | None,
        image_filenames: Sequence[Mapping[str, ImageFilename]],
        transform: Transform,
        num_channels: int,
        mask_dir: Path | None = None,
    ):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_channels = num_channels

        image_mode = (
            None
            if Env.LIGHTLY_TRAIN_IMAGE_MODE.value is None
            else ImageMode(Env.LIGHTLY_TRAIN_IMAGE_MODE.value)
        )
        if image_mode is None:
            image_mode = (
                ImageMode.RGB if self.num_channels == 3 else ImageMode.UNCHANGED
            )

        if image_mode not in (ImageMode.RGB, ImageMode.UNCHANGED):
            raise ValueError(
                f"Invalid image mode: '{image_mode}'. "
                f"Supported modes are '{[ImageMode.RGB.value, ImageMode.UNCHANGED.value]}'."
            )
        self.image_mode = image_mode

    def __getitem__(self, idx: int) -> DatasetItem:
        filename = self.image_filenames[idx]["filenames"]
        if self.image_dir is None:
            image = file_helpers.open_image_numpy(Path(filename), mode=self.image_mode)
        else:
            image = file_helpers.open_image_numpy(
                self.image_dir / filename, mode=self.image_mode
            )

        input: TransformInput = {"image": image}

        if self.mask_dir:
            maskname = Path(filename).with_suffix(".png")
            mask = file_helpers.open_image_numpy(
                self.mask_dir / maskname, mode=ImageMode.MASK
            )
            input["mask"] = mask

        # (H, W, C) -> (C, H, W)
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
