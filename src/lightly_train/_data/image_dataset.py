#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from PIL import ImageFile
from torch.utils.data import Dataset

from lightly_train._data import file_helpers
from lightly_train._env import Env
from lightly_train.types import DatasetItem, ImageFilename, Transform, TransformInput

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
            image = file_helpers.open_image_numpy(Path(filename), mode=self.image_mode)
        else:
            image = file_helpers.open_image_numpy(
                self.image_dir / filename, mode=self.image_mode
            )

        input: TransformInput = {"image": image}

        if self.mask_dir:
            maskname = Path(filename).with_suffix(".png")
            mask = file_helpers.open_image_numpy(self.mask_dir / maskname, mode="L")
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
