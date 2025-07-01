#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

import albumentations as A

from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataset,
    MaskSemanticSegmentationDatasetArgs,
)
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
)

from .. import helpers


class DummyTransform(TaskTransform):
    def __init__(self, transform_args: TaskTransformArgs):
        super().__init__(transform_args=transform_args)
        self.transform = A.Compose(
            [
                A.Resize(32, 32),
                A.ToTensorV2(),
            ]
        )

    def __call__(self, input: TaskTransformInput) -> TaskTransformOutput:
        output: TaskTransformOutput = self.transform(**input)
        return output


class TestMaskSemanticSegmentationDataset:
    def test__getitem__(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.jpg", "image1.jpg"]
        mask_filenames = ["image0.png", "image1.png"]
        helpers.create_images(image_dir, files=image_filenames)
        helpers.create_masks(mask_dir, files=mask_filenames, num_classes=2)

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir=mask_dir,
            classes={0: "background", 1: "object"},
        )
        transform = DummyTransform(transform_args=TaskTransformArgs())
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_filenames=list(dataset_args.list_image_filenames()),
            transform=transform,
        )

        assert len(dataset) == 2
        for item in dataset:  # type: ignore[attr-defined]
            assert item["image"].shape == (3, 32, 32)
            assert item["mask"].shape == (32, 32)
            assert item["mask"].min() >= 0
            assert item["mask"].max() <= 1
        assert sorted(item["image_path"] for item in dataset) == [  # type: ignore[attr-defined]
            str(image_dir / "image0.jpg"),
            str(image_dir / "image1.jpg"),
        ]
