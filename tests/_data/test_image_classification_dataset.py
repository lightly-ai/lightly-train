#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lightly_train._data.image_classification_dataset import (
    ImageClassificationDataArgs,
    ImageClassificationDataset,
)
from lightly_train._transforms.task_transform import TaskTransform, TaskTransformArgs

from .. import helpers


class IdentityTaskTransformArgs(TaskTransformArgs):
    """Dummy args class for the identity transform."""

    pass


class IdentityTaskTransform(TaskTransform):
    transform_args_cls = IdentityTaskTransformArgs

    def __call__(self, input: Any) -> Any:
        return input


class TestImageClassifiactionDataset:
    def test__image_folder(self, tmp_path: Path) -> None:
        # Create the dummy dataset.
        num_files_per_class = 2
        classes = {0: "class_0", 1: "class_1"}
        helpers.create_image_classification_dataset(
            tmp_path=tmp_path,
            class_names=list(classes.values()),
            num_files_per_class=num_files_per_class,
            height=64,
            width=128,
        )

        args = ImageClassificationDataArgs(
            train=tmp_path / "train",
            val=tmp_path / "val",
            names=classes,
        )
        train_args = args.get_train_args()
        val_args = args.get_val_args()

        train_dataset = ImageClassificationDataset(
            dataset_args=train_args,
            transform=_get_transform(),
            image_info=list(train_args.list_image_info()),
        )

        val_dataset = ImageClassificationDataset(
            dataset_args=args.get_val_args(),
            transform=_get_transform(),
            image_info=list(val_args.list_image_info()),
        )

        assert len(train_dataset) == len(classes) * num_files_per_class
        assert len(val_dataset) == len(classes) * num_files_per_class

        sample = train_dataset[0]
        assert sample["image"].dtype == torch.float32
        assert sample["image"].shape == (3, 64, 128)
        assert sample["classes"].dtype == torch.long
        assert sample["classes"].shape == (1,)
        # Classes are mapped to internal class ids in [0, num_included_classes - 1]
        assert torch.all(sample["classes"] <= 1)

        sample = val_dataset[0]
        assert sample["image"].shape == (3, 64, 128)
        assert sample["classes"].shape == (1,)


def _get_transform() -> IdentityTaskTransform:
    transform_args = IdentityTaskTransformArgs()
    return IdentityTaskTransform(transform_args=transform_args)
