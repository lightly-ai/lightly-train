#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torchvision.transforms import functional as F

from lightly_train._data.image_classification_dataset import (
    ImageClassificationDataArgs,
    ImageClassificationDataset,
)
from lightly_train._transforms.task_transform import TaskTransform, TaskTransformArgs

from .. import helpers


class IdentityTaskTransformArgs(TaskTransformArgs):
    """Dummy args class for the identity transform."""

    pass


class DummyTaskTransform(TaskTransform):
    transform_args_cls = IdentityTaskTransformArgs

    def __call__(self, input: Any) -> Any:
        image = input["image"]
        image = F.to_tensor(image)
        image = F.normalize(
            image, mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]
        )
        return {"image": image}


class TestImageClassificationDataset:
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
            classes=classes,
        )
        train_args = args.get_train_args()
        val_args = args.get_val_args()

        train_dataset = ImageClassificationDataset(
            dataset_args=train_args,
            transform=_get_transform(),
            image_info=list(train_args.list_image_info()),
        )

        val_dataset = ImageClassificationDataset(
            dataset_args=val_args,
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
        assert sample["image"].dtype == torch.float32
        assert sample["image"].shape == (3, 64, 128)
        assert sample["classes"].dtype == torch.long
        assert sample["classes"].shape == (1,)
        # Classes are mapped to internal class ids in [0, num_included_classes - 1]
        assert torch.all(sample["classes"] <= 1)

    def test__image_folder_non_contiguous(self, tmp_path: Path) -> None:
        # Create the dummy dataset.
        num_files_per_class = 2
        classes = {3: "class_3", 7: "class_7"}
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
            classes=classes,
        )
        train_args = args.get_train_args()
        val_args = args.get_val_args()

        train_dataset = ImageClassificationDataset(
            dataset_args=train_args,
            transform=_get_transform(),
            image_info=list(train_args.list_image_info()),
        )
        val_dataset = ImageClassificationDataset(
            dataset_args=val_args,
            transform=_get_transform(),
            image_info=list(val_args.list_image_info()),
        )

        assert len(train_dataset) == len(classes) * num_files_per_class

        # Collect the seen classes.
        seen: set[int] = set()
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            assert sample["classes"].dtype == torch.long
            assert sample["classes"].shape == (1,)

            internal_id = int(sample["classes"].item())
            # Internal IDs must be contiguous in [0, num_classes-1]
            assert 0 <= internal_id < len(classes)
            seen.add(internal_id)

        # We should see both classes in the dataset.
        assert seen == {0, 1}

        # Check the mapping is as expected.
        assert set(train_dataset.class_id_to_internal_class_id.values()) == {0, 1}
        assert set(train_dataset.class_id_to_internal_class_id.keys()) == {3, 7}

        # Check that the train and validation mapping is identical
        assert (
            train_dataset.class_id_to_internal_class_id
            == val_dataset.class_id_to_internal_class_id
        )

    @pytest.mark.parametrize(
        "csv_label_type,csv_label_col",
        [("id", "class_id"), ("name", "label")],
    )
    @pytest.mark.parametrize("label_delimiter", [",", ";"])
    def test__csv_random_multilabel_encoded_in_path(
        self,
        tmp_path: Path,
        csv_label_type: Literal["id", "name"],
        csv_label_col: str,
        label_delimiter: str,
    ) -> None:
        # Set the classes.
        classes = {3: "class_3", 7: "class_7", 11: "class_11"}

        # Create the dataset (images and csv files).
        helpers.create_multilabel_image_classification_dataset(
            tmp_path=tmp_path,
            classes=classes,
            num_files=6,
            height=64,
            width=128,
            csv_image_col="image_path",
            csv_label_col=csv_label_col,
            csv_label_type=csv_label_type,
            label_delimiter=label_delimiter,
        )

        args = ImageClassificationDataArgs(
            train=tmp_path / "train",
            val=tmp_path / "val",
            classes=classes,
            train_csv=tmp_path / "train.csv",
            val_csv=tmp_path / "val.csv",
            csv_image_col="image_path",
            csv_label_col=csv_label_col,
            csv_label_type=csv_label_type,
            label_delimiter=label_delimiter,
        )

        train_args = args.get_train_args()
        val_args = args.get_val_args()

        train_dataset = ImageClassificationDataset(
            dataset_args=train_args,
            transform=_get_transform(),
            image_info=list(train_args.list_image_info()),
        )
        val_dataset = ImageClassificationDataset(
            dataset_args=val_args,
            transform=_get_transform(),
            image_info=list(val_args.list_image_info()),
        )

        # Verify the length of datasets.
        assert len(train_dataset) == 6
        assert len(val_dataset) == 6

        ext_to_int = train_dataset.class_id_to_internal_class_id
        assert set(ext_to_int.keys()) == set(classes.keys())
        assert set(ext_to_int.values()) == set(range(len(classes)))

        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            img_path = Path(sample["image_path"])
            assert img_path.is_absolute()

            m = re.search(r"ids=([0-9\-]+)__i=", img_path.name)
            assert m is not None
            expected_ext_ids = [int(x) for x in m.group(1).split("-") if x != ""]

            expected_int_ids = torch.tensor(
                [ext_to_int[cid] for cid in expected_ext_ids],
                dtype=torch.long,
            )

            assert sample["classes"].dtype == torch.long
            assert torch.equal(sample["classes"], expected_int_ids)

    def test__csv_ignore_classes_filters_labels_and_skips_empty(
        self, tmp_path: Path
    ) -> None:
        # Create the dummy dataset.
        num_files_per_class = 2
        classes = {3: "class_3", 7: "class_7", 11: "class_11"}
        helpers.create_image_classification_dataset(
            tmp_path=tmp_path,
            class_names=list(classes.values()),
            num_files_per_class=num_files_per_class,
            height=64,
            width=128,
        )

        train_dir = tmp_path / "train"

        # Pick one image per class folder (absolute paths).
        img3 = sorted((train_dir / "class_3").iterdir())[0].resolve()
        img7 = sorted((train_dir / "class_7").iterdir())[0].resolve()
        img11 = sorted((train_dir / "class_11").iterdir())[0].resolve()

        # Create a CSV:
        # - First row: mixed labels (3,7) -> after ignore [7], should become (3)
        # - Second row: only ignored label (7) -> should be skipped
        # - Third row: label (11) -> kept
        csv_path = tmp_path / "train.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "class_id"])
            writer.writeheader()
            writer.writerow({"image_path": str(img3), "class_id": "3,7"})
            writer.writerow({"image_path": str(img7), "class_id": "7"})
            writer.writerow({"image_path": str(img11), "class_id": "11"})

        args = ImageClassificationDataArgs(
            train=train_dir,
            val=tmp_path / "val",
            classes=classes,
            train_csv=csv_path,
            csv_image_col="image_path",
            csv_label_col="class_id",
            csv_label_type="id",
            label_delimiter=",",
            ignore_classes={7},
        )

        train_args = args.get_train_args()

        # list_image_info() should already apply ignore filtering:
        # - (3,7) -> (3)
        # - (7) -> removed (skipped)
        # - (11) -> kept
        image_info = list(train_args.list_image_info())
        assert len(image_info) == 2
        assert set(item["class_id"] for item in image_info) == {"3", "11"}

        train_dataset = ImageClassificationDataset(
            dataset_args=train_args,
            transform=_get_transform(),
            image_info=image_info,
        )

        # Mapping should only include non-ignored classes.
        assert set(train_dataset.class_id_to_internal_class_id.keys()) == {3, 11}
        assert set(train_dataset.class_id_to_internal_class_id.values()) == {0, 1}

        # Verify samples contain only non-ignored internal labels and shapes are correct.
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            assert sample["image"].dtype == torch.float32
            assert sample["image"].shape == (3, 64, 128)
            assert sample["classes"].dtype == torch.long
            assert sample["classes"].ndim == 1
            assert torch.all(sample["classes"] <= 1)
            assert (
                sample["classes"].numel() == 1
            )  # After ignore, each sample is single-labeled in this context.


def _get_transform() -> DummyTaskTransform:
    transform_args = IdentityTaskTransformArgs()
    return DummyTaskTransform(transform_args=transform_args)
