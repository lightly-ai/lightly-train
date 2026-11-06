#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightning_fabric import Fabric
from torchvision.transforms import functional as F

from lightly_train._commands import train_task_helpers
from lightly_train._data.image_classification_dataset import (
    ImageClassificationDataset,
    ImageClassificationMulticlassDataArgs,
    ImageClassificationMultilabelDataArgs,
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
        helpers.create_multiclass_image_classification_dataset(
            tmp_path=tmp_path,
            class_names=list(classes.values()),
            num_files_per_class=num_files_per_class,
            height=64,
            width=128,
        )

        args = ImageClassificationMulticlassDataArgs(
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
        helpers.create_multiclass_image_classification_dataset(
            tmp_path=tmp_path,
            class_names=list(classes.values()),
            num_files_per_class=num_files_per_class,
            height=64,
            width=128,
        )

        args = ImageClassificationMulticlassDataArgs(
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
        "csv_label_type,csv_label_column",
        [("id", "class_id"), ("name", "label")],
    )
    @pytest.mark.parametrize("label_delimiter", [",", ";"])
    def test__csv_random_multilabel_encoded_in_path(
        self,
        tmp_path: Path,
        csv_label_type: Literal["id", "name"],
        csv_label_column: str,
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
            csv_image_column="image_path",
            csv_label_column=csv_label_column,
            csv_label_type=csv_label_type,
            label_delimiter=label_delimiter,
        )

        args = ImageClassificationMultilabelDataArgs(
            train=tmp_path / "train.csv",
            val=tmp_path / "val.csv",
            classes=classes,
            csv_image_column="image_path",
            csv_label_column=csv_label_column,
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
            assert torch.equal(
                sample["classes"].sort().values, expected_int_ids.sort().values
            )

    def test__csv_ignore_classes_filters_labels_and_skips_empty(
        self, tmp_path: Path
    ) -> None:
        # Create the dummy dataset.
        num_files_per_class = 2
        classes = {3: "class_3", 7: "class_7", 11: "class_11"}
        helpers.create_multiclass_image_classification_dataset(
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

        args = ImageClassificationMulticlassDataArgs(
            train=csv_path,
            val=csv_path,  # Using the same CSV for the sake of the test.
            classes=classes,
            csv_image_column="image_path",
            csv_label_column="class_id",
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


class TestImageClassificationMmapHash:
    @pytest.mark.parametrize("split", ["train", "val"])
    @pytest.mark.parametrize(
        "args_cls",
        [ImageClassificationMulticlassDataArgs, ImageClassificationMultilabelDataArgs],
    )
    def test_mmap_hash__csv_edits_refresh_reused_labels(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        split: str,
        args_cls: type[
            ImageClassificationMulticlassDataArgs
            | ImageClassificationMultilabelDataArgs
        ],
    ) -> None:
        monkeypatch.setenv("LIGHTLY_TRAIN_MMAP_REUSE_FILE", "1")
        helpers.create_image(path=tmp_path / "image.png", height=8, width=8)
        for name in ("train", "val"):
            (tmp_path / f"{name}.csv").write_text(
                "image_path,label\nimage.png,class_0\n", encoding="utf-8"
            )
        args = args_cls(
            train=tmp_path / "train.csv",
            val=tmp_path / "val.csv",
            classes={0: "class_0", 1: "class_1"},
        )
        if split == "train":
            get_hash = args.train_data_mmap_hash
            get_dataset_args = args.get_train_args
            get_other_hash = args.val_data_mmap_hash
        else:
            get_hash = args.val_data_mmap_hash
            get_dataset_args = args.get_val_args
            get_other_hash = args.train_data_mmap_hash

        fabric = Fabric(accelerator="cpu", devices=1)
        fabric.launch()
        original_hash = get_hash()
        other_hash = get_other_hash()
        with train_task_helpers.get_dataset_temp_mmap_path(
            fabric=fabric, data_hash=original_hash, out=tmp_path / "first"
        ) as original_path:
            image_info = train_task_helpers.get_dataset_mmap_file(
                fabric=fabric,
                items=get_dataset_args().list_image_info(),
                mmap_filepath=original_path,
            )
            assert image_info[0]["class_id"] == "0"

        assert get_hash() == original_hash
        csv_path = tmp_path / f"{split}.csv"
        stat = csv_path.stat()
        csv_path.write_text("image_path,label\nimage.png,class_1\n", encoding="utf-8")
        os.utime(csv_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

        with train_task_helpers.get_dataset_temp_mmap_path(
            fabric=fabric, data_hash=get_hash(), out=tmp_path / "second"
        ) as updated_path:
            updated_image_info = train_task_helpers.get_dataset_mmap_file(
                fabric=fabric,
                items=get_dataset_args().list_image_info(),
                mmap_filepath=updated_path,
            )
            assert updated_image_info[0]["class_id"] == "1"
            assert updated_path != original_path
        assert get_other_hash() == other_hash

    def test_mmap_hash_is_deterministic(self, tmp_path: Path) -> None:
        classes = {0: "class_0", 1: "class_1"}
        helpers.create_multiclass_image_classification_dataset(
            tmp_path=tmp_path,
            class_names=list(classes.values()),
            num_files_per_class=2,
            height=32,
            width=32,
        )
        args = ImageClassificationMulticlassDataArgs(
            train=tmp_path / "train",
            val=tmp_path / "val",
            classes=classes,
        )
        assert args.train_data_mmap_hash() == args.train_data_mmap_hash()
        assert args.val_data_mmap_hash() == args.val_data_mmap_hash()
