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

import albumentations as A
import pytest
import torch
from torch import Tensor

from lightly_train._data.mask_semantic_segmentation_dataset import (
    ClassInfo,
    MaskSemanticSegmentationDataArgs,
    MaskSemanticSegmentationDataset,
    MaskSemanticSegmentationDatasetArgs,
    SplitArgs,
)
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
    TaskTransformInput,
    TaskTransformOutput,
)

from .. import helpers


class DummyTransform(TaskTransform):
    transform_args_cls = TaskTransformArgs

    def __init__(self, transform_args: TaskTransformArgs):
        super().__init__(transform_args=transform_args)
        self.transform = A.Compose(
            [
                A.Resize(32, 32),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                A.pytorch.transforms.ToTensorV2(),
            ]
        )

    def __call__(self, input: TaskTransformInput) -> TaskTransformOutput:
        output: TaskTransformOutput = self.transform(**input)
        return output


class TestMaskSemanticSegmentationDataArgs:
    @pytest.mark.parametrize(
        "classes_input, expected_checks",
        [
            # Test with all dict input (overlapping values)
            (
                {
                    0: {"name": "background", "values": [0, 5]},
                    1: {"name": "vehicle", "values": [1, 2, 3]},
                },
                {
                    0: ("background", [0, 5]),
                    1: ("vehicle", [1, 2, 3]),
                },
            ),
            # Test with all dict input (non-overlapping values)
            (
                {
                    0: {"name": "background", "values": [4]},
                    5: {"name": "vehicle", "values": [1, 2, 3]},
                },
                {
                    0: ("background", [4]),
                    5: ("vehicle", [1, 2, 3]),
                },
            ),
            # Test with all string input
            (
                {
                    0: "background",
                    1: "vehicle",
                },
                {
                    0: ("background", [0]),
                    1: ("vehicle", [1]),
                },
            ),
            # Test with mixed input
            (
                {
                    0: "background",
                    1: {"name": "vehicle", "values": [1, 2, 3]},
                },
                {
                    0: ("background", [0]),
                    1: ("vehicle", [1, 2, 3]),
                },
            ),
        ],
    )
    def test_validate_classes(
        self,
        classes_input: dict[int, str | dict[str, str | list[int]]],
        expected_checks: dict[int, tuple[str, list[int]]],
        tmp_path: Path,
    ) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        dataset_args = MaskSemanticSegmentationDataArgs(
            train=SplitArgs(images=image_dir, masks=mask_dir),
            val=SplitArgs(images=image_dir, masks=mask_dir),
            classes=classes_input,  # type: ignore[arg-type]
        )

        # Check that all inputs were converted to ClassInfo objects
        assert set(dataset_args.classes.keys()) == set(expected_checks.keys()), (
            "Class IDs don't match"
        )

        # Check that names and values match expected
        for class_id, (expected_name, expected_values) in expected_checks.items():
            class_info = dataset_args.classes[class_id]
            assert isinstance(class_info, ClassInfo)
            assert class_info.name == expected_name
            assert class_info.values == set(expected_values)

    @pytest.mark.parametrize(
        "invalid_classes",
        [
            # Invalid ClassInfo structure
            {0: {"invalid": "structure"}},
            # Invalid values type in ClassInfo
            {0: {"name": "background", "values": "0"}},
        ],
    )
    def test_validate_classes__invalid_input(
        self, invalid_classes: dict[int, Any], tmp_path: Path
    ) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        # Test that invalid input raises validation error
        with pytest.raises(ValueError):
            MaskSemanticSegmentationDataArgs(
                train=SplitArgs(images=image_dir, masks=mask_dir),
                val=SplitArgs(images=image_dir, masks=mask_dir),
                classes=invalid_classes,
            )

    def test_validate_classes__mapping_to_multiple_class_labels(
        self, tmp_path: Path
    ) -> None:
        """Test that overlapping values in different ClassInfo instances raise validation error."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        # Test that overlapping values in ClassInfo instances raise error
        classes_with_all_mappings = {
            0: {"name": "background", "values": [0, 1, 2]},
            5: {
                "name": "vehicle",
                "values": [2, 3, 4],
            },  # Value 2 is mapped to both background and vehicle
        }

        with pytest.raises(
            ValueError,
            match="Invalid class mapping: Class 2 appears in multiple class definitions. ",
        ):
            MaskSemanticSegmentationDataArgs(
                train=SplitArgs(images=image_dir, masks=mask_dir),
                val=SplitArgs(images=image_dir, masks=mask_dir),
                classes=classes_with_all_mappings,  # type: ignore[arg-type]
            )

        # Test that overlapping values in ClassInfo instances raise error
        classes_with_partial_mappings = {
            0: {"name": "background", "values": [0, 1, 2]},
            1: "vehicle",  # Implicitly maps to {1}, Value 1 is mapped to both background and vehicle
        }

        with pytest.raises(
            ValueError,
            match="Invalid class mapping: Class 1 appears in multiple class definitions. ",
        ):
            MaskSemanticSegmentationDataArgs(
                train=SplitArgs(images=image_dir, masks=mask_dir),
                val=SplitArgs(images=image_dir, masks=mask_dir),
                classes=classes_with_partial_mappings,  # type: ignore[arg-type]
            )

    def test_included_classes(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        classes = {
            0: "background",
            1: {"name": "vehicle", "values": [1, 2, 3]},
            4: "person",
        }
        ignore_classes = {1, 4}  # Ignore vehicle class key and person
        expected_included = {0: "background"}

        dataset_args = MaskSemanticSegmentationDataArgs(
            train=SplitArgs(images=image_dir, masks=mask_dir),
            val=SplitArgs(images=image_dir, masks=mask_dir),
            classes=classes,  # type: ignore[arg-type]
            ignore_classes=ignore_classes,
        )

        assert dataset_args.included_classes == expected_included


class TestMaskSemanticSegmentationDataset:
    @pytest.mark.parametrize(
        "num_classes, expected_mask_dtype, ignore_index",
        [
            (5, torch.long, -100),
            (500, torch.long, -100),
        ],
    )
    def test__getitem__(
        self,
        num_classes: int,
        expected_mask_dtype: torch.dtype,
        tmp_path: Path,
        ignore_index: int,
    ) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.jpg", "image1.jpg"]
        mask_filenames = ["image0.png", "image1.png"]
        helpers.create_images(image_dir, files=image_filenames)
        helpers.create_masks(mask_dir, files=mask_filenames, num_classes=num_classes)

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir=mask_dir,
            classes={
                0: ClassInfo(name="background", values={0}),
                1: ClassInfo(name="object", values={1}),
            },
            ignore_index=ignore_index,
        )
        transform = DummyTransform(transform_args=TaskTransformArgs())
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_filenames=list(dataset_args.list_image_filenames()),
            transform=transform,
        )

        assert len(dataset) == 2
        for item in dataset:  # type: ignore[attr-defined]
            assert isinstance(item["image"], Tensor)
            assert item["image"].shape == (3, 32, 32)
            assert item["image"].dtype == torch.float32
            assert isinstance(item["mask"], Tensor)
            assert item["mask"].shape == (32, 32)
            assert item["mask"].dtype == expected_mask_dtype

            # Need conversion to int because min/max is not implemented for uint16.
            # All valid (non-ignored) pixels should be between 0 and num_classes-1
            mask = item["mask"]
            valid_pixels = mask != ignore_index
            if valid_pixels.any():
                assert mask[valid_pixels].min() >= 0
                assert mask[valid_pixels].max() < num_classes

            # Ignored pixels should exactly match ignore_index
            ignored_pixels = mask == ignore_index
            assert (ignored_pixels.sum() + valid_pixels.sum()) == mask.numel()
        assert sorted(item["image_path"] for item in dataset) == [  # type: ignore[attr-defined]
            str(image_dir / "image0.jpg"),
            str(image_dir / "image1.jpg"),
        ]

    def test_get_class_mapping(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.jpg"]
        mask_filenames = ["image0.png"]
        helpers.create_images(image_dir, files=image_filenames)
        helpers.create_masks(mask_dir, files=mask_filenames, num_classes=5)

        classes = {
            0: ClassInfo(name="background", values={0, 5}),
            1: ClassInfo(name="vehicle", values={1, 2, 3}),
        }
        expected_mapping = {0: 0, 1: 1}

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir=mask_dir,
            classes=classes,
            check_empty_targets=False,
            ignore_index=-100,
        )
        transform = DummyTransform(transform_args=TaskTransformArgs())
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_filenames=list(dataset_args.list_image_filenames()),
            transform=transform,
        )

        assert dataset.class_mapping == expected_mapping

    def test_get_class_mapping__ignore_classes(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.jpg"]
        mask_filenames = ["image0.png"]
        helpers.create_images(image_dir, files=image_filenames)
        helpers.create_masks(mask_dir, files=mask_filenames, num_classes=5)

        classes = {
            1: ClassInfo(name="vehicle", values={1, 2, 3}),
            4: ClassInfo(name="ignore_me", values={4}),
            5: ClassInfo(name="person", values={5}),
        }
        ignore_classes = {4}
        expected_mapping = {1: 0, 5: 1}

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir=mask_dir,
            classes=classes,
            ignore_classes=ignore_classes,
            check_empty_targets=False,
            ignore_index=-100,
        )
        transform = DummyTransform(transform_args=TaskTransformArgs())
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_filenames=list(dataset_args.list_image_filenames()),
            transform=transform,
        )

        assert dataset.class_mapping == expected_mapping
