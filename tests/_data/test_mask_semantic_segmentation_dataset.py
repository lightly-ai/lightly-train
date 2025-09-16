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

import numpy as np
import pytest
import torch
from torch import Tensor

from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
    MaskSemanticSegmentationDataset,
    MaskSemanticSegmentationDatasetArgs,
    MultiChannelClassInfo,
    SingleChannelClassInfo,
    SplitArgs,
)
from lightly_train._transforms.semantic_segmentation_transform import (
    SemanticSegmentationTransform,
    SemanticSegmentationTransformArgs,
)
from lightly_train._transforms.transform import (
    NormalizeArgs,
    SmallestMaxSizeArgs,
)

from .. import helpers


def _dummy_transform(num_channels: int = 3) -> SemanticSegmentationTransform:
    args = SemanticSegmentationTransformArgs(
        ignore_index=-100,
        image_size=(32, 32),
        channel_drop=None,
        num_channels=num_channels,
        normalize=NormalizeArgs(),
        random_flip=None,
        color_jitter=None,
        scale_jitter=None,
        smallest_max_size=SmallestMaxSizeArgs(max_size=32, prob=1.0),
        random_crop=None,
    )
    args.resolve_auto()
    args.resolve_incompatible()
    return SemanticSegmentationTransform(args)


class TestMaskSemanticSegmentationDataArgs:
    @pytest.mark.parametrize(
        "classes_input, expected_checks",
        [
            # Test with all dict input (overlapping values)
            (
                {
                    0: {"name": "background", "labels": [0, 5]},
                    1: {"name": "vehicle", "labels": [1, 2, 3]},
                },
                {
                    0: ("background", [0, 5]),
                    1: ("vehicle", [1, 2, 3]),
                },
            ),
            # Test with all dict input (non-overlapping values)
            (
                {
                    0: {"name": "background", "labels": [4]},
                    5: {"name": "vehicle", "labels": [1, 2, 3]},
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
                    1: {"name": "vehicle", "labels": [1, 2, 3]},
                },
                {
                    0: ("background", [0]),
                    1: ("vehicle", [1, 2, 3]),
                },
            ),
        ],
    )
    def test_validate_class_labels(
        self,
        classes_input: dict[int, Any],
        expected_checks: dict[int, tuple[str, list[int]]],
        tmp_path: Path,
    ) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        dataset_args = MaskSemanticSegmentationDataArgs(
            train=SplitArgs(images=image_dir, masks=mask_dir),
            val=SplitArgs(images=image_dir, masks=mask_dir),
            classes=classes_input,
        )

        # Check that all inputs were converted to ClassInfo objects
        assert set(dataset_args.classes.keys()) == set(expected_checks.keys()), (
            "Class IDs don't match"
        )

        # Check that names and values match expected
        for class_id, (expected_name, expected_values) in expected_checks.items():
            class_info = dataset_args.classes[class_id]
            assert isinstance(class_info, SingleChannelClassInfo)
            assert class_info.name == expected_name
            assert class_info.labels == set(expected_values)

    @pytest.mark.parametrize(
        "classes_input, expected_checks",
        [
            # Test with single multi-channel values
            (
                {
                    0: {"name": "unlabeled", "labels": [(0, 0, 0), (255, 255, 255)]},
                    1: {"name": "road", "labels": [(128, 128, 128)]},
                },
                {
                    0: ("unlabeled", [(0, 0, 0), (255, 255, 255)]),
                    1: ("road", [(128, 128, 128)]),
                },
            ),
            # Test with multiple multi-channel values
            (
                {
                    0: {"name": "unlabeled", "labels": [(0, 0, 0), (255, 255, 255)]},
                    1: {"name": "road", "labels": [(128, 128, 128), (128, 255, 255)]},
                },
                {
                    0: ("unlabeled", [(0, 0, 0), (255, 255, 255)]),
                    1: ("road", [(128, 128, 128), (128, 255, 255)]),
                },
            ),
        ],
    )
    def test_validate_class_multi_channel_masks(
        self,
        classes_input: dict[int, Any],
        expected_checks: dict[int, tuple[str, list[tuple[int, int, int]]]],
        tmp_path: Path,
    ) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        dataset_args = MaskSemanticSegmentationDataArgs(
            train=SplitArgs(images=image_dir, masks=mask_dir),
            val=SplitArgs(images=image_dir, masks=mask_dir),
            classes=classes_input,
        )

        # Check that all inputs were converted to ClassInfo objects
        assert set(dataset_args.classes.keys()) == set(expected_checks.keys()), (
            "Class IDs don't match"
        )

        # Check that names and labels match expected
        for class_id, (
            expected_name,
            expected_channel_values,
        ) in expected_checks.items():
            class_info = dataset_args.classes[class_id]
            assert isinstance(class_info, MultiChannelClassInfo)
            assert class_info.name == expected_name
            assert class_info.labels == set(expected_channel_values)

    @pytest.mark.parametrize(
        "invalid_classes",
        [
            # Invalid ClassInfo structure
            {0: {"invalid": "structure"}},
            # Invalid values type in ClassInfo
            {0: {"name": "background", "labels": "0"}},
        ],
    )
    def test_validate_class__invalid_inputs(
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

    @pytest.mark.parametrize(
        "mixed_classes",
        [
            # Mixed multi-channel values and class name strings
            {
                0: {"name": "unlabeled", "labels": [(0, 0, 0), (255, 255, 255)]},
                1: "road",
            },
            # Mixed multi-channel values and integer labels
            {
                0: {"name": "unlabeled", "labels": [(0, 0, 0), (255, 255, 255)]},
                1: {"name": "road", "labels": [1, 2]},
            },
        ],
    )
    def test_validate_class__invalid_inputs_mixed_types(
        self, mixed_classes: dict[int, Any], tmp_path: Path
    ) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        # Test that mixed class types raise validation error
        with pytest.raises(
            ValueError,
            match="Invalid class mapping: mixed class types detected.",
        ):
            MaskSemanticSegmentationDataArgs(
                train=SplitArgs(images=image_dir, masks=mask_dir),
                val=SplitArgs(images=image_dir, masks=mask_dir),
                classes=mixed_classes,
            )

    @pytest.mark.parametrize(
        "classes,expected_match",
        [
            (
                {
                    0: {"name": "background", "labels": [0, 1, 2]},
                    5: {"name": "vehicle", "labels": [2, 3, 4]},
                },
                "Invalid class mapping: integer label 2 appears in multiple class definitions. ",
            ),
            (
                {
                    0: {"name": "background", "labels": [0, 1, 2]},
                    1: "vehicle",
                },
                "Invalid class mapping: integer label 1 appears in multiple class definitions. ",
            ),
            (
                {
                    0: {"name": "background", "labels": [(0, 0, 0), (128, 128, 128)]},
                    1: {"name": "vehicle", "labels": [(128, 128, 128), (255, 0, 0)]},
                },
                "Invalid class mapping: tuple label \\(128, 128, 128\\) appears in multiple class definitions. ",
            ),
        ],
    )
    def test_validate_class__mapping_to_multiple_classes(
        self, tmp_path: Path, classes: dict[int, Any], expected_match: str
    ) -> None:
        """Test that overlapping values in different ClassInfo instances raise validation error."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        with pytest.raises(ValueError, match=expected_match):
            MaskSemanticSegmentationDataArgs(
                train=SplitArgs(images=image_dir, masks=mask_dir),
                val=SplitArgs(images=image_dir, masks=mask_dir),
                classes=classes,
            )

    @pytest.mark.parametrize(
        "classes,ignore_classes,expected_included",
        [
            (
                {
                    0: "background",
                    1: {"name": "vehicle", "labels": [1, 2, 3]},
                    4: "person",
                },
                {1, 4},
                {0: "background"},
            ),
            (
                {
                    0: {"name": "background", "labels": [(0, 0, 0)]},
                    1: {"name": "vehicle", "labels": [(128, 128, 128)]},
                    4: {"name": "person", "labels": [(255, 255, 255)]},
                },
                {1, 4},
                {0: "background"},
            ),
            (
                {
                    0: {"name": "background", "labels": [(0, 0, 0), (64, 64, 64)]},
                    1: {"name": "road", "labels": [(128, 128, 128)]},
                    2: {"name": "vehicle", "labels": [(255, 0, 0)]},
                },
                {2},
                {0: "background", 1: "road"},
            ),
        ],
    )
    def test_included_classes(
        self,
        tmp_path: Path,
        classes: dict[int, Any],
        ignore_classes: set[int],
        expected_included: dict[int, str],
    ) -> None:
        """Test that included_classes property works correctly for both labels and channel values."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        dataset_args = MaskSemanticSegmentationDataArgs(
            train=SplitArgs(images=image_dir, masks=mask_dir),
            val=SplitArgs(images=image_dir, masks=mask_dir),
            classes=classes,
            ignore_classes=ignore_classes,
        )

        assert dataset_args.included_classes == expected_included


class TestMaskSemanticSegmentationDatasetArgs:
    def test_mask_dir_or_file__filename_template_string(self, tmp_path: Path) -> None:
        """Test that template string is used as-is when it contains format placeholders."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.jpg", "image1.jpg"]
        mask_filenames = ["image0.png", "image1.png"]

        helpers.create_images(image_dir, files=image_filenames)
        helpers.create_masks(mask_dir, files=mask_filenames, num_classes=1)

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir_or_file="{image_path.parent.parent}/masks/{image_path.stem}.png",
            classes={0: SingleChannelClassInfo(name="background", labels={0})},
            ignore_index=-100,
        )

        filepaths = list(dataset_args.list_image_info())

        assert len(filepaths) == 2
        expected_pairs = [
            {
                "image_filepaths": str(image_dir / "image0.jpg"),
                "mask_filepaths": str(mask_dir / "image0.png"),
            },
            {
                "image_filepaths": str(image_dir / "image1.jpg"),
                "mask_filepaths": str(mask_dir / "image1.png"),
            },
        ]
        # Convert to tuples for comparison since dicts are not hashable
        filepaths_tuples = [
            (fp["image_filepaths"], fp["mask_filepaths"]) for fp in filepaths
        ]
        expected_tuples = [
            (ep["image_filepaths"], ep["mask_filepaths"]) for ep in expected_pairs
        ]
        assert set(filepaths_tuples) == set(expected_tuples)

    def test_mask_dir_or_file__directory_path(self, tmp_path: Path) -> None:
        """Test that directory path gets converted to template string when no format placeholders exist."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.jpg", "image1.jpg"]
        mask_filenames = ["image0.png", "image1.png"]

        helpers.create_images(image_dir, files=image_filenames)
        helpers.create_masks(mask_dir, files=mask_filenames, num_classes=1)

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir_or_file=str(mask_dir),
            classes={0: SingleChannelClassInfo(name="background", labels={0})},
            ignore_index=-100,
        )

        filepaths = list(dataset_args.list_image_info())

        assert len(filepaths) == 2
        expected_pairs = [
            {
                "image_filepaths": str(image_dir / "image0.jpg"),
                "mask_filepaths": str(mask_dir / "image0.png"),
            },
            {
                "image_filepaths": str(image_dir / "image1.jpg"),
                "mask_filepaths": str(mask_dir / "image1.png"),
            },
        ]
        # Convert to tuples for comparison since dicts are not hashable
        filepaths_tuples = [
            (fp["image_filepaths"], fp["mask_filepaths"]) for fp in filepaths
        ]
        expected_tuples = [
            (ep["image_filepaths"], ep["mask_filepaths"]) for ep in expected_pairs
        ]
        assert set(filepaths_tuples) == set(expected_tuples)


class TestMaskSemanticSegmentationDataset:
    @pytest.mark.parametrize(
        "num_classes, num_channels, expected_mask_dtype, ignore_index",
        [
            (5, 3, torch.long, -100),
            (5, 4, torch.long, -100),
            (500, 3, torch.long, -100),
        ],
    )
    def test__getitem__integer_masks(
        self,
        num_classes: int,
        num_channels: int,
        expected_mask_dtype: torch.dtype,
        tmp_path: Path,
        ignore_index: int,
    ) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.png", "image1.png"]
        mask_filenames = ["image0.png", "image1.png"]

        helpers.create_images(
            image_dir,
            files=image_filenames,
            num_channels=num_channels,
            mode="RGB" if num_channels == 3 else "RGBA",
        )
        helpers.create_masks(mask_dir, files=mask_filenames, num_classes=num_classes)

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir_or_file=str(mask_dir),
            classes={
                i: SingleChannelClassInfo(name=f"class_{i}", labels={i})
                for i in range(num_classes)
            },
            ignore_index=ignore_index,
        )
        transform = _dummy_transform(num_channels=num_channels)
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_info=list(dataset_args.list_image_info()),
            transform=transform,
        )

        assert len(dataset) == 2
        for item in dataset:  # type: ignore[attr-defined]
            assert isinstance(item["image"], Tensor)
            assert item["image"].shape == (num_channels, 32, 32)
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
            str(image_dir / "image0.png"),
            str(image_dir / "image1.png"),
        ]

    @pytest.mark.parametrize(
        "num_classes, num_channels, expected_mask_dtype, ignore_index",
        [
            (5, 3, torch.long, -100),
            (5, 4, torch.long, -100),
            (500, 3, torch.long, -100),
        ],
    )
    def test__getitem__multi_channel_masks(
        self,
        num_classes: int,
        num_channels: int,
        expected_mask_dtype: torch.dtype,
        tmp_path: Path,
        ignore_index: int,
    ) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.png", "image1.png"]
        mask_filenames = ["image0.png", "image1.png"]

        helpers.create_images(
            image_dir,
            files=image_filenames,
            num_channels=num_channels,
            mode="RGB" if num_channels == 3 else "RGBA",
        )

        channel_values_set: set[tuple[int, ...]] = set()
        while len(channel_values_set) < num_classes:
            channel_value = tuple(int(x) for x in np.random.randint(0, 256, size=3))
            channel_values_set.add(channel_value)
        channel_values = list(channel_values_set)
        helpers.create_multi_channel_masks(
            mask_dir,
            files=mask_filenames,
            values=channel_values,
        )

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir_or_file=str(mask_dir),
            classes={
                i: MultiChannelClassInfo(name=f"class_{i}", labels={channel_values[i]})
                for i in range(num_classes)
            },
            ignore_index=ignore_index,
        )
        transform = _dummy_transform(num_channels=num_channels)
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_info=list(dataset_args.list_image_info()),
            transform=transform,
        )

        assert len(dataset) == 2
        for item in dataset:  # type: ignore[attr-defined]
            assert isinstance(item["image"], Tensor)
            assert item["image"].shape == (num_channels, 32, 32)
            assert item["image"].dtype == torch.float32
            assert isinstance(item["mask"], Tensor)
            assert item["mask"].shape == (32, 32)
            assert item["mask"].dtype == expected_mask_dtype

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
            str(image_dir / "image0.png"),
            str(image_dir / "image1.png"),
        ]

    def test__getitem__shape_mismatch_error(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        # Create image with one size
        helpers.create_image(image_dir / "image0.jpg", height=64, width=64)

        # Create mask with different size
        helpers.create_mask(
            mask_dir / "image0.png", height=128, width=128, num_classes=1
        )

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir_or_file=str(mask_dir),
            classes={0: SingleChannelClassInfo(name="class_0", labels={0})},
            ignore_index=-100,
        )
        transform = _dummy_transform(num_channels=3)
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_info=list(dataset_args.list_image_info()),
            transform=transform,
        )

        with pytest.raises(
            ValueError,
            match="Shape mismatch: image \\(height, width\\) is .* while mask \\(height, width\\) is .*",
        ):
            dataset[0]

    def test__getitem__multi_channel_mask_with_label_classes_error(
        self, tmp_path: Path
    ) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"

        # Create a regular image
        helpers.create_image(image_dir / "image0.jpg", height=64, width=64)

        # Create a multi-channel mask (3 channels)
        helpers.create_multi_channel_mask(mask_dir / "image0.png", height=64, width=64)

        # Use SingleChannelClassInfo (instead of MultiChannelClassInfo)
        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir_or_file=str(mask_dir),
            classes={0: SingleChannelClassInfo(name="class_0", labels={0})},
            ignore_index=-100,
        )
        transform = _dummy_transform(num_channels=3)
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_info=list(dataset_args.list_image_info()),
            transform=transform,
        )

        with pytest.raises(
            ValueError,
            match="Expected tuple labels specified in `classes` for multi-channel masks",
        ):
            dataset[0]

    def test_get_class_mapping__labels(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.jpg"]
        mask_filenames = ["image0.png"]

        helpers.create_images(image_dir, files=image_filenames)
        helpers.create_masks(mask_dir, files=mask_filenames, num_classes=5)

        classes: dict[int, SingleChannelClassInfo | MultiChannelClassInfo] = {
            0: SingleChannelClassInfo(name="background", labels={0, 5}),
            1: SingleChannelClassInfo(name="vehicle", labels={1, 2, 3}),
        }
        expected_mapping = {0: 0, 1: 1}

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir_or_file=str(mask_dir),
            classes=classes,
            ignore_index=-100,
        )
        transform = _dummy_transform()
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_info=list(dataset_args.list_image_info()),
            transform=transform,
        )

        assert dataset.class_mapping == expected_mapping

    def test_get_class_mapping__ignore_classes__labels(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.jpg"]
        mask_filenames = ["image0.png"]

        helpers.create_images(image_dir, files=image_filenames)
        helpers.create_masks(mask_dir, files=mask_filenames, num_classes=5)

        classes: dict[int, SingleChannelClassInfo | MultiChannelClassInfo] = {
            1: SingleChannelClassInfo(name="vehicle", labels={1, 2, 3}),
            4: SingleChannelClassInfo(name="ignore_me", labels={4}),
            5: SingleChannelClassInfo(name="person", labels={5}),
        }
        ignore_classes = {4}
        expected_mapping = {1: 0, 5: 1}

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir_or_file=str(mask_dir),
            classes=classes,
            ignore_classes=ignore_classes,
            ignore_index=-100,
        )
        transform = _dummy_transform()
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_info=list(dataset_args.list_image_info()),
            transform=transform,
        )

        assert dataset.class_mapping == expected_mapping

    def test_get_class_mapping__multi_channel_masks(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.jpg"]
        mask_filenames = ["image0.png"]

        helpers.create_images(image_dir, files=image_filenames)
        helpers.create_multi_channel_masks(
            mask_dir, files=mask_filenames, values=[(0, 0, 0), (128, 128, 128)]
        )

        classes: dict[int, SingleChannelClassInfo | MultiChannelClassInfo] = {
            0: MultiChannelClassInfo(name="background", labels={(0, 0, 0)}),
            1: MultiChannelClassInfo(name="road", labels={(128, 128, 128)}),
        }
        expected_mapping = {0: 0, 1: 1}

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir_or_file=str(mask_dir),
            classes=classes,
            ignore_index=-100,
        )
        transform = _dummy_transform(num_channels=3)
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_info=list(dataset_args.list_image_info()),
            transform=transform,
        )

        assert dataset.class_mapping == expected_mapping

    def test_get_class_mapping__ignore_classes__multi_channel_masks(
        self, tmp_path: Path
    ) -> None:
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_filenames = ["image0.jpg"]
        mask_filenames = ["image0.png"]

        helpers.create_images(image_dir, files=image_filenames)
        palette = [(0, 0, 0), (128, 128, 128), (255, 0, 0)]
        helpers.create_multi_channel_masks(
            mask_dir, files=mask_filenames, values=palette
        )

        classes: dict[int, SingleChannelClassInfo | MultiChannelClassInfo] = {
            1: MultiChannelClassInfo(name="road", labels={(128, 128, 128)}),
            4: MultiChannelClassInfo(name="ignore_me", labels={(255, 0, 0)}),
            5: MultiChannelClassInfo(name="person", labels={(0, 0, 0)}),
        }
        ignore_classes = {4}
        expected_mapping = {1: 0, 5: 1}

        dataset_args = MaskSemanticSegmentationDatasetArgs(
            image_dir=image_dir,
            mask_dir_or_file=str(mask_dir),
            classes=classes,
            ignore_classes=ignore_classes,
            ignore_index=-100,
        )
        transform = _dummy_transform(num_channels=3)
        dataset = MaskSemanticSegmentationDataset(
            dataset_args=dataset_args,
            image_info=list(dataset_args.list_image_info()),
            transform=transform,
        )

        assert dataset.class_mapping == expected_mapping
