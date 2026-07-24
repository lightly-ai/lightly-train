#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import pytest

from lightly_train._data import data_helpers
from lightly_train._data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from lightly_train._data.coco_object_detection_dataset import (
    SplitArgs as COCOObjectDetectionSplitArgs,
)
from lightly_train._data.image_classification_dataset import (
    ImageClassificationMulticlassDataArgs,
    ImageClassificationMultilabelDataArgs,
)
from lightly_train._data.instance_segmentation_dataset import (
    COCOInstanceSegmentationDataArgs,
    YOLOInstanceSegmentationDataArgs,
)
from lightly_train._data.instance_segmentation_dataset import (
    COCOSplitArgs as COCOInstanceSegmentationSplitArgs,
)
from lightly_train._data.mask_panoptic_segmentation_dataset import (
    MaskPanopticSegmentationDataArgs,
)
from lightly_train._data.mask_panoptic_segmentation_dataset import (
    SplitArgs as MaskPanopticSegmentationSplitArgs,
)
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
)
from lightly_train._data.mask_semantic_segmentation_dataset import (
    SplitArgs as MaskSemanticSegmentationSplitArgs,
)
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._data.yolo_oriented_object_detection_dataset import (
    YOLOOrientedObjectDetectionDataArgs,
)


class TestImageClassificationMulticlassDataArgs:
    def test_resolves_dict_paths_relative_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data_args = ImageClassificationMulticlassDataArgs(
            train="train",
            val="val",
            test="test",
            classes={0: "class_a"},
        )

        data_helpers.resolve_data_paths(data_args)

        assert data_args.train == (tmp_path / "train").resolve()
        assert data_args.val == (tmp_path / "val").resolve()
        assert data_args.test == (tmp_path / "test").resolve()

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "configs" / "data.yaml"
        data_args = ImageClassificationMulticlassDataArgs(
            train="train",
            val="val",
            test="test",
            classes={0: "class_a"},
        )
        data_args.data_config_file = data_yaml

        data_helpers.resolve_data_paths(data_args)

        assert data_args.train == (tmp_path / "configs" / "train").resolve()
        assert data_args.val == (tmp_path / "configs" / "val").resolve()
        assert data_args.test == (tmp_path / "configs" / "test").resolve()


class TestImageClassificationMultilabelDataArgs:
    def test_resolves_dict_paths_relative_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data_args = ImageClassificationMultilabelDataArgs(
            train="train.csv",
            val="val.csv",
            test="test.csv",
            classes={0: "class_a"},
        )

        data_helpers.resolve_data_paths(data_args)

        assert data_args.train == (tmp_path / "train.csv").resolve()
        assert data_args.val == (tmp_path / "val.csv").resolve()
        assert data_args.test == (tmp_path / "test.csv").resolve()

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "configs" / "data.yaml"
        data_args = ImageClassificationMultilabelDataArgs(
            train="train.csv",
            val="val.csv",
            test="test.csv",
            classes={0: "class_a"},
        )
        data_args.data_config_file = data_yaml

        data_helpers.resolve_data_paths(data_args)

        assert data_args.train == (tmp_path / "configs" / "train.csv").resolve()
        assert data_args.val == (tmp_path / "configs" / "val.csv").resolve()
        assert data_args.test == (tmp_path / "configs" / "test.csv").resolve()


class TestYOLOObjectDetectionDataArgs:
    def test_resolves_dict_paths_relative_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data_args = YOLOObjectDetectionDataArgs(
            path="dataset",
            train="images/train",
            val="images/val",
            names={0: "class_a"},
        )

        data_helpers.resolve_data_paths(data_args)

        assert data_args.path == (tmp_path / "dataset").resolve()
        assert data_args.train == Path("images/train")
        assert data_args.val == Path("images/val")

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "configs" / "data.yaml"
        data_args = YOLOObjectDetectionDataArgs(
            path="dataset",
            train="images/train",
            val="images/val",
            names={0: "class_a"},
        )
        data_args.data_config_file = data_yaml

        data_helpers.resolve_data_paths(data_args)

        assert data_args.path == (tmp_path / "configs" / "dataset").resolve()
        assert data_args.train == Path("images/train")
        assert data_args.val == Path("images/val")


class TestCOCOObjectDetectionDataArgs:
    def test_resolves_dict_paths_relative_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data_args = COCOObjectDetectionDataArgs(
            train=COCOObjectDetectionSplitArgs(
                annotations="annotations/train.json",
                images="images/train",
            ),
            val=COCOObjectDetectionSplitArgs(
                annotations="annotations/val.json",
                images=tmp_path / "absolute/val_images",
            ),
        )

        data_helpers.resolve_data_paths(data_args)

        assert (
            data_args.train.annotations
            == (tmp_path / "annotations/train.json").resolve()
        )
        assert data_args.train.images == Path("images/train")
        assert (
            data_args.val.annotations == (tmp_path / "annotations/val.json").resolve()
        )
        assert data_args.val.images == (tmp_path / "absolute/val_images").resolve()

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "configs" / "data.yaml"
        data_args = COCOObjectDetectionDataArgs(
            train=COCOObjectDetectionSplitArgs(
                annotations="annotations/train.json",
                images="images/train",
            ),
            val=COCOObjectDetectionSplitArgs(
                annotations="annotations/val.json",
                images=tmp_path / "configs" / "absolute/val_images",
            ),
        )
        data_args.data_config_file = data_yaml

        data_helpers.resolve_data_paths(data_args)

        assert (
            data_args.train.annotations
            == (tmp_path / "configs" / "annotations/train.json").resolve()
        )
        assert data_args.train.images == Path("images/train")
        assert (
            data_args.val.annotations
            == (tmp_path / "configs" / "annotations/val.json").resolve()
        )
        assert (
            data_args.val.images
            == (tmp_path / "configs" / "absolute/val_images").resolve()
        )


class TestYOLOOrientedObjectDetectionDataArgs:
    def test_resolves_dict_paths_relative_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data_args = YOLOOrientedObjectDetectionDataArgs(
            path="dataset",
            train="images/train",
            val="images/val",
            names={0: "class_a"},
        )

        data_helpers.resolve_data_paths(data_args)

        assert data_args.path == (tmp_path / "dataset").resolve()
        assert data_args.train == Path("images/train")
        assert data_args.val == Path("images/val")

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "configs" / "data.yaml"
        data_args = YOLOOrientedObjectDetectionDataArgs(
            path="dataset",
            train="images/train",
            val="images/val",
            names={0: "class_a"},
        )
        data_args.data_config_file = data_yaml

        data_helpers.resolve_data_paths(data_args)

        assert data_args.path == (tmp_path / "configs" / "dataset").resolve()
        assert data_args.train == Path("images/train")
        assert data_args.val == Path("images/val")


class TestYOLOInstanceSegmentationDataArgs:
    def test_resolves_dict_paths_relative_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data_args = YOLOInstanceSegmentationDataArgs(
            path="dataset",
            train="images/train",
            val="images/val",
            names={0: "class_a"},
        )

        data_helpers.resolve_data_paths(data_args)

        assert data_args.path == (tmp_path / "dataset").resolve()
        assert data_args.train == Path("images/train")
        assert data_args.val == Path("images/val")

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "configs" / "data.yaml"
        data_args = YOLOInstanceSegmentationDataArgs(
            path="dataset",
            train="images/train",
            val="images/val",
            names={0: "class_a"},
        )
        data_args.data_config_file = data_yaml

        data_helpers.resolve_data_paths(data_args)

        assert data_args.path == (tmp_path / "configs" / "dataset").resolve()
        assert data_args.train == Path("images/train")
        assert data_args.val == Path("images/val")


class TestCOCOInstanceSegmentationDataArgs:
    def test_resolves_dict_paths_relative_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data_args = COCOInstanceSegmentationDataArgs(
            train=COCOInstanceSegmentationSplitArgs(
                annotations="annotations/train.json",
                images="images/train",
            ),
            val=COCOInstanceSegmentationSplitArgs(
                annotations="annotations/val.json",
                images=tmp_path / "absolute/val_images",
            ),
        )

        data_helpers.resolve_data_paths(data_args)

        assert (
            data_args.train.annotations
            == (tmp_path / "annotations/train.json").resolve()
        )
        assert data_args.train.images == Path("images/train")
        assert (
            data_args.val.annotations == (tmp_path / "annotations/val.json").resolve()
        )
        assert data_args.val.images == (tmp_path / "absolute/val_images").resolve()

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "configs" / "data.yaml"
        data_args = COCOInstanceSegmentationDataArgs(
            train=COCOInstanceSegmentationSplitArgs(
                annotations="annotations/train.json",
                images="images/train",
            ),
            val=COCOInstanceSegmentationSplitArgs(
                annotations="annotations/val.json",
                images=tmp_path / "configs" / "absolute/val_images",
            ),
        )
        data_args.data_config_file = data_yaml

        data_helpers.resolve_data_paths(data_args)

        assert (
            data_args.train.annotations
            == (tmp_path / "configs" / "annotations/train.json").resolve()
        )
        assert data_args.train.images == Path("images/train")
        assert (
            data_args.val.annotations
            == (tmp_path / "configs" / "annotations/val.json").resolve()
        )
        assert (
            data_args.val.images
            == (tmp_path / "configs" / "absolute/val_images").resolve()
        )


class TestMaskSemanticSegmentationDataArgs:
    def test_resolves_dict_paths_relative_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "classes.json").write_text('{"0": "background", "1": "car"}')
        data_args = MaskSemanticSegmentationDataArgs(
            train=MaskSemanticSegmentationSplitArgs(
                images="images/train", masks="masks/train"
            ),
            val=MaskSemanticSegmentationSplitArgs(
                images="images/val", masks="masks/val"
            ),
            classes="classes.json",
        )

        data_helpers.resolve_data_paths(data_args)

        assert data_args.train.images == (tmp_path / "images/train").resolve()
        assert data_args.train.masks == (tmp_path / "masks/train").resolve()
        assert data_args.val.images == (tmp_path / "images/val").resolve()
        assert data_args.val.masks == (tmp_path / "masks/val").resolve()
        assert isinstance(data_args.classes, dict)
        assert data_args.classes[0].name == "background"
        assert data_args.classes[1].name == "car"

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "configs" / "data.yaml"
        data_yaml.parent.mkdir()
        (tmp_path / "configs" / "classes.json").write_text(
            '{"0": "background", "1": "car"}'
        )
        data_args = MaskSemanticSegmentationDataArgs(
            train=MaskSemanticSegmentationSplitArgs(
                images="images/train", masks="masks/train"
            ),
            val=MaskSemanticSegmentationSplitArgs(
                images="images/val", masks="masks/val"
            ),
            classes="classes.json",
        )
        data_args.data_config_file = data_yaml

        data_helpers.resolve_data_paths(data_args)

        assert (
            data_args.train.images == (tmp_path / "configs" / "images/train").resolve()
        )
        assert data_args.train.masks == (tmp_path / "configs" / "masks/train").resolve()
        assert data_args.val.images == (tmp_path / "configs" / "images/val").resolve()
        assert data_args.val.masks == (tmp_path / "configs" / "masks/val").resolve()
        assert isinstance(data_args.classes, dict)
        assert data_args.classes[0].name == "background"
        assert data_args.classes[1].name == "car"


class TestMaskPanopticSegmentationDataArgs:
    def test_resolves_dict_paths_relative_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data_args = MaskPanopticSegmentationDataArgs(
            train=MaskPanopticSegmentationSplitArgs(
                images="images/train",
                masks="masks/train",
                annotations="annotations/train.json",
            ),
            val=MaskPanopticSegmentationSplitArgs(
                images="images/val",
                masks="masks/val",
                annotations="annotations/val.json",
            ),
        )

        data_helpers.resolve_data_paths(data_args)

        assert data_args.train.images == (tmp_path / "images/train").resolve()
        assert data_args.train.masks == (tmp_path / "masks/train").resolve()
        assert (
            data_args.train.annotations
            == (tmp_path / "annotations/train.json").resolve()
        )
        assert data_args.val.images == (tmp_path / "images/val").resolve()
        assert data_args.val.masks == (tmp_path / "masks/val").resolve()
        assert (
            data_args.val.annotations == (tmp_path / "annotations/val.json").resolve()
        )

    def test_resolves_paths_relative_to_data_config_file(self, tmp_path: Path) -> None:
        data_yaml = tmp_path / "configs" / "data.yaml"
        data_args = MaskPanopticSegmentationDataArgs(
            train=MaskPanopticSegmentationSplitArgs(
                images="images/train",
                masks="masks/train",
                annotations="annotations/train.json",
            ),
            val=MaskPanopticSegmentationSplitArgs(
                images="images/val",
                masks="masks/val",
                annotations="annotations/val.json",
            ),
        )
        data_args.data_config_file = data_yaml

        data_helpers.resolve_data_paths(data_args)

        assert (
            data_args.train.images == (tmp_path / "configs" / "images/train").resolve()
        )
        assert data_args.train.masks == (tmp_path / "configs" / "masks/train").resolve()
        assert (
            data_args.train.annotations
            == (tmp_path / "configs" / "annotations/train.json").resolve()
        )
        assert data_args.val.images == (tmp_path / "configs" / "images/val").resolve()
        assert data_args.val.masks == (tmp_path / "configs" / "masks/val").resolve()
        assert (
            data_args.val.annotations
            == (tmp_path / "configs" / "annotations/val.json").resolve()
        )
