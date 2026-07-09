#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

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
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._data.yolo_oriented_object_detection_dataset import (
    YOLOOrientedObjectDetectionDataArgs,
)


def _with_data_config_file(
    data_args: TaskDataArgs, data_config_file: Path
) -> TaskDataArgs:
    data_args.data_config_file = data_config_file
    return data_args


def _assert_image_classification_paths(data_args: TaskDataArgs, base_dir: Path) -> None:
    assert isinstance(data_args, ImageClassificationMulticlassDataArgs)
    assert data_args.train == (base_dir / "train").resolve()
    assert data_args.val == (base_dir / "val").resolve()
    assert data_args.test == (base_dir / "test").resolve()


def _assert_image_classification_multilabel_paths(
    data_args: TaskDataArgs, base_dir: Path
) -> None:
    assert isinstance(data_args, ImageClassificationMultilabelDataArgs)
    assert data_args.train == (base_dir / "train.csv").resolve()
    assert data_args.val == (base_dir / "val.csv").resolve()
    assert data_args.test == (base_dir / "test.csv").resolve()


def _assert_yolo_object_detection_paths(
    data_args: TaskDataArgs, base_dir: Path
) -> None:
    assert isinstance(data_args, YOLOObjectDetectionDataArgs)
    assert data_args.path == (base_dir / "dataset").resolve()
    assert data_args.train == Path("images/train")
    assert data_args.val == Path("images/val")


def _assert_coco_object_detection_paths(
    data_args: TaskDataArgs, base_dir: Path
) -> None:
    assert isinstance(data_args, COCOObjectDetectionDataArgs)
    assert (
        data_args.train.annotations == (base_dir / "annotations/train.json").resolve()
    )
    assert data_args.val.annotations == (base_dir / "annotations/val.json").resolve()
    assert data_args.train.images == Path("images/train")
    assert data_args.val.images == (base_dir / "absolute/val_images").resolve()


def _assert_yolo_oriented_object_detection_paths(
    data_args: TaskDataArgs, base_dir: Path
) -> None:
    assert isinstance(data_args, YOLOOrientedObjectDetectionDataArgs)
    assert data_args.path == (base_dir / "dataset").resolve()
    assert data_args.train == Path("images/train")
    assert data_args.val == Path("images/val")


def _assert_yolo_instance_segmentation_paths(
    data_args: TaskDataArgs, base_dir: Path
) -> None:
    assert isinstance(data_args, YOLOInstanceSegmentationDataArgs)
    assert data_args.path == (base_dir / "dataset").resolve()
    assert data_args.train == Path("images/train")
    assert data_args.val == Path("images/val")


def _assert_coco_instance_segmentation_paths(
    data_args: TaskDataArgs, base_dir: Path
) -> None:
    assert isinstance(data_args, COCOInstanceSegmentationDataArgs)
    assert (
        data_args.train.annotations == (base_dir / "annotations/train.json").resolve()
    )
    assert data_args.val.annotations == (base_dir / "annotations/val.json").resolve()
    assert data_args.train.images == Path("images/train")
    assert data_args.val.images == (base_dir / "absolute/val_images").resolve()


def _assert_mask_semantic_segmentation_paths(
    data_args: TaskDataArgs, base_dir: Path
) -> None:
    assert isinstance(data_args, MaskSemanticSegmentationDataArgs)
    assert data_args.train.images == (base_dir / "images/train").resolve()
    assert data_args.train.masks == (base_dir / "masks/train").resolve()
    assert data_args.val.images == (base_dir / "images/val").resolve()
    assert data_args.val.masks == (base_dir / "masks/val").resolve()
    assert not isinstance(data_args.classes, Path)
    assert data_args.classes[0].name == "background"
    assert data_args.classes[1].name == "car"


def _assert_mask_panoptic_segmentation_paths(
    data_args: TaskDataArgs, base_dir: Path
) -> None:
    assert isinstance(data_args, MaskPanopticSegmentationDataArgs)
    assert data_args.train.images == (base_dir / "images/train").resolve()
    assert data_args.train.masks == (base_dir / "masks/train").resolve()
    assert (
        data_args.train.annotations == (base_dir / "annotations/train.json").resolve()
    )
    assert data_args.val.images == (base_dir / "images/val").resolve()
    assert data_args.val.masks == (base_dir / "masks/val").resolve()
    assert data_args.val.annotations == (base_dir / "annotations/val.json").resolve()


@pytest.mark.parametrize(
    ("make_data_args", "assert_paths"),
    [
        pytest.param(
            lambda base_dir: ImageClassificationMulticlassDataArgs(
                train="train",
                val="val",
                test="test",
                classes={0: "class_a"},
            ),
            _assert_image_classification_paths,
            id="image-classification-multiclass",
        ),
        pytest.param(
            lambda base_dir: ImageClassificationMultilabelDataArgs(
                train="train.csv",
                val="val.csv",
                test="test.csv",
                classes={0: "class_a"},
            ),
            _assert_image_classification_multilabel_paths,
            id="image-classification-multilabel",
        ),
        pytest.param(
            lambda base_dir: YOLOObjectDetectionDataArgs(
                path="dataset",
                train="images/train",
                val="images/val",
                names={0: "class_a"},
            ),
            _assert_yolo_object_detection_paths,
            id="yolo-object-detection",
        ),
        pytest.param(
            lambda base_dir: COCOObjectDetectionDataArgs(
                train=COCOObjectDetectionSplitArgs(
                    annotations="annotations/train.json",
                    images="images/train",
                ),
                val=COCOObjectDetectionSplitArgs(
                    annotations="annotations/val.json",
                    images=base_dir / "absolute/val_images",
                ),
            ),
            _assert_coco_object_detection_paths,
            id="coco-object-detection",
        ),
        pytest.param(
            lambda base_dir: YOLOOrientedObjectDetectionDataArgs(
                path="dataset",
                train="images/train",
                val="images/val",
                names={0: "class_a"},
            ),
            _assert_yolo_oriented_object_detection_paths,
            id="yolo-oriented-object-detection",
        ),
        pytest.param(
            lambda base_dir: YOLOInstanceSegmentationDataArgs(
                path="dataset",
                train="images/train",
                val="images/val",
                names={0: "class_a"},
            ),
            _assert_yolo_instance_segmentation_paths,
            id="yolo-instance-segmentation",
        ),
        pytest.param(
            lambda base_dir: COCOInstanceSegmentationDataArgs(
                train=COCOInstanceSegmentationSplitArgs(
                    annotations="annotations/train.json",
                    images="images/train",
                ),
                val=COCOInstanceSegmentationSplitArgs(
                    annotations="annotations/val.json",
                    images=base_dir / "absolute/val_images",
                ),
            ),
            _assert_coco_instance_segmentation_paths,
            id="coco-instance-segmentation",
        ),
        pytest.param(
            lambda base_dir: MaskSemanticSegmentationDataArgs(
                train=MaskSemanticSegmentationSplitArgs(
                    images="images/train", masks="masks/train"
                ),
                val=MaskSemanticSegmentationSplitArgs(
                    images="images/val", masks="masks/val"
                ),
                classes="classes.json",
            ),
            _assert_mask_semantic_segmentation_paths,
            id="mask-semantic-segmentation",
        ),
        pytest.param(
            lambda base_dir: MaskPanopticSegmentationDataArgs(
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
            ),
            _assert_mask_panoptic_segmentation_paths,
            id="mask-panoptic-segmentation",
        ),
    ],
)
def test_task_data_args_resolve_paths_relative_to_data_config_file(
    tmp_path: Path,
    make_data_args: Callable[[Path], TaskDataArgs],
    assert_paths: Callable[[TaskDataArgs, Path], None],
) -> None:
    data_config_file = tmp_path / "configs" / "data.yaml"
    data_config_file.parent.mkdir()
    base_dir = data_config_file.parent
    (base_dir / "classes.json").write_text('{"0": "background", "1": "car"}')
    data_args = _with_data_config_file(make_data_args(base_dir), data_config_file)

    data_args.resolve_data_paths()

    assert_paths(data_args, base_dir)


def test_task_data_args_resolve_paths_relative_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_args = YOLOObjectDetectionDataArgs(
        path="dataset",
        train="images/train",
        val="images/val",
        names={0: "class_a"},
    )

    data_args.resolve_data_paths()

    assert data_args.path == (tmp_path / "dataset").resolve()
    assert data_args.train == Path("images/train")
    assert data_args.val == Path("images/val")
