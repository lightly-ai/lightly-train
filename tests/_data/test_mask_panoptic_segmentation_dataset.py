#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path

from lightly_train._data.mask_panoptic_segmentation_dataset import (
    MaskPanopticSegmentationDataArgs,
    SplitArgs,
)

from .. import helpers


class TestMaskPanopticSegmentationMmapHash:
    @staticmethod
    def _make_args(tmp_path: Path) -> MaskPanopticSegmentationDataArgs:
        helpers.create_coco_panoptic_segmentation_dataset(tmp_path)
        return MaskPanopticSegmentationDataArgs(
            train=SplitArgs(
                images=tmp_path / "images" / "train",
                masks=tmp_path / "annotations" / "train",
                annotations=tmp_path / "annotations" / "train.json",
            ),
            val=SplitArgs(
                images=tmp_path / "images" / "val",
                masks=tmp_path / "annotations" / "val",
                annotations=tmp_path / "annotations" / "val.json",
            ),
        )

    def test_mmap_hash_is_deterministic(self, tmp_path: Path) -> None:
        args = self._make_args(tmp_path)
        assert args.train_data_mmap_hash() == args.train_data_mmap_hash()
        assert args.val_data_mmap_hash() == args.val_data_mmap_hash()

    def test_mmap_hash_changes_when_annotations_modified(self, tmp_path: Path) -> None:
        args = self._make_args(tmp_path)
        hash_before = args.train_data_mmap_hash()
        annotations_path = tmp_path / "annotations" / "train.json"
        st = annotations_path.stat()
        os.utime(annotations_path, (st.st_atime, st.st_mtime + 1))
        assert args.train_data_mmap_hash() != hash_before
