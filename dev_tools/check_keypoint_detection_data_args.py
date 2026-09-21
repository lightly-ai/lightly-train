#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Checks the keypoint detection data args against real annotation files.

This is a developer tool, not a test: it downloads data and is therefore never run in
CI. Its purpose is twofold.

1. It exercises ``COCOKeypointDetectionDataArgs`` and ``YOLOKeypointDetectionDataArgs``
   against real annotations rather than synthetic ones.
2. It reports the structural properties of those annotations that the synthetic test
   fixtures in ``tests/helpers.py`` reproduce, so that the fixtures are derived from real
   data instead of from assumptions.

Run it with:

    uv run python dev_tools/check_keypoint_detection_data_args.py

Findings as of 2026-09-21, which the fixtures and the readers are built on:

COCO keypoints (4 images, 14 annotations)
    - ``categories[].skeleton`` is one-indexed: indices run from 1 to 17 for 17
      keypoints.
    - Visibility flags use all three values, 0/1/2.
    - Every keypoint with visibility 0 has coordinates exactly (0, 0); there were no
      counterexamples.
    - ``num_keypoints`` agreed with a recount of the non-zero visibility flags for all 14
      annotations, but the readers recount anyway because third-party exports are not
      always consistent.
    - ``num_keypoints`` is 0 for some annotations, so a COCO keypoint file legitimately
      contains instances without a single labeled keypoint.
    - No visible keypoint fell outside the image bounds in this file.

YOLO pose, num_dims == 3 (8 label files, 21 instances)
    - 56 values per line, matching 5 + 17 * 3.
    - Visibility flags are written as floats, e.g. "2.000000", so they have to be parsed
      via float before int.
    - Every keypoint with visibility 0 has coordinates exactly (0, 0), 149 out of 149.
    - No negative values.

YOLO pose, num_dims == 2 (263 label files, 3156 keypoints)
    - 29 values per line, matching 5 + 12 * 2.
    - Not a single (0, 0) coordinate pair and not a single negative coordinate. The
      format simply carries no visibility information, so the reader treats every
      keypoint of a num_dims == 2 dataset as labeled rather than guessing a sentinel.
"""

from __future__ import annotations

import collections
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from lightly_train._data.keypoint_detection_dataset import (
    COCOKeypointDetectionDataArgs,
    COCOSplitArgs,
    YOLOKeypointDetectionDataArgs,
)
from lightly_train._data.keypoint_helpers import KeypointSetArgs

# A real COCO person_keypoints file, pinned to a commit so that the findings above stay
# reproducible. mmpose is Apache-2.0 licensed.
COCO_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/open-mmlab/mmpose/"
    "ec2f372f002d1d534ea01a13033d09f5483256db/tests/data/coco/test_coco.json"
)
# A real YOLO pose dataset with num_dims == 3. Only the data is used here; no code is
# taken from Ultralytics. Swap the URL for any other YOLO pose export if preferred.
YOLO_POSE_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip"
)
YOLO_POSE_NUM_KEYPOINTS = 17
YOLO_POSE_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())
    return destination


def print_keypoint_set(keypoint_set: KeypointSetArgs) -> None:
    print("  keypoint set:")
    print(f"    num_keypoints: {keypoint_set.num_keypoints}")
    print(f"    names        : {keypoint_set.names}")
    print(f"    sigmas       : {keypoint_set.sigmas}")
    print(f"    flip_idx     : {keypoint_set.flip_idx}")
    print(f"    skeleton     : {keypoint_set.skeleton}")


def print_rows(rows: List[Dict[str, str]], limit: int = 2) -> None:
    for row in rows[:limit]:
        bboxes = json.loads(row["bboxes"])
        keypoints = json.loads(row["keypoints"])
        visibility = json.loads(row["keypoint_visibility"])
        print(f"    {Path(row['image_path']).name}: {len(bboxes)} instance(s)")
        for i, (box, points, flags) in enumerate(zip(bboxes, keypoints, visibility)):
            rounded_box = [round(value, 4) for value in box]
            head = [[round(value, 4) for value in point] for point in points[:3]]
            print(f"      instance {i}: bbox={rounded_box}")
            print(f"        first 3 keypoints: {head}")
            print(f"        visibility       : {flags}")


def check_columns(rows: List[Dict[str, str]], num_keypoints: int) -> int:
    """Asserts the column contract and returns the total number of instances."""
    total = 0
    for row in rows:
        bboxes = json.loads(row["bboxes"])
        class_labels = json.loads(row["class_labels"])
        keypoints = json.loads(row["keypoints"])
        visibility = json.loads(row["keypoint_visibility"])
        assert all(isinstance(value, str) for value in row.values())
        assert len(bboxes) == len(class_labels) == len(keypoints) == len(visibility)
        for points, flags in zip(keypoints, visibility):
            assert len(points) == num_keypoints
            assert len(flags) == num_keypoints
            for point, flag in zip(points, flags):
                if flag == 0:
                    assert point == [0.0, 0.0], point
        total += len(bboxes)
    return total


def check_coco(work_dir: Path) -> None:
    print("=" * 72)
    print("COCO keypoints")
    print("=" * 72)
    annotations = download(COCO_ANNOTATIONS_URL, work_dir / "coco" / "annotations.json")
    raw: Dict[str, Any] = json.loads(annotations.read_text())

    print("\n  structural properties of the raw file:")
    print(f"    images: {len(raw['images'])}, annotations: {len(raw['annotations'])}")
    for category in raw["categories"]:
        names = category.get("keypoints") or []
        skeleton = category.get("skeleton") or []
        flat = [index for pair in skeleton for index in pair]
        print(
            f"    category {category['id']} ({category.get('name')!r}): "
            f"{len(names)} keypoint names, {len(skeleton)} skeleton edges"
        )
        if flat:
            print(
                f"      skeleton index range: {min(flat)}..{max(flat)} for "
                f"{len(names)} keypoints -> "
                f"{'one-indexed' if min(flat) == 1 else 'zero-indexed'}"
            )
    visibility_values: collections.Counter[int] = collections.Counter()
    unlabeled_at_origin: collections.Counter[bool] = collections.Counter()
    num_keypoints_mismatches = 0
    for annotation in raw["annotations"]:
        flat_keypoints = annotation.get("keypoints")
        if not flat_keypoints:
            continue
        flags = flat_keypoints[2::3]
        visibility_values.update(flags)
        recount = sum(1 for flag in flags if flag > 0)
        if recount != annotation.get("num_keypoints"):
            num_keypoints_mismatches += 1
        for x, y, flag in zip(
            flat_keypoints[0::3], flat_keypoints[1::3], flat_keypoints[2::3]
        ):
            if flag == 0:
                unlabeled_at_origin[x == 0 and y == 0] += 1
    print(f"    visibility values: {dict(sorted(visibility_values.items()))}")
    print(f"    unlabeled keypoints at (0, 0): {dict(unlabeled_at_origin)}")
    print(
        f"    annotations where num_keypoints disagrees with the flags: "
        f"{num_keypoints_mismatches}"
    )
    print(
        f"    num_keypoints values: "
        f"{sorted({a.get('num_keypoints') for a in raw['annotations']})}"
    )

    split = COCOSplitArgs(annotations=annotations)
    data_args = COCOKeypointDetectionDataArgs(train=split, val=split)
    print("\n  as read by COCOKeypointDetectionDataArgs:")
    print_keypoint_set(data_args.keypoint_set)
    print(f"  included classes: {data_args.included_classes}")

    rows = list(data_args.get_train_args().list_image_info())
    total = check_columns(rows, data_args.keypoint_set.num_keypoints)
    print(f"  rows: {len(rows)} (file has {len(raw['images'])} images)")
    print(f"  instances kept: {total} (file has {len(raw['annotations'])} annotations)")
    assert len(rows) == len(raw["images"])
    assert total == len(raw["annotations"])
    print_rows(rows)

    strict_args = COCOKeypointDetectionDataArgs(train=split, val=split, min_keypoints=1)
    strict_rows = list(strict_args.get_train_args().list_image_info())
    strict_total = check_columns(strict_rows, strict_args.keypoint_set.num_keypoints)
    print(
        f"  instances kept with min_keypoints=1: {strict_total} "
        f"({total - strict_total} dropped)"
    )


def check_yolo(work_dir: Path) -> None:
    print()
    print("=" * 72)
    print("YOLO pose")
    print("=" * 72)
    archive = download(YOLO_POSE_URL, work_dir / "yolo" / "dataset.zip")
    extract_dir = work_dir / "yolo" / "extracted"
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(extract_dir)
    dataset_dirs = [path for path in extract_dir.iterdir() if path.is_dir()]
    assert len(dataset_dirs) == 1, dataset_dirs
    dataset_dir = dataset_dirs[0]

    label_paths = sorted(dataset_dir.glob("labels/*/*.txt"))
    values_per_line: collections.Counter[int] = collections.Counter()
    for label_path in label_paths:
        for line in label_path.read_text().splitlines():
            if line.strip():
                values_per_line[len(line.split())] += 1
    print("\n  structural properties of the raw labels:")
    print(f"    label files: {len(label_paths)}")
    for count, occurrences in sorted(values_per_line.items()):
        for num_dims in (2, 3):
            if (count - 5) % num_dims == 0:
                print(
                    f"    {occurrences} line(s) with {count} values -> "
                    f"kpt_shape=[{(count - 5) // num_dims}, {num_dims}]"
                )

    data_args = YOLOKeypointDetectionDataArgs(
        path=dataset_dir,
        train="images/train",
        val="images/val",
        names={0: "person"},
        kpt_shape=[YOLO_POSE_NUM_KEYPOINTS, 3],
        flip_idx=YOLO_POSE_FLIP_IDX,
    )
    print("\n  as read by YOLOKeypointDetectionDataArgs:")
    print_keypoint_set(data_args.keypoint_set)
    print(f"  included classes: {data_args.included_classes}")

    for split_name, dataset_args in (
        ("train", data_args.get_train_args()),
        ("val", data_args.get_val_args()),
    ):
        rows = list(dataset_args.list_image_info())
        total = check_columns(rows, data_args.keypoint_set.num_keypoints)
        print(f"  {split_name}: {len(rows)} images, {total} instances")
        print_rows(rows, limit=1)


def main() -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="lightly-train-keypoint-check-"))
    try:
        check_coco(work_dir=work_dir)
        check_yolo(work_dir=work_dir)
        print()
        print("all checks passed")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
