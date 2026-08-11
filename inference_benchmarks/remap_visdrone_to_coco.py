#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Rewrite a VisDrone yolo dataset so that its class ids become COCO class ids.

The COCO-pretrained detection checkpoints predict contiguous ids 0..79 in the standard
COCO class order, and benchmark_object_detection matches ground truth against
predictions by integer class id without aligning them by name. A VisDrone data config
therefore has to be translated before it can be benchmarked: its native ids
(0 pedestrian .. 9 motor) mean something entirely different to the checkpoint.

This script writes a new dataset root that
    - symlinks each split's `images` directory to the original images, so that no image
      is copied,
    - writes new `labels` files with the class ids remapped from VisDrone to COCO,
      dropping the boxes of classes that have no COCO counterpart,
    - writes a `*_coco_remapped.yaml` data config listing all 80 COCO classes.

All 80 names are listed even though only six of them occur in the labels. The yolo
dataset maps the keys of `names` to internal ids by enumerating them in insertion order,
so a sparse subset such as {0, 1, 2, 3, 5, 7} would be compacted to 0..5 and would no
longer line up with the checkpoint's ids, silently destroying the mAP. Listing all 80
makes that mapping the identity. Classes that have predictions but no ground truth are
reported as AP=-1 and excluded from the mean, so the unused names do not depress the
reported mAP.

The `images` symlinks point at absolute paths: the source dataset lives at a fixed
location and the output tree is a disposable derivative of it.

Usage:
    python remap_visdrone_to_coco.py \\
        --data /path/to/visdrone/data.yaml --out /path/to/visdrone_coco_remapped
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# COCO 80-class names in the order the COCO-pretrained checkpoints emit them, so that the
# index into this list is the class id.
COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# The mapping to edit if you disagree with it. Keys are VisDrone class names as they
# appear in the source data config, values are COCO class names, or None for classes
# whose boxes are dropped.
#
# pedestrian and people both become person: VisDrone splits humans into walking/standing
# and everything else, COCO does not. van becomes car rather than truck, which is what
# the COCO annotations do with passenger vans. tricycle and awning-tricycle have no COCO
# counterpart at all; mapping them to motorcycle would score a different object, so their
# boxes are dropped instead.
VISDRONE_TO_COCO_CLASS_NAME: dict[str, str | None] = {
    "pedestrian": "person",
    "people": "person",
    "bicycle": "bicycle",
    "car": "car",
    "van": "car",
    "truck": "truck",
    "tricycle": None,
    "awning-tricycle": None,
    "bus": "bus",
    "motor": "motorcycle",
}

# Split keys read from the source config. Only val is required.
SPLIT_KEYS = ("train", "val", "test")


@dataclass(frozen=True)
class Split:
    """One split of the source dataset and its counterpart in the output tree."""

    key: str
    # The value as it appears in the source config, e.g. "VisDrone2019-DET-val/images".
    # Copied verbatim into the generated config.
    images: str
    src_images_dir: Path
    src_labels_dir: Path
    out_images_link: Path
    out_labels_dir: Path


@dataclass
class RemapStats:
    """Box and file counters, printed as a table at the end of the run."""

    # Keyed by VisDrone class name.
    kept: Counter[str] = field(default_factory=Counter)
    dropped: Counter[str] = field(default_factory=Counter)
    # Keyed by split.
    label_files: Counter[str] = field(default_factory=Counter)
    images_without_labels: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the source VisDrone data YAML in yolo format, or to the dataset "
        "root containing a data.yaml.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Root directory of the remapped dataset. Must differ from the source root. "
        "The generated data YAML is written into it.",
    )
    parser.add_argument(
        "--yaml-name",
        default=None,
        help="File name of the generated data YAML. Defaults to "
        "'<source root name>_coco_remapped.yaml'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the labels directories and images symlinks of an existing output "
        "tree. Only paths created by this script are removed, a real images directory is "
        "never deleted.",
    )
    parser.add_argument(
        "--skip-name-check",
        action="store_true",
        help="Do not require every class name of the source config to appear in "
        "VISDRONE_TO_COCO_CLASS_NAME. Unknown classes are dropped.",
    )
    return parser.parse_args()


def load_source_config(data: Path) -> tuple[Path, dict[str, Any]]:
    """Return the resolved path of the source config and its parsed contents."""
    if data.is_dir():
        data = data / "data.yaml"
    data = data.resolve()
    with data.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected {data} to contain a YAML mapping.")
    config: dict[str, Any] = loaded
    if config.get("format") != "yolo":
        raise ValueError(
            f"Expected a yolo data config, got format={config.get('format')!r} in {data}."
        )
    return data, config


def get_source_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names")
    if not isinstance(names, dict):
        raise ValueError("The source config must contain a 'names' mapping.")
    return {int(class_id): str(name) for class_id, name in names.items()}


def get_source_root(*, config: dict[str, Any], source_config_file: Path) -> Path:
    """Resolve the dataset root, mirroring how lightly-train resolves a relative path."""
    root = Path(str(config.get("path", "."))).expanduser()
    if not root.is_absolute():
        root = source_config_file.parent / root
    return root.resolve()


def build_class_id_mapping(
    names: dict[int, str], *, skip_name_check: bool
) -> dict[int, int]:
    """Map source class ids to COCO class ids.

    The mapping is resolved by class name, so it does not depend on the order of 'names'
    in the source config. Classes without a COCO counterpart are absent from the result
    and their boxes are dropped.
    """
    coco_name_to_id = {name: index for index, name in enumerate(COCO_CLASS_NAMES)}
    unknown_targets = sorted(
        target
        for target in VISDRONE_TO_COCO_CLASS_NAME.values()
        if target is not None and target not in coco_name_to_id
    )
    if unknown_targets:
        raise ValueError(
            f"VISDRONE_TO_COCO_CLASS_NAME maps to unknown COCO classes: {unknown_targets}."
        )

    unmapped = sorted(set(names.values()) - set(VISDRONE_TO_COCO_CLASS_NAME))
    if unmapped and not skip_name_check:
        raise ValueError(
            f"The source config contains classes that VISDRONE_TO_COCO_CLASS_NAME does "
            f"not cover: {unmapped}. Extend the mapping or pass --skip-name-check to "
            f"drop their boxes."
        )

    mapping: dict[int, int] = {}
    for class_id, class_name in names.items():
        coco_name = VISDRONE_TO_COCO_CLASS_NAME.get(class_name)
        if coco_name is not None:
            mapping[class_id] = coco_name_to_id[coco_name]
    return mapping


def labels_path_for(images: Path) -> Path:
    """Replace the first 'images' component with 'labels', as lightly-train does.

    Mirrors the labels path derivation of the yolo dataset so that the tree written here
    is the tree the dataset looks for.
    """
    parts = list(images.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            break
    else:
        raise ValueError(
            f"Expected the split path to include an 'images' directory, got {images}."
        )
    return Path(*parts)


def collect_splits(
    *, config: dict[str, Any], source_root: Path, out_root: Path
) -> list[Split]:
    splits: list[Split] = []
    for key in SPLIT_KEYS:
        images = config.get(key)
        if images is None:
            continue
        images_path = Path(str(images))
        labels_path = labels_path_for(images_path)
        splits.append(
            Split(
                key=key,
                images=str(images),
                src_images_dir=(source_root / images_path).resolve(),
                src_labels_dir=(source_root / labels_path).resolve(),
                out_images_link=out_root / images_path,
                out_labels_dir=out_root / labels_path,
            )
        )
    if not any(split.key == "val" for split in splits):
        raise ValueError("The source config must define a 'val' split.")
    return splits


def prepare_out_split(split: Split, *, overwrite: bool) -> None:
    """Create the images symlink and an empty labels directory for one split."""
    for path in (split.out_images_link, split.out_labels_dir):
        if not (path.exists() or path.is_symlink()):
            continue
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Pass --overwrite to replace it."
            )
        if path.is_symlink():
            # Never rmtree a symlink, out_images_link points at the original images.
            path.unlink()
        elif path == split.out_labels_dir:
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"{path} is a real directory and not a symlink created by this script. "
                f"Refusing to delete it."
            )

    for src_dir in (split.src_images_dir, split.src_labels_dir):
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Source directory {src_dir} does not exist.")

    split.out_labels_dir.mkdir(parents=True)
    split.out_images_link.parent.mkdir(parents=True, exist_ok=True)
    split.out_images_link.symlink_to(split.src_images_dir, target_is_directory=True)


def remap_label_file(
    *,
    src: Path,
    dst: Path,
    class_id_mapping: dict[int, int],
    names: dict[int, str],
    stats: RemapStats,
) -> None:
    """Rewrite one yolo label file, keeping the box coordinates byte-identical."""
    out_lines: list[str] = []
    for number, line in enumerate(
        src.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(
                f"{src}:{number}: expected 5 fields, got {len(fields)}: {line!r}"
            )
        class_id = int(fields[0])
        class_name = names.get(class_id, f"<unknown id {class_id}>")
        coco_id = class_id_mapping.get(class_id)
        if coco_id is None:
            stats.dropped[class_name] += 1
            continue
        stats.kept[class_name] += 1
        # The box fields are copied verbatim so that the geometry does not change.
        out_lines.append(" ".join([str(coco_id), *fields[1:]]))
    # Written even when empty: an image whose boxes were all dropped still contributes
    # false positives to the mAP, and a missing file would look like a failed conversion.
    dst.write_text("".join(f"{line}\n" for line in out_lines), encoding="utf-8")


def remap_split(
    *,
    split: Split,
    class_id_mapping: dict[int, int],
    names: dict[int, str],
    stats: RemapStats,
) -> None:
    label_stems: set[str] = set()
    for src_label_file in sorted(split.src_labels_dir.glob("*.txt")):
        remap_label_file(
            src=src_label_file,
            dst=split.out_labels_dir / src_label_file.name,
            class_id_mapping=class_id_mapping,
            names=names,
            stats=stats,
        )
        label_stems.add(src_label_file.stem)
    stats.label_files[split.key] = len(label_stems)

    image_stems = {
        path.stem for path in split.src_images_dir.iterdir() if path.is_file()
    }
    stats.images_without_labels[split.key] = len(image_stems - label_stems)


def write_data_config(
    *, path: Path, splits: list[Split], source_config_file: Path
) -> None:
    config: dict[str, Any] = {"format": "yolo", "path": "."}
    for split in splits:
        config[split.key] = split.images
    config["names"] = dict(enumerate(COCO_CLASS_NAMES))
    header = (
        f"# Generated by inference_benchmarks/remap_visdrone_to_coco.py from\n"
        f"# {source_config_file}. Do not edit, regenerate instead.\n"
        f"#\n"
        f"# The class ids in labels/ are COCO ids, not VisDrone ids. All 80 COCO classes\n"
        f"# are listed so that the internal class id mapping is the identity and lines\n"
        f"# up with the ids the COCO-pretrained checkpoints predict.\n"
    )
    # sort_keys=False keeps format/path/train/val/names in this order, and keeps names in
    # COCO order, which is what makes the internal mapping the identity.
    path.write_text(header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def print_summary(
    *,
    stats: RemapStats,
    names: dict[int, str],
    class_id_mapping: dict[int, int],
    config_path: Path,
) -> None:
    print("\n| VisDrone class | COCO id | COCO class | boxes kept | boxes dropped |")
    print("| --- | ---: | --- | ---: | ---: |")
    for class_id, class_name in sorted(names.items()):
        coco_id = class_id_mapping.get(class_id)
        coco_id_str = "-" if coco_id is None else str(coco_id)
        coco_name = "(dropped)" if coco_id is None else COCO_CLASS_NAMES[coco_id]
        print(
            f"| {class_name} | {coco_id_str} | {coco_name} "
            f"| {stats.kept[class_name]} | {stats.dropped[class_name]} |"
        )
    print(
        f"\nBoxes: {sum(stats.kept.values())} kept, {sum(stats.dropped.values())} dropped."
    )
    for split_key in stats.label_files:
        print(
            f"Split {split_key}: {stats.label_files[split_key]} label files, "
            f"{stats.images_without_labels[split_key]} images without a label file."
        )
    print(f"\nWrote data config to {config_path}")


def main() -> None:
    args = parse_args()
    source_config_file, config = load_source_config(Path(args.data).expanduser())
    source_root = get_source_root(config=config, source_config_file=source_config_file)
    out_root = Path(args.out).expanduser().resolve()
    if out_root == source_root:
        raise ValueError("--out must differ from the source dataset root.")

    names = get_source_names(config)
    class_id_mapping = build_class_id_mapping(
        names, skip_name_check=args.skip_name_check
    )
    splits = collect_splits(config=config, source_root=source_root, out_root=out_root)

    stats = RemapStats()
    out_root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        print(
            f"Remapping {split.key}: {split.src_labels_dir} -> {split.out_labels_dir}"
        )
        prepare_out_split(split, overwrite=args.overwrite)
        remap_split(
            split=split,
            class_id_mapping=class_id_mapping,
            names=names,
            stats=stats,
        )

    yaml_name = args.yaml_name or f"{source_root.name}_coco_remapped.yaml"
    config_path = out_root / yaml_name
    write_data_config(
        path=config_path, splits=splits, source_config_file=source_config_file
    )
    print_summary(
        stats=stats,
        names=names,
        class_id_mapping=class_id_mapping,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
