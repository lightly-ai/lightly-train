"""Train an LT-DETR FastViT-T8 model on COCO 2017.

Example:
    python examples/train_object_detection_fastvit_t8_coco.py \
        --coco-root /path/to/coco2017 \
        --out out/fastvit_t8_coco

The COCO root is expected to contain ``train2017/``, ``val2017/``, and
``annotations/instances_{train,val}2017.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightly_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco-root",
        type=Path,
        required=True,
        help="Root directory of the COCO 2017 dataset.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("out/fastvit_t8_coco"),
        help="Directory for checkpoints and exported models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Global batch size. Omit to use LightlyTrain's model default.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total training steps. Omit to use LightlyTrain's model default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = args.coco_root / "annotations"

    train_kwargs: dict[str, object] = {
        "out": str(args.out),
        # FastViT-T8 is the smallest FastViT LT-DETR variant. Its backbone is
        # initialized from Apple's distilled ImageNet-1K checkpoint.
        "model": "fastvit/fastvit_t8-ltdetr",
        "data": {
            "format": "coco",
            "train": {
                "annotations": str(annotations / "instances_train2017.json"),
                "images": str(args.coco_root / "train2017"),
            },
            "val": {
                "annotations": str(annotations / "instances_val2017.json"),
                "images": str(args.coco_root / "val2017"),
            },
        },
    }
    if args.batch_size is not None:
        train_kwargs["batch_size"] = args.batch_size
    if args.steps is not None:
        train_kwargs["steps"] = args.steps

    lightly_train.train_object_detection(**train_kwargs)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
