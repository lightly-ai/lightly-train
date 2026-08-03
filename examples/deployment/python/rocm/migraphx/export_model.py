#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Export an LT-DETR checkpoint to a compiled MIGraphX engine.

Example:
    python export_model.py --out /workspace/dinov3-vitt16-ltdetr-coco.mxr
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import lightly_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="dinov3/vitt16-ltdetr-coco",
        help="LightlyTrain checkpoint name or path to export.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output path for the compiled MIGraphX engine (.mxr). A sibling "
        "ONNX file is written alongside it.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Static batch size to export the engine with.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Precision to compile the MIGraphX engine with.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    precision: Literal["fp32", "fp16"] = args.precision

    model = lightly_train.load_model(args.checkpoint, device="cpu")

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    model.export_migraphx(out, batch_size=args.batch_size, precision=precision)

    print(f"Exported ONNX model to {out.with_suffix('.onnx')}")
    print(f"Exported MIGraphX engine to {out}")


if __name__ == "__main__":
    main()
