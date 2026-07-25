#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal


def compile_migraphx(
    *, onnx_path: Path, out: Path, precision: Literal["fp32", "fp16"]
) -> None:
    """Compile an ONNX model into a serialized MIGraphX GPU program."""
    import migraphx  # type: ignore[import-not-found]

    program = migraphx.parse_onnx(str(onnx_path))
    if precision == "fp16":
        migraphx.quantize_fp16(program)
    program.compile(t=migraphx.get_target("gpu"))
    migraphx.save(program, str(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--precision", required=True, choices=("fp32", "fp16"))
    args = parser.parse_args()
    compile_migraphx(
        onnx_path=args.onnx,
        out=args.out,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
