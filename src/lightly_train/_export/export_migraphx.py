#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from lightly_train._export.export_onnx import ONNXExportMixin
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)


class MIGraphXExportMixin(ONNXExportMixin):
    """Export ONNX-capable task models as compiled MIGraphX programs."""

    def export_migraphx(
        self,
        out: PathLike,
        *,
        precision: Literal["fp32", "fp16"] = "fp32",
    ) -> None:
        """Export the model to a compiled MIGraphX program.

        The intermediate FP32 ONNX model is saved alongside ``out`` with an
        ``.onnx`` extension. For FP16 programs, MIGraphX converts that model
        before GPU compilation.

        MIGraphX must be installed separately from AMD's ROCm apt repository.
        See the official MIGraphX documentation for installation instructions:
        https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/install.html
        """
        if precision not in ("fp32", "fp16"):
            raise ValueError(
                f"Invalid precision '{precision}'. Must be one of 'fp32', 'fp16'."
            )

        try:
            import migraphx
        except ModuleNotFoundError as error:
            if error.name != "migraphx":
                raise
            raise ImportError(
                "MIGraphX is not installed. Install it from AMD's ROCm apt "
                "repository with `sudo apt install migraphx`. Its Python bindings "
                "usually reside in '/opt/rocm/lib/'; add that directory to "
                "PYTHONPATH, for example: `export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH`. "
                "For official installation instructions, see "
                "https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/install.html"
            ) from error

        onnx_out = Path(out).with_suffix(".onnx")
        self.export_onnx(out=onnx_out)

        logger.info(f"Exporting MIGraphX model to '{out}'")
        program = migraphx.parse_onnx(str(onnx_out))
        if precision == "fp16":
            migraphx.quantize_fp16(program)
        program.compile(t=migraphx.get_target("gpu"))
        migraphx.save(program, str(out))
