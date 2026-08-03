#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from lightly_train._export.export_onnx import ONNXExportMixin
from lightly_train.types import PathLike

logger = logging.getLogger(__name__)

_DISABLED_MIGRAPHX_PASS = "simplify_reshapes"


def _unique_onnx_name(*, base: str, existing_names: set[str]) -> str:
    name = base
    suffix = 0
    while name in existing_names:
        suffix += 1
        name = f"{base}_{suffix}"
    existing_names.add(name)
    return name


def _prepare_onnx_for_migraphx(onnx_out: Path) -> None:
    """Apply compatibility rewrites required by MIGraphX 2.15.

    MIGraphX evaluates Resize incorrectly when its input is the strided output of
    an NHWC-to-NCHW transpose. Move eligible static four-dimensional resizes
    before the transpose, which is mathematically equivalent and gives Resize a
    contiguous input.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(onnx_out)
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    producers = {
        output: node for node in model.graph.node for output in node.output if output
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    existing_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    existing_names.update(initializers)
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    replacements: dict[str, tuple[onnx.NodeProto, onnx.NodeProto]] = {}
    replaced_resize_outputs: set[str] = set()
    stale_value_info_names: set[str] = set()
    for resize in model.graph.node:
        if resize.op_type != "Resize" or len(resize.input) < 4:
            continue
        resize_attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in resize.attribute
        }
        if resize_attributes.get("mode", b"nearest") != b"linear":
            continue
        if "axes" in resize_attributes:
            continue

        transpose_output = resize.input[0]
        producer = producers.get(transpose_output)
        if producer is None or producer.op_type != "Transpose":
            continue
        transpose_attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in producer.attribute
        }
        if transpose_attributes.get("perm") != [0, 3, 1, 2]:
            continue
        if consumers.get(transpose_output) != [resize]:
            continue

        sizes_name = resize.input[3]
        if not sizes_name or sizes_name not in initializers:
            continue
        sizes = numpy_helper.to_array(initializers[sizes_name])
        if sizes.shape != (4,):
            continue

        nhwc_sizes_name = _unique_onnx_name(
            base=f"{sizes_name}_migraphx_nhwc", existing_names=existing_names
        )
        model.graph.initializer.append(
            numpy_helper.from_array(sizes[[0, 2, 3, 1]], name=nhwc_sizes_name)
        )
        resized_nhwc_name = _unique_onnx_name(
            base=f"{resize.output[0]}_migraphx_nhwc", existing_names=existing_names
        )
        rewritten_resize = helper.make_node(
            "Resize",
            [producer.input[0], resize.input[1], resize.input[2], nhwc_sizes_name],
            [resized_nhwc_name],
            name=f"{resize.name}_migraphx_contiguous",
        )
        rewritten_resize.attribute.extend(resize.attribute)
        rewritten_transpose = helper.make_node(
            "Transpose",
            [resized_nhwc_name],
            list(resize.output),
            name=f"{producer.name}_migraphx_after_resize",
            perm=[0, 3, 1, 2],
        )
        replacements[transpose_output] = (rewritten_resize, rewritten_transpose)
        replaced_resize_outputs.update(resize.output)
        stale_value_info_names.add(transpose_output)

    if replacements:
        rewritten_nodes: list[onnx.NodeProto] = []
        for node in model.graph.node:
            replacement = next(
                (
                    replacements[output]
                    for output in node.output
                    if output in replacements
                ),
                None,
            )
            if replacement is not None:
                rewritten_nodes.extend(replacement)
            elif not any(output in replaced_resize_outputs for output in node.output):
                rewritten_nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)
        retained_value_info = [
            value_info
            for value_info in model.graph.value_info
            if value_info.name not in stale_value_info_names
        ]
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_value_info)

    changed = False
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        attributes = [
            attribute
            for attribute in node.attribute
            if not (
                attribute.name == "keep_aspect_ratio_policy"
                and attribute.s == b"stretch"
            )
        ]
        if len(attributes) != len(node.attribute):
            del node.attribute[:]
            node.attribute.extend(attributes)
            changed = True
    if changed or replacements:
        onnx.checker.check_model(model)
        onnx.save(model, onnx_out)


def _migraphx_compile_environment() -> dict[str, str]:
    env = os.environ.copy()
    disabled_passes = {
        name for name in env.get("MIGRAPHX_DISABLE_PASSES", "").split(",") if name
    }
    disabled_passes.add(_DISABLED_MIGRAPHX_PASS)
    env["MIGRAPHX_DISABLE_PASSES"] = ",".join(sorted(disabled_passes))
    return env


def _compile_migraphx(
    *, onnx_out: Path, out: Path, precision: Literal["fp32", "fp16"]
) -> None:
    command = [
        sys.executable,
        "-m",
        "lightly_train._export._compile_migraphx",
        "--onnx",
        str(onnx_out),
        "--out",
        str(out),
        "--precision",
        precision,
    ]
    try:
        subprocess.run(command, check=True, env=_migraphx_compile_environment())
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"MIGraphX compilation failed with exit code {error.returncode}."
        ) from error


class MIGraphXExportMixin(ONNXExportMixin):
    """Export ONNX-capable task models as compiled MIGraphX programs."""

    def export_migraphx(
        self,
        out: PathLike,
        *,
        precision: Literal["fp32", "fp16"] = "fp32",
        batch_size: int = 1,
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

        if importlib.util.find_spec("migraphx") is None:
            raise ImportError(
                "MIGraphX is not installed. Install it from AMD's ROCm apt "
                "repository with `sudo apt install migraphx`. Its Python bindings "
                "usually reside in '/opt/rocm/lib/'; add that directory to "
                "PYTHONPATH, for example: `export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH`. "
                "For official installation instructions, see "
                "https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/install.html"
            )

        onnx_out = Path(out).with_suffix(".onnx")
        self.export_onnx(
            out=onnx_out,
            batch_size=batch_size,
            dynamic_batch_size=False,
            opset_version=19,
            simplify=False,
        )
        _prepare_onnx_for_migraphx(onnx_out)

        logger.info(f"Exporting MIGraphX model to '{out}'")
        _compile_migraphx(onnx_out=onnx_out, out=Path(out), precision=precision)
