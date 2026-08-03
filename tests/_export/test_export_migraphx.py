#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

from lightly_train._export.export_migraphx import (
    _compile_migraphx,
    _migraphx_compile_environment,
    _prepare_onnx_for_migraphx,
)


def _write_transpose_resize_model(
    path: Path, *, shared_transpose: bool = False
) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    nodes = [
        helper.make_node(
            "Transpose", ["images"], ["transposed"], name="transpose", perm=[0, 3, 1, 2]
        ),
        helper.make_node(
            "Resize",
            ["transposed", "", "", "sizes"],
            ["resized"],
            name="resize",
            mode="linear",
            coordinate_transformation_mode="half_pixel",
            keep_aspect_ratio_policy="stretch",
        ),
    ]
    outputs = [
        helper.make_tensor_value_info("resized", TensorProto.FLOAT, [1, 3, 8, 8])
    ]
    if shared_transpose:
        nodes.append(
            helper.make_node("Identity", ["transposed"], ["identity"], name="identity")
        )
        outputs.append(
            helper.make_tensor_value_info("identity", TensorProto.FLOAT, [1, 3, 16, 16])
        )
    graph = helper.make_graph(
        nodes,
        "transpose_resize",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 16, 16, 3])],
        outputs,
        [numpy_helper.from_array(np.array([1, 3, 8, 8]), name="sizes")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    onnx.save(model, path)


def test_prepare_onnx_for_migraphx__moves_resize_before_transpose(
    tmp_path: Path,
) -> None:
    pytest.importorskip("onnxruntime")
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    path = tmp_path / "model.onnx"
    _write_transpose_resize_model(path)
    inputs = np.random.default_rng(0).normal(size=(1, 16, 16, 3)).astype(np.float32)
    expected = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {"images": inputs}
    )

    _prepare_onnx_for_migraphx(path)

    actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
        None, {"images": inputs}
    )
    np.testing.assert_array_equal(actual[0], expected[0])

    model = onnx.load(path)
    assert [node.op_type for node in model.graph.node] == ["Resize", "Transpose"]
    resize = model.graph.node[0]
    resize_attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in resize.attribute
    }
    assert "keep_aspect_ratio_policy" not in resize_attributes
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    np.testing.assert_array_equal(initializers[resize.input[3]], np.array([1, 8, 8, 3]))


def test_prepare_onnx_for_migraphx__leaves_shared_transpose_in_place(
    tmp_path: Path,
) -> None:
    onnx = pytest.importorskip("onnx")

    path = tmp_path / "model.onnx"
    _write_transpose_resize_model(path, shared_transpose=True)

    _prepare_onnx_for_migraphx(path)

    model = onnx.load(path)
    assert [node.op_type for node in model.graph.node] == [
        "Transpose",
        "Resize",
        "Identity",
    ]


def test_migraphx_compile_environment__adds_pass_and_preserves_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MIGRAPHX_DISABLE_PASSES", "foo,bar")

    env = _migraphx_compile_environment()

    assert env["MIGRAPHX_DISABLE_PASSES"] == "bar,foo,simplify_reshapes"
    assert env is not os.environ
    assert os.environ["MIGRAPHX_DISABLE_PASSES"] == "foo,bar"


def test_compile_migraphx__uses_isolated_child_environment(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MIGRAPHX_DISABLE_PASSES", raising=False)
    run = mocker.patch("lightly_train._export.export_migraphx.subprocess.run")
    onnx_out = tmp_path / "model.onnx"
    out = tmp_path / "model.mxr"

    _compile_migraphx(onnx_out=onnx_out, out=out, precision="fp16")

    run.assert_called_once()
    command = run.call_args.args[0]
    assert command == [
        sys.executable,
        "-m",
        "lightly_train._export._compile_migraphx",
        "--onnx",
        str(onnx_out),
        "--out",
        str(out),
        "--precision",
        "fp16",
    ]
    assert run.call_args.kwargs["check"] is True
    assert run.call_args.kwargs["env"]["MIGRAPHX_DISABLE_PASSES"] == "simplify_reshapes"


def test_compile_migraphx__wraps_child_failure(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    mocker.patch(
        "lightly_train._export.export_migraphx.subprocess.run",
        side_effect=subprocess.CalledProcessError(returncode=7, cmd=[]),
    )

    with pytest.raises(RuntimeError, match="exit code 7"):
        _compile_migraphx(
            onnx_out=tmp_path / "model.onnx",
            out=tmp_path / "model.mxr",
            precision="fp32",
        )
