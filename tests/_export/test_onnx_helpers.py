#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lightning_utilities.core.imports import RequirementCache

from lightly_train._export.onnx_helpers import remove_redundant_casts

pytestmark = pytest.mark.skipif(
    not RequirementCache("onnx"), reason="onnx not installed"
)


def _make_model(graph: Any) -> Any:
    import onnx

    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )


def test_removes_fp32_fp16_fp32_pair(tmp_path: Path) -> None:
    """A simple FP32->FP16->FP32 round-trip pair should be removed entirely."""
    import onnx

    cast_to_fp16 = onnx.helper.make_node(
        "Cast", ["X"], ["X_fp16"], name="cast_down", to=onnx.TensorProto.FLOAT16
    )
    cast_to_fp32 = onnx.helper.make_node(
        "Cast", ["X_fp16"], ["Y"], name="cast_up", to=onnx.TensorProto.FLOAT
    )
    graph = onnx.helper.make_graph(
        [cast_to_fp16, cast_to_fp32],
        "test",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])],
    )
    model_path = str(tmp_path / "model.onnx")
    onnx.save(_make_model(graph), model_path)

    model = onnx.load(model_path)
    remove_redundant_casts(model)

    assert len(model.graph.node) == 0


def test_preserves_non_cast_nodes(tmp_path: Path) -> None:
    """Non-cast nodes between other operations should be preserved."""
    import onnx

    relu = onnx.helper.make_node("Relu", ["X"], ["X_relu"], name="relu")
    cast_to_fp16 = onnx.helper.make_node(
        "Cast", ["X_relu"], ["X_fp16"], name="cast_down", to=onnx.TensorProto.FLOAT16
    )
    cast_to_fp32 = onnx.helper.make_node(
        "Cast", ["X_fp16"], ["Y"], name="cast_up", to=onnx.TensorProto.FLOAT
    )
    graph = onnx.helper.make_graph(
        [relu, cast_to_fp16, cast_to_fp32],
        "test",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])],
        value_info=[
            onnx.helper.make_tensor_value_info("X_relu", onnx.TensorProto.FLOAT, [1, 3])
        ],
    )
    model_path = str(tmp_path / "model.onnx")
    onnx.save(_make_model(graph), model_path)

    model = onnx.load(model_path)
    remove_redundant_casts(model)

    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Relu"
    # Relu output should feed directly to the graph output after rewiring.
    assert model.graph.output[0].name == "X_relu"


def test_keeps_fp16_cast_with_multiple_consumers(tmp_path: Path) -> None:
    """FP16 cast with multiple consumers: only the back-cast is removed, not the
    FP16 cast itself, since other consumers still need the FP16 tensor."""
    import onnx

    cast_to_fp16 = onnx.helper.make_node(
        "Cast", ["X"], ["X_fp16"], name="cast_down", to=onnx.TensorProto.FLOAT16
    )
    cast_to_fp32 = onnx.helper.make_node(
        "Cast", ["X_fp16"], ["Y"], name="cast_up", to=onnx.TensorProto.FLOAT
    )
    relu = onnx.helper.make_node("Relu", ["X_fp16"], ["Z"], name="relu")
    graph = onnx.helper.make_graph(
        [cast_to_fp16, cast_to_fp32, relu],
        "test",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3])],
        [
            onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3]),
            onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT16, [1, 3]),
        ],
    )
    model_path = str(tmp_path / "model.onnx")
    onnx.save(_make_model(graph), model_path)

    model = onnx.load(model_path)
    remove_redundant_casts(model)

    op_types = [n.op_type for n in model.graph.node]
    # The FP16 cast is kept (has 2 consumers), the FP32 back-cast is removed.
    assert "Relu" in op_types
    assert op_types.count("Cast") == 1
    cast_node = [n for n in model.graph.node if n.op_type == "Cast"][0]
    to_attr = next(a for a in cast_node.attribute if a.name == "to")
    assert to_attr.i == onnx.TensorProto.FLOAT16


def test_skips_fp16_input_cast_pair(tmp_path: Path) -> None:
    """A Cast(fp16->fp16->fp32) should NOT be removed because the input is
    already FP16, not FP32."""
    import onnx

    cast_to_fp16 = onnx.helper.make_node(
        "Cast", ["X"], ["X_fp16"], name="cast_down", to=onnx.TensorProto.FLOAT16
    )
    cast_to_fp32 = onnx.helper.make_node(
        "Cast", ["X_fp16"], ["Y"], name="cast_up", to=onnx.TensorProto.FLOAT
    )
    graph = onnx.helper.make_graph(
        [cast_to_fp16, cast_to_fp32],
        "test",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT16, [1, 3])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])],
    )
    model_path = str(tmp_path / "model.onnx")
    onnx.save(_make_model(graph), model_path)

    model = onnx.load(model_path)
    remove_redundant_casts(model)

    # Nothing should be removed since input is FP16, not FP32.
    assert len(model.graph.node) == 2


def test_preserves_useful_casts(tmp_path: Path) -> None:
    """A single FP32->FP16 cast (no back-cast) should be preserved."""
    import onnx

    cast_to_fp16 = onnx.helper.make_node(
        "Cast", ["X"], ["Y"], name="cast_down", to=onnx.TensorProto.FLOAT16
    )
    graph = onnx.helper.make_graph(
        [cast_to_fp16],
        "test",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [1, 3])],
    )
    model_path = str(tmp_path / "model.onnx")
    onnx.save(_make_model(graph), model_path)

    model = onnx.load(model_path)
    remove_redundant_casts(model)

    assert [n.op_type for n in model.graph.node] == ["Cast"]


def test_removes_multiple_pairs(tmp_path: Path) -> None:
    """Multiple independent FP32->FP16->FP32 pairs should all be removed."""
    import onnx

    nodes = []
    for i in range(3):
        nodes.append(
            onnx.helper.make_node(
                "Cast",
                [f"X{i}"],
                [f"X{i}_fp16"],
                name=f"cast_down_{i}",
                to=onnx.TensorProto.FLOAT16,
            )
        )
        nodes.append(
            onnx.helper.make_node(
                "Cast",
                [f"X{i}_fp16"],
                [f"Y{i}"],
                name=f"cast_up_{i}",
                to=onnx.TensorProto.FLOAT,
            )
        )

    graph = onnx.helper.make_graph(
        nodes,
        "test",
        [
            onnx.helper.make_tensor_value_info(f"X{i}", onnx.TensorProto.FLOAT, [1, 3])
            for i in range(3)
        ],
        [
            onnx.helper.make_tensor_value_info(f"Y{i}", onnx.TensorProto.FLOAT, [1, 3])
            for i in range(3)
        ],
    )
    model_path = str(tmp_path / "model.onnx")
    onnx.save(_make_model(graph), model_path)

    model = onnx.load(model_path)
    remove_redundant_casts(model)

    assert len(model.graph.node) == 0


def test_in_place_overwrite(tmp_path: Path) -> None:
    """Passing the same path for input and output should work (in-place)."""
    import onnx

    cast_to_fp16 = onnx.helper.make_node(
        "Cast", ["X"], ["X_fp16"], name="cast_down", to=onnx.TensorProto.FLOAT16
    )
    cast_to_fp32 = onnx.helper.make_node(
        "Cast", ["X_fp16"], ["Y"], name="cast_up", to=onnx.TensorProto.FLOAT
    )
    graph = onnx.helper.make_graph(
        [cast_to_fp16, cast_to_fp32],
        "test",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])],
    )
    model_path = str(tmp_path / "model.onnx")
    onnx.save(_make_model(graph), model_path)

    model = onnx.load(model_path)
    remove_redundant_casts(model)
    onnx.save(model, model_path)

    reloaded = onnx.load(model_path)
    assert len(reloaded.graph.node) == 0


def test_rewires_downstream_consumers(tmp_path: Path) -> None:
    """After removing a cast pair, downstream nodes should reference the
    original FP32 tensor, not the removed cast's output."""
    import onnx

    cast_to_fp16 = onnx.helper.make_node(
        "Cast", ["X"], ["X_fp16"], name="cast_down", to=onnx.TensorProto.FLOAT16
    )
    cast_to_fp32 = onnx.helper.make_node(
        "Cast", ["X_fp16"], ["X_back"], name="cast_up", to=onnx.TensorProto.FLOAT
    )
    relu = onnx.helper.make_node("Relu", ["X_back"], ["Y"], name="relu")
    graph = onnx.helper.make_graph(
        [cast_to_fp16, cast_to_fp32, relu],
        "test",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])],
        value_info=[
            onnx.helper.make_tensor_value_info("X_back", onnx.TensorProto.FLOAT, [1, 3])
        ],
    )
    model_path = str(tmp_path / "model.onnx")
    onnx.save(_make_model(graph), model_path)

    model = onnx.load(model_path)
    remove_redundant_casts(model)

    assert len(model.graph.node) == 1
    relu_node = model.graph.node[0]
    assert relu_node.op_type == "Relu"
    # Relu should now consume "X" directly, not "X_back".
    assert relu_node.input[0] == "X"
