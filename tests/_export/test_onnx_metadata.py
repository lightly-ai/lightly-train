#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path
from typing import Any

import pytest
from lightning_utilities.core.imports import RequirementCache

from lightly_train._export.onnx_helpers import write_onnx_metadata

pytestmark = pytest.mark.skipif(
    not RequirementCache("onnx"), reason="onnx not installed"
)


def _make_model(graph: Any) -> Any:
    import onnx

    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )


def test_write_onnx_metadata__merges_and_preserves(tmp_path: Path) -> None:
    import onnx

    graph = onnx.helper.make_graph([], "test", [], [])
    path = tmp_path / "model.onnx"
    onnx.save(_make_model(graph), str(path))

    write_onnx_metadata(path, {"a": "1", "b": "2"})
    write_onnx_metadata(path, {"b": "22", "c": "3"})

    model = onnx.load(str(path))
    metadata = {entry.key: entry.value for entry in model.metadata_props}
    assert metadata == {"a": "1", "b": "22", "c": "3"}
