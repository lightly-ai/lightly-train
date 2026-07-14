#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import contextlib
import contextvars
import logging
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lightning_utilities.core.imports import RequirementCache

if TYPE_CHECKING:
    import onnx

logger = logging.getLogger(__name__)

_TORCH_DYNAMO_MIN_VERSION = "2.5.0"
_TORCH_DYNAMO_AVAILABLE = RequirementCache(f"torch>={_TORCH_DYNAMO_MIN_VERSION}")


def check_torch_dynamo_requirements() -> None:
    """Raise if the installed torch version does not support dynamo ONNX export."""
    if not _TORCH_DYNAMO_AVAILABLE:
        raise RuntimeError(
            f"Dynamo support requires torch >= {_TORCH_DYNAMO_MIN_VERSION} "
            f", but found torch {torch.__version__}."
        )


_PRECALCULATE_FOR_ONNX_EXPORT = contextvars.ContextVar(
    "PRECALCULATE_FOR_ONNX_EXPORT", default=False
)


def is_in_precalculate_for_onnx_export() -> bool:
    return _PRECALCULATE_FOR_ONNX_EXPORT.get()


@contextlib.contextmanager
def precalculate_for_onnx_export() -> Iterator[None]:
    """
    For certain models we want to precalculate some values and store them in the model
    before exporting the model to ONNX. In order to avoid having to pass those options
    through all methods we have this context manager. Therefore, one should call
    ```
    with precalculate_for_onnx_export():
        model(example_input)
    ```
    before running `torch.onnx.export(model, example_input)`.
    In the relevant part of the model we can check if we are in this context with
    `is_in_precalculate_for_onnx_export()`.
    """
    token = _PRECALCULATE_FOR_ONNX_EXPORT.set(True)
    try:
        yield
    finally:
        _PRECALCULATE_FOR_ONNX_EXPORT.reset(token)


def write_onnx_metadata(out: str | Path, metadata: dict[str, str]) -> None:
    """Merge string key/value pairs into an ONNX model's metadata_props in-place.

    Existing metadata entries are preserved; keys in ``metadata`` override entries
    with the same key. The model is loaded from ``out`` and saved back to it.
    """
    import onnx

    model = onnx.load(str(out))
    merged = {entry.key: entry.value for entry in model.metadata_props}
    merged.update({str(key): str(value) for key, value in metadata.items()})

    del model.metadata_props[:]
    for key, value in merged.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(out))


def remove_redundant_casts(model: onnx.ModelProto) -> None:
    """Remove redundant -> Cast16 -> Cast32 -> pairs from an ONNX model in-place."""
    import onnx
    from onnx import TensorProto

    graph = model.graph

    # Build a map of tensor name -> element type for all tensors in the graph.
    tensor_types: dict[str, int] = {}
    for inp in graph.input:
        if inp.type.HasField("tensor_type"):
            tensor_types[inp.name] = inp.type.tensor_type.elem_type
    for init in graph.initializer:
        tensor_types[init.name] = init.data_type
    for vi in graph.value_info:
        if vi.type.HasField("tensor_type"):
            tensor_types[vi.name] = vi.type.tensor_type.elem_type
    for node in graph.node:
        if node.op_type == "Cast":
            to_attr = next((a for a in node.attribute if a.name == "to"), None)
            if to_attr:
                tensor_types[node.output[0]] = to_attr.i

    # Build a map of tensor name -> list of nodes that consume it.
    input_to_consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for inp in node.input:
            input_to_consumers.setdefault(inp, []).append(node)

    nodes_to_remove: set[int] = set()
    rewire: dict[str, str] = {}

    # Find X(fp32) -> Cast16 -> Cast32 -> Y patterns and mark for removal.
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        to_attr = next((a for a in node.attribute if a.name == "to"), None)
        if to_attr is None or to_attr.i != TensorProto.FLOAT16:
            continue

        input_type = tensor_types.get(node.input[0])
        if input_type != TensorProto.FLOAT:
            continue

        consumers = input_to_consumers.get(node.output[0], [])
        for consumer in consumers:
            if consumer.op_type != "Cast":
                continue
            to_attr2 = next((a for a in consumer.attribute if a.name == "to"), None)
            if to_attr2 is None or to_attr2.i != TensorProto.FLOAT:
                continue

            # Only remove Cast16 if it has no other consumers.
            if len(consumers) == 1:
                nodes_to_remove.add(id(node))
            # Always remove Cast32 and rewire its output to Cast16's fp32 input.
            nodes_to_remove.add(id(consumer))
            rewire[consumer.output[0]] = node.input[0]

    # Apply rewiring: replace all references to removed Cast32 outputs.
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp in rewire:
                node.input[i] = rewire[inp]

    identity_nodes = []
    for out in graph.output:
        if out.name in rewire:
            identity_nodes.append(
                onnx.helper.make_node("Identity", [rewire[out.name]], [out.name])
            )

    new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
    new_nodes.extend(identity_nodes)
    del graph.node[:]
    graph.node.extend(new_nodes)

    logger.info("Removed %d redundant Cast nodes", len(nodes_to_remove))


def fix_topological_order(model: onnx.ModelProto) -> None:
    """Sort graph nodes into topological order in-place."""
    from collections import deque

    graph = model.graph
    nodes = list(graph.node)
    if not nodes:
        return

    available: set[str] = set()
    for inp in graph.input:
        available.add(inp.name)
    for init in graph.initializer:
        available.add(init.name)

    output_to_node: dict[str, int] = {}
    for i, node in enumerate(nodes):
        for out in node.output:
            output_to_node[out] = i

    deps: dict[int, set[int]] = {i: set() for i in range(len(nodes))}
    for i, node in enumerate(nodes):
        for inp in node.input:
            if inp and inp not in available and inp in output_to_node:
                deps[i].add(output_to_node[inp])

    in_degree = {i: len(d) for i, d in deps.items()}
    dependents: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for i, dep_set in deps.items():
        for d in dep_set:
            dependents[d].append(i)

    queue: deque[int] = deque(i for i, d in in_degree.items() if d == 0)
    sorted_nodes = []
    while queue:
        idx = queue.popleft()
        sorted_nodes.append(nodes[idx])
        for dep in dependents[idx]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    del graph.node[:]
    graph.node.extend(sorted_nodes)


class ONNXPrecision(str, Enum):
    F16_TRUE = "16-true"
    F32_TRUE = "32-true"

    def torch_dtype(self) -> torch.dtype:
        return {
            ONNXPrecision.F32_TRUE: torch.float32,
            ONNXPrecision.F16_TRUE: torch.float16,
        }[self]
