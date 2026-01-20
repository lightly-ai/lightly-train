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
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch

from lightly_train import _logging
from lightly_train.types import PathLike

if TYPE_CHECKING:
    import tensorrt as trt  # type: ignore[import-untyped,import-not-found]

logger = logging.getLogger(__name__)


@torch.no_grad()
def export_tensorrt(
    *,
    export_onnx_fn: Callable[..., None],
    out: PathLike,
    precision: Literal["auto", "fp32", "fp16"],
    model_dtype: torch.dtype,
    onnx_args: dict[str, Any] | None = None,
    max_batchsize: int = 1,
    opt_batchsize: int = 1,
    min_batchsize: int = 1,
    fp32_attention_scores: bool = False,
    verbose: bool = False,
    debug: bool = False,
    update_network_fn: Callable[[trt.INetworkDefinition], None] | None = None,
) -> None:
    """Build a TensorRT engine from an ONNX model.

    .. note::
        TensorRT is not part of LightlyTrain’s dependencies and must be installed separately.
        Installation depends on your OS, Python version, GPU, and NVIDIA driver/CUDA setup.
        See the `TensorRT documentation <https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html>`_ for more details.
        On CUDA 12.x systems you can often install the Python package via `pip install tensorrt-cu12`.

    This loads the ONNX file, parses it with TensorRT, infers the static input
    shape (C, H, W) from the `"images"` input, and creates an engine with a
    dynamic batch dimension in the range `[min_batchsize, opt_batchsize, max_batchsize]`.
    Spatial dimensions must be static in the ONNX model (dynamic H/W are not yet supported).

    The engine is serialized and written to `out`.

    Args:
        export_onnx_fn:
            A callable that exports the model to ONNX when called with
            keyword arguments. Typically this is the self.export_onnx method for
            the model to be exported.
        out:
            Path where the TensorRT engine will be saved.
        precision:
            Precision hint used for TensorRT engine building.
        model_dtype:
            The dtype of the model being exported. Used to determine how to handle
            `precision="auto"`.
        onnx_args:
            Optional arguments to pass to `export_onnx` when exporting
            the ONNX model prior to building the TensorRT engine. If None,
            default arguments are used and the ONNX file is saved alongside
            the TensorRT engine with the same name but `.onnx` extension.
        max_batchsize:
            Maximum supported batch size.
        opt_batchsize:
            Batch size TensorRT optimizes for.
        min_batchsize:
            Minimum supported batch size.
        fp32_attention_scores:
            Force attention score computations to use FP32 precision.
        verbose:
            Enable verbose TensorRT logging.
        debug:
            Enable debug mode for TensorRT engine building.
        update_network_fn:
            Optional function that takes the TensorRT network definition
            as input and can be used to modify the network before building
            the engine.

    Raises:
        FileNotFoundError: If the ONNX file does not exist.
        RuntimeError: If the ONNX cannot be parsed or engine building fails.
        ValueError: If batch size constraints are invalid or H/W are dynamic.
    """
    # Try to import TensorRT.
    try:
        import tensorrt as trt  # type: ignore[import-untyped,import-not-found]
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "TensorRT is required, but is not installed.\n"
            "Install TensorRT for your system by following NVIDIA's guide:\n"
            "https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html"
        ) from e

    # TODO(Guarin, 12/25): Move warnings module out of commands subpackage and
    # move import to the top of the file.
    from lightly_train._commands import _warnings

    # Set up logging.
    _warnings.filter_export_warnings()
    _logging.set_up_console_logging()

    trt_logger = trt.Logger(
        trt.Logger.VERBOSE if (verbose or debug) else trt.Logger.INFO
    )

    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, trt_logger)

    onnx_args = {} if onnx_args is None else onnx_args
    onnx_args.setdefault("out", Path(out).with_suffix(".onnx"))

    # Export the model to ONNX.
    export_onnx_fn(**onnx_args)
    onnx_out = onnx_args["out"]

    if debug:
        _debug_mark_all_layers_as_outputs(onnx_out)

    logger.info(f"Loading ONNX file from {onnx_out}")
    with open(onnx_out, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")

    if fp32_attention_scores:
        _force_fp32_for_attention_scores(network)

    if update_network_fn is not None:
        update_network_fn(network)

    # Infer input shape from the ONNX model
    images_input = None
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if inp.name == "images":  # your ONNX export uses this name
            images_input = inp
            break

    # Raise error if input not found.
    if images_input is None:
        raise RuntimeError("Could not find 'images' input in ONNX network.")

    # Get input shape.
    input_shape = images_input.shape
    _, C, H, W = input_shape

    # Verify that H and W are not dynamic, i.e., not -1.
    # TODO(Thomas, 12/25): Support dynamic H and W in the future.
    if H == -1 or W == -1:
        raise ValueError("Dynamic image height and width are not supported yet.")
    logger.info(f"Detected input shape: (N, {C}, {H}, {W})")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Avoid TF32 in mixed precision paths (can affect stability)
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)

    # Use FP16 if requested and supported.
    if precision == "fp16" or (model_dtype == torch.float16 and precision == "auto"):
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

            # Ensure TensorRT respects layer.precision and tensor dtype overrides.
            if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            elif hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

            logger.info("FP16 optimization enabled.")
        else:
            logger.warning("FP16 not supported on this platform. Proceeding with FP32.")

    if debug:
        config.set_flag(trt.BuilderFlag.DEBUG)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        logger.info("Debug mode enabled.")

    profile = builder.create_optimization_profile()
    if not (min_batchsize <= opt_batchsize <= max_batchsize):
        raise ValueError("Batch sizes must satisfy: min <= opt <= max")
    profile.set_shape(
        "images",
        min=(min_batchsize, C, H, W),
        opt=(opt_batchsize, C, H, W),
        max=(max_batchsize, C, H, W),
    )
    config.add_optimization_profile(profile)

    logger.info("Building TensorRT engine...")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        raise RuntimeError("Failed to build the engine.")

    logger.info(f"Saving engine to {out}")
    with open(out, "wb") as f:
        f.write(engine)
    logger.info("Engine export complete.")


def _force_fp32_for_attention_scores(net: trt.INetworkDefinition) -> None:
    import tensorrt as trt

    io_tensors = {
        *(net.get_input(i) for i in range(net.num_inputs)),
        *(net.get_output(i) for i in range(net.num_outputs)),
    }

    # Collect inputs of layers whose name contains "Softmax"
    softmax_inputs: set[str] = set()
    for i in range(net.num_layers):
        layer = net.get_layer(i)
        if "Softmax" in layer.name:
            inp = layer.get_input(0)
            if inp is not None:
                softmax_inputs.add(inp.name)

    forced_softmax = 0
    forced_matmul = 0

    for i in range(net.num_layers):
        layer = net.get_layer(i)

        # Force all Softmax layers to FP32 (covers attention + class softmax)
        if "Softmax" in layer.name:
            layer.precision = trt.DataType.FLOAT
            for j in range(layer.num_outputs):
                out = layer.get_output(j)
                if out in io_tensors:
                    out.dtype = trt.DataType.FLOAT
            forced_softmax += 1
            logger.debug(f"Forcing FP32 for Softmax layer: {layer.name}")
            continue

        # Force only MatMul whose output feeds a Softmax (attention scores)
        if "MatMul" in layer.name:
            for j in range(layer.num_outputs):
                out = layer.get_output(j)
                if out is not None and out.name in softmax_inputs:
                    layer.precision = trt.DataType.FLOAT
                    if out in io_tensors:
                        out.dtype = trt.DataType.FLOAT
                    forced_matmul += 1
                    logger.debug(
                        f"Forcing FP32 for attention-score MatMul layer: {layer.name}"
                    )
                    break
            continue

    logger.info(
        f"Forced FP32 on Softmax layers: {forced_softmax}, attention-score MatMul layers: {forced_matmul}"
    )


def _debug_mark_all_layers_as_outputs(out: Path) -> None:
    """Mark all intermediate tensors as model outputs for debugging.

    You can then debug the model with:
        polygraphy run model.trt --model-type engine --trt --validate --fail-fast > out_trt.txt

    And compare with ONNX:
        polygraphy run model.onnx --trt --fp16 --validate --fail-fast --trt-outputs $(cat all_outputs.txt) > out_onnx.txt
    """
    from typing import Set

    import onnx
    from onnx import helper, shape_inference

    def save_all_outputs(inp: Path, out: Path) -> None:
        m = onnx.load(inp)

        seen: set[str] = set()
        with open(out, "w") as f:
            for n in m.graph.node:
                for o in n.output:
                    if o and o not in seen:
                        seen.add(o)
                        f.write(o + "\n")

    save_all_outputs(out, out.parent / "all_outputs.txt")

    def make_all_outputs(inp: Path, out: Path) -> None:
        m = onnx.load(inp)
        m = shape_inference.infer_shapes(m)  # ensures type info
        g = m.graph

        existing: Set[str] = {o.name for o in g.output}
        vi = {x.name: x for x in list(g.value_info) + list(g.input) + list(g.output)}

        for n in g.node:
            for t in n.output:
                if not t or t in existing or t not in vi:
                    continue
                tt = vi[t].type.tensor_type
                g.output.append(helper.make_tensor_value_info(t, tt.elem_type, None))
                existing.add(t)

        onnx.save(m, out)

    make_all_outputs(out, out)
