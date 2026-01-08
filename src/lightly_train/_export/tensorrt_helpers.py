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
from typing import TYPE_CHECKING, Any, Callable

import torch

from lightly_train import _logging
from lightly_train.types import PathLike

if TYPE_CHECKING:
    import tensorrt as trt  # type: ignore[import-untyped,import-not-found]

logger = logging.getLogger(__name__)


@torch.no_grad()
def export_tensorrt(
    export_onnx_fn: Callable[..., None],
    out: PathLike,
    onnx_args: dict[str, Any] | None = None,
    max_batchsize: int = 1,
    opt_batchsize: int = 1,
    min_batchsize: int = 1,
    use_fp16: bool = False,
    fp32_attention_scores: bool = False,
    verbose: bool = False,
    debug: bool = False,
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
        use_fp16:
            Enable FP16 precision if supported by the platform.
        fp32_attention_scores:
            Force attention score computations to use FP32 precision.
        verbose:
            Enable verbose TensorRT logging.
        debug:
            Enable debug mode for TensorRT engine building.

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

    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, trt_logger)

    # Set the ONNX export path.
    if onnx_args is None:
        onnx_args = {}
    onnx_args.setdefault("out", Path(out).with_suffix(".onnx"))

    # Export the model to ONNX.
    export_onnx_fn(**onnx_args)

    onnx_out = onnx_args["out"]
    logger.info(f"Loading ONNX file from {onnx_out}")
    with open(onnx_out, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")

    if fp32_attention_scores:
        _force_fp32_for_attention_scores(network)

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

    # Verify that fp16 can be used if requested.
    if use_fp16:
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
    """Force FP32 precision for attention score computations in the network.

    This fixes TRT FP16 NaNs while keeping most of the network FP16.
    This is required for EoMT with FP16.
    """
    import tensorrt as trt

    force_fp32_names = {"/MatMul", "/Softmax"}
    for i in range(net.num_layers):
        layer = net.get_layer(i)
        if layer.name in force_fp32_names:
            layer.precision = trt.DataType.FLOAT
            for j in range(layer.num_outputs):
                out_tensor = layer.get_output(j)
                if out_tensor is not None:
                    out_tensor.dtype = trt.DataType.FLOAT
            logger.info(f"Forcing FP32 for layer: {layer.name} ({layer.type})")
