#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from typing_extensions import override

from lightly_train._commands.benchmark_types import (
    BenchmarkObjectDetectionBatch,
    ONNXBackendArgs,
    TensorRTBackendArgs,
    TorchBackendArgs,
)
from lightly_train._pre_post_processing.object_detection import (
    ObjectDetectionBatchOutput,
    ObjectDetectionMetadata,
    ObjectDetectionPostprocessor,
    ObjectDetectionPrediction,
    ObjectDetectionPreprocessor,
)
from lightly_train._task_models.task_model import TaskModel


@dataclass(frozen=True)
class ObjectDetectionPipeline:
    """The pre- and postprocessing that every backend shares.

    The batch is normalized by the model's own :class:`ObjectDetectionPreprocessor` and
    the raw outputs are decoded by the model's own
    :class:`ObjectDetectionPostprocessor`, so what the benchmark compares is the
    runtime, not the pipeline wrapped around it. Backends hold one of these rather than
    inheriting it, so each states in a single place where its torch-side work runs and
    in what dtype its runtime wants the input.

    Attributes:
        preprocessor:
            The model's preprocessor. Only its batched stage runs here; the per-image
            stage already ran in the dataloader.
        postprocessor: The model's postprocessor.
        device:
            Device the torch-side pre- and postprocessing runs on. This is where the
            runtime takes its input and hands back its outputs, which is not always
            where it computes: ONNX Runtime returns host-side numpy even with a CUDA
            provider.
        input_dtype:
            Dtype the model input is cast to after preprocessing, or ``None`` to feed
            it unchanged.
        threshold: Detections with a score <= threshold are discarded.
    """

    preprocessor: ObjectDetectionPreprocessor
    postprocessor: ObjectDetectionPostprocessor
    device: torch.device
    input_dtype: torch.dtype | None = None
    threshold: float = 0.0

    def __post_init__(self) -> None:
        # The postprocessor holds buffers, so it has to sit where the raw outputs do.
        self.postprocessor.to(self.device)

    def preprocess(self, images: Tensor) -> Tensor:
        """Normalize the stacked batch and bring it into the runtime's input format."""
        images = self.preprocessor.preprocess_batch(images.to(self.device))
        if self.input_dtype is not None:
            images = images.to(self.input_dtype)
        # A TensorRT engine binds its input by data pointer, so the input has to be
        # contiguous. For the other runtimes this is a no-op.
        return images.contiguous()

    def postprocess(
        self,
        raw_outputs: ObjectDetectionBatchOutput,
        metadata: Sequence[ObjectDetectionMetadata],
    ) -> list[ObjectDetectionPrediction]:
        """Decode raw outputs into one prediction per image."""
        return self.postprocessor.postprocess(
            # Decode in fp32 whatever precision the runtime ran in, so that the
            # backends differ in how they infer and not in how they decode.
            raw=raw_outputs.to(torch.float32),
            metadata=metadata,
            threshold=self.threshold,
        )


class ObjectDetectionBackend(ABC):
    """Runs benchmark batches through one inference runtime.

    Only the forward pass differs between the runtimes, and only the forward pass is
    timed. Everything around it is the shared :class:`ObjectDetectionPipeline`, which
    every backend builds in its own ``__init__``.

    The per-image metadata needed to decode the outputs is not re-derived here. It is
    what the preprocessor recorded when the dataloader prepared the batch and travels
    on the batch itself, which is also what lets a tiled (SAHI) batch work unchanged:
    an image then occupies several rows and only its metadata knows how many.
    """

    #: Built by every backend in its own ``__init__``.
    pipeline: ObjectDetectionPipeline

    @property
    def device(self) -> torch.device:
        """Device the torch-side stages run on."""
        return self.pipeline.device

    def run_batch(
        self, batch: BenchmarkObjectDetectionBatch
    ) -> tuple[list[ObjectDetectionPrediction], float]:
        """Run one batch and return its per-image predictions and the forward time."""
        images = self.pipeline.preprocess(batch["image"])
        self.setup_forward(images)
        start_predict = time.perf_counter()
        raw_outputs = self.forward(images)
        time_predict = time.perf_counter() - start_predict
        predictions = self.pipeline.postprocess(raw_outputs, batch["metadata"])
        return predictions, time_predict

    def setup_forward(self, images: Tensor) -> None:
        """Bind the batch to the runtime, before the clock starts.

        A hook for runtimes that need per-batch setup which is not inference and must
        not be timed, such as TensorRT allocating and binding its output buffers.
        Does nothing by default.
        """

    @abstractmethod
    def forward(self, images: Tensor) -> ObjectDetectionBatchOutput:
        """Run the batch through the runtime and return its raw outputs.

        This call is timed, so implementations must block until the outputs are
        actually available and do nothing beyond the inference itself. Wrapping the
        runtime's buffers in an :class:`ObjectDetectionBatchOutput` is part of it and
        is free: the exported graphs name their outputs after the fields of that
        dataclass, and the buffers are adopted rather than copied.
        """


class TorchBackend(ObjectDetectionBackend):
    """Runs the model as-is in PyTorch."""

    def __init__(
        self,
        model: TaskModel,
        backend_args: TorchBackendArgs,
        device: torch.device,
        preprocessor: ObjectDetectionPreprocessor,
        postprocessor: ObjectDetectionPostprocessor,
        threshold: float = 0.0,
    ) -> None:
        self.pipeline = ObjectDetectionPipeline(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            # Precision is handled by autocast, so the input stays as preprocessed.
            input_dtype=None,
            threshold=threshold,
        )
        self.model = model.to(device)
        self.backend_args = backend_args

        # TaskModel.deploy() defaults to a no-op that does not leave training mode,
        # so eval() has to be explicit.
        self.model.eval()
        self.model.deploy()

        if backend_args.compile:
            # Compiles the module's __call__ in place, so forward keeps calling the
            # model directly.
            self.model.compile()  # type: ignore[no-untyped-call]

        self.autocast_dtype = {
            "fp16-mixed": torch.float16,
            "bf16-mixed": torch.bfloat16,
        }.get(backend_args.precision)

    @override
    def forward(self, images: Tensor) -> ObjectDetectionBatchOutput:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype or torch.float16,
            enabled=self.autocast_dtype is not None,
        ):
            raw_outputs = self.model(images)
        # CUDA kernels are queued asynchronously, so without this the timer would stop
        # once the work is submitted rather than once it is done.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return cast(ObjectDetectionBatchOutput, raw_outputs)


_ONNX_PROVIDERS: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
}


class ONNXBackend(ObjectDetectionBackend):
    """Exports the model to ONNX and runs it through ONNX Runtime."""

    def __init__(
        self,
        model: TaskModel,
        backend_args: ONNXBackendArgs,
        batch_size: int,
        out_dir: Path,
        device: str,
        preprocessor: ObjectDetectionPreprocessor,
        postprocessor: ObjectDetectionPostprocessor,
        threshold: float = 0.0,
    ) -> None:

        import onnxruntime as ort

        self.pipeline = ObjectDetectionPipeline(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            # ONNX Runtime always returns host-side numpy outputs regardless of the
            # execution provider, so the torch-side pre- and postprocessing stay on
            # CPU no matter which provider runs the graph.
            device=torch.device("cpu"),
            input_dtype=torch.float16 if backend_args.precision == "fp16" else None,
            threshold=threshold,
        )

        # The exported graph is device independent, and exporting from CPU keeps the
        # trace off the device whose throughput is about to be measured.
        model.to("cpu")

        runtime_device = torch.device(device)

        export_kwargs = (
            dict(backend_args.export_args) if backend_args.export_args else {}
        )
        export_kwargs.setdefault("verify", False)
        export_kwargs["precision"] = backend_args.precision
        # If dynamic_batch_size is disabled, enforce the batch_size matches the benchmark batch_size.
        if not export_kwargs.get("dynamic_batch_size", True):
            export_kwargs["batch_size"] = batch_size
        # export_onnx is defined on subclasses, not on the base TaskModel.
        model.export_onnx(out=out_dir / "model.onnx", **export_kwargs)  # type: ignore[operator]

        providers = _ONNX_PROVIDERS[backend_args.provider]
        available = set(ort.get_available_providers())
        missing = [p for p in providers if p not in available]
        if missing:
            raise RuntimeError(
                f"ONNX provider '{backend_args.provider}' requires {missing} but only "
                f"{sorted(available)} are available."
            )
        # Set device_id so the correct GPU is used in multi-GPU setups.
        device_id = (
            runtime_device.index
            if runtime_device.type == "cuda" and runtime_device.index is not None
            else 0
        )
        provider_options: list[dict[str, Any]] = []
        for p in providers:
            opts: dict[str, Any] = {}
            if p == "TensorrtExecutionProvider":
                opts["trt_detailed_build_log"] = True
                opts["device_id"] = device_id
            elif p == "CUDAExecutionProvider":
                opts["device_id"] = device_id
            provider_options.append(opts)
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 2  # 0=Verbose, 1=Info, 2=Warning, 3=Error
        self.session = ort.InferenceSession(
            str(out_dir / "model.onnx"),
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )

        # Verify the requested provider is actually being used (not silently fallen back).
        active_providers = self.session.get_providers()
        expected_provider = providers[0]
        if expected_provider not in active_providers:
            raise RuntimeError(
                f"ONNX provider '{backend_args.provider}' failed to initialize. "
                f"Requested {expected_provider} but session is using {active_providers}. "
                f"Make sure the required libraries are installed and on LD_LIBRARY_PATH. "
                f"For pip-installed TensorRT, try: "
                f'LD_LIBRARY_PATH=$(python -c "import tensorrt_libs; print(tensorrt_libs.__path__[0])") '
                f"<your command>"
            )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    @override
    def forward(self, images: Tensor) -> ObjectDetectionBatchOutput:
        # session.run() takes numpy arrays and blocks until the outputs are back on
        # the host, whichever provider ran the graph.
        raw_outputs = self.session.run(
            output_names=self.output_names,
            input_feed={self.input_name: images.numpy()},
        )
        # The exported graph names its outputs after the fields of
        # ObjectDetectionBatchOutput, and from_numpy adopts the buffers rather than
        # copying them.
        return ObjectDetectionBatchOutput(
            **{
                name: torch.from_numpy(value)
                for name, value in zip(self.output_names, raw_outputs)
            }
        )


class TensorRTBackend(ObjectDetectionBackend):
    """Builds a TensorRT engine from the model and runs it directly."""

    def __init__(
        self,
        model: TaskModel,
        backend_args: TensorRTBackendArgs,
        batch_size: int,
        out_dir: Path,
        device: str,
        preprocessor: ObjectDetectionPreprocessor,
        postprocessor: ObjectDetectionPostprocessor,
        threshold: float = 0.0,
    ) -> None:
        import tensorrt as trt  # type: ignore[import-untyped,import-not-found]

        self.pipeline = ObjectDetectionPipeline(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            # The engine reads and writes device memory, so the torch-side stages run
            # on the same device and no host round trip is needed.
            device=torch.device(device),
            input_dtype=torch.float16 if backend_args.precision == "fp16" else None,
            threshold=threshold,
        )

        # Export model to TensorRT engine.
        engine_path = out_dir / "model.engine"
        export_kwargs = (
            dict(backend_args.export_args) if backend_args.export_args else {}
        )
        export_kwargs.setdefault("max_batchsize", batch_size)
        export_kwargs.setdefault("opt_batchsize", batch_size)
        export_kwargs.setdefault("min_batchsize", 1)
        export_kwargs.setdefault("verbose", False)
        export_kwargs["precision"] = backend_args.precision
        # Disable ONNX verification by default to avoid numerical precision
        # mismatches.
        onnx_args = export_kwargs.pop("onnx_args", {})
        onnx_args.setdefault("verify", False)
        # If dynamic_batch_size is disabled in onnx_args, enforce the
        # batch_size matches and set all TensorRT profile batch sizes to match
        # (static shape).
        if not onnx_args.get("dynamic_batch_size", True):
            onnx_args["batch_size"] = batch_size
            export_kwargs["min_batchsize"] = batch_size
            export_kwargs["opt_batchsize"] = batch_size
            export_kwargs["max_batchsize"] = batch_size
        export_kwargs["onnx_args"] = onnx_args
        model.to(device)
        model.export_tensorrt(out=engine_path, **export_kwargs)  # type: ignore[operator]

        # Load TensorRT engine.
        trt_logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Derive input/output tensor names from the engine (TensorRT 10.x API).
        self.output_names: list[str] = []
        self.input_name = ""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        # Get input shape (may have dynamic batch dimension = -1).
        self.input_shape = list(self.engine.get_tensor_shape(self.input_name))

        # Create CUDA stream for async execution on the target device.
        self.stream = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]

        # Filled in by setup_forward, which runs before every timed forward.
        self.output_buffers: dict[str, Tensor] = {}

        self.trt = trt

    @override
    def setup_forward(self, images: Tensor) -> None:
        # Bind the input. The engine has a dynamic batch dimension, so it has to be
        # told the shape it is about to run.
        self.context.set_input_shape(
            self.input_name, [images.shape[0]] + self.input_shape[1:]
        )
        self.context.set_tensor_address(self.input_name, images.data_ptr())

        # TensorRT writes into buffers the caller owns. Allocating them here keeps the
        # allocation out of the timed forward.
        self.output_buffers = {}
        for name in self.output_names:
            shape = list(self.context.get_tensor_shape(name))
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.zeros(1, dtype=dtype)).dtype
            buffer = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self.context.set_tensor_address(name, buffer.data_ptr())
            self.output_buffers[name] = buffer

    @override
    def forward(self, images: Tensor) -> ObjectDetectionBatchOutput:
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        # The engine names its outputs after the fields of ObjectDetectionBatchOutput.
        return ObjectDetectionBatchOutput(**self.output_buffers)
