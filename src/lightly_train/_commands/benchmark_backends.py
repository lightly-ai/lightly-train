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
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from typing_extensions import override

from lightly_train._commands.benchmark_types import (
    ObjectDetectionPrediction,
    ONNXBackendArgs,
    TensorRTBackendArgs,
    TorchBackendArgs,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import ObjectDetectionBatch


def _rescale_and_filter_predictions(
    *,
    labels: Tensor,
    boxes: Tensor,
    scores: Tensor,
    metadata: list[dict[str, int]],
    model_w: int,
    model_h: int,
    threshold: float,
) -> list[ObjectDetectionPrediction]:
    """Rescale boxes from model input size to original image coordinates and filter by score threshold."""
    results: list[ObjectDetectionPrediction] = []
    for i in range(len(metadata)):
        orig_w = metadata[i]["orig_w"]
        orig_h = metadata[i]["orig_h"]
        img_boxes = boxes[i].clone()
        img_boxes[:, 0] *= orig_w / model_w
        img_boxes[:, 1] *= orig_h / model_h
        img_boxes[:, 2] *= orig_w / model_w
        img_boxes[:, 3] *= orig_h / model_h

        keep = scores[i] > threshold
        results.append(
            {
                "bboxes": img_boxes[keep],
                "scores": scores[i][keep],
                "labels": labels[i][keep],
            }
        )
    return results


class ObjectDetectionBackend(ABC):
    """Object detection backend."""

    @abstractmethod
    def run_batch(
        self, batch: ObjectDetectionBatch
    ) -> tuple[list[ObjectDetectionPrediction], float]:
        pass


class TorchBackend(ObjectDetectionBackend):
    def __init__(
        self,
        model: TaskModel,
        backend_args: TorchBackendArgs,
        device: torch.device,
        threshold: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.backend_args = backend_args
        self.threshold = threshold

        if hasattr(self.model, "deploy"):
            self.model.deploy()  # type: ignore[operator]
        else:
            self.model.eval()

        if backend_args.compile:
            self.model.forward_backend = torch.compile(self.model.forward_backend)  # type: ignore[method-assign]

    @override
    def run_batch(
        self, batch: ObjectDetectionBatch
    ) -> tuple[list[ObjectDetectionPrediction], float]:
        # preprocess
        images = batch["image"].to(self.device)
        metadata = [dict(orig_w=w, orig_h=h) for w, h in batch["original_size"]]

        # predict
        precision = self.backend_args.precision
        autocast_dtype = {
            "fp16-mixed": torch.float16,
            "bf16-mixed": torch.bfloat16,
        }.get(precision)
        start_predict = time.perf_counter()
        with torch.autocast(
            device_type=self.device.type,
            dtype=autocast_dtype or torch.float16,
            enabled=autocast_dtype is not None,
        ):
            raw_outputs = self.model.forward_backend(images)
        time_predict = time.perf_counter() - start_predict

        # postprocess
        return self.model.postprocess(
            raw_outputs=raw_outputs, metadata=metadata, threshold=self.threshold
        ), time_predict


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
    def __init__(
        self,
        model: TaskModel,
        backend_args: ONNXBackendArgs,
        batch_size: int,
        out_dir: Path,
        device: str,
        threshold: float = 0.0,
    ) -> None:

        import onnxruntime as ort

        self.threshold = threshold

        self.device = torch.device(device)

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
            self.device.index
            if self.device.type == "cuda" and self.device.index is not None
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

        self.precision = backend_args.precision
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    @override
    def run_batch(
        self, batch: ObjectDetectionBatch
    ) -> tuple[list[ObjectDetectionPrediction], float]:

        # preprocess
        # ONNX Runtime session.run() takes numpy arrays. The provider
        # (CPU/CUDA) handles device placement internally.
        images = batch["image"]
        if self.precision == "fp16":
            images = images.half()
        _, _, model_h, model_w = images.shape
        input_feed = {self.input_name: images.cpu().numpy()}
        metadata = [dict(orig_w=w, orig_h=h) for w, h in batch["original_size"]]

        # predict
        start_predict = time.perf_counter()
        raw_outputs = self.session.run(
            output_names=self.output_names, input_feed=input_feed
        )
        time_predict = time.perf_counter() - start_predict

        # postprocess
        outputs = dict(zip(self.output_names, raw_outputs))
        labels = torch.from_numpy(outputs["labels"])
        boxes_unscaled = torch.from_numpy(outputs["boxes"])
        scores = torch.from_numpy(outputs["scores"])

        # The ONNX forward() rescales boxes to the model input size when
        # orig_target_size is not provided. Rescale to original image
        # coordinates.
        results = _rescale_and_filter_predictions(
            labels=labels,
            boxes=boxes_unscaled,
            scores=scores,
            metadata=metadata,
            model_w=model_w,
            model_h=model_h,
            threshold=self.threshold,
        )
        return results, time_predict


class TensorRTBackend(ObjectDetectionBackend):
    def __init__(
        self,
        model: TaskModel,
        backend_args: TensorRTBackendArgs,
        batch_size: int,
        out_dir: Path,
        device: str,
        threshold: float = 0.0,
    ) -> None:
        import tensorrt as trt  # type: ignore[import-untyped,import-not-found]

        self.device = torch.device(device)
        self.precision = backend_args.precision
        self.threshold = threshold

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

        self.trt = trt

    @override
    def run_batch(
        self, batch: ObjectDetectionBatch
    ) -> tuple[list[ObjectDetectionPrediction], float]:
        import numpy as np

        # Preprocess.
        images = batch["image"]
        metadata = [dict(orig_w=w, orig_h=h) for w, h in batch["original_size"]]
        _, _, model_h, model_w = images.shape
        if self.precision == "fp16":
            images = images.half()
        images = images.to(device=self.device).contiguous()

        current_batch_size = images.shape[0]

        # Set input shape for dynamic batch dimension.
        actual_input_shape = [current_batch_size] + self.input_shape[1:]
        self.context.set_input_shape(self.input_name, actual_input_shape)

        # Set input tensor address.
        self.context.set_tensor_address(self.input_name, images.data_ptr())

        # Allocate output buffers and set their addresses.
        outputs: dict[str, Tensor] = {}
        for name in self.output_names:
            shape = list(self.context.get_tensor_shape(name))
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.zeros(1, dtype=dtype)).dtype
            outputs[name] = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self.context.set_tensor_address(name, outputs[name].data_ptr())

        # Predict.
        start_predict = time.perf_counter()
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        time_predict = time.perf_counter() - start_predict

        # Postprocess.
        labels_batch = outputs["labels"].cpu()
        boxes_batch = outputs["boxes"].cpu()
        scores_batch = outputs["scores"].cpu()

        results = _rescale_and_filter_predictions(
            labels=labels_batch,
            boxes=boxes_batch,
            scores=scores_batch,
            metadata=metadata,
            model_w=model_w,
            model_h=model_h,
            threshold=self.threshold,
        )
        return results, time_predict
