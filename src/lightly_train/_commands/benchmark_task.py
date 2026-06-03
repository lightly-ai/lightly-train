#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Sequence, TypedDict

import torch
from pydantic import Field
from torch import Tensor
from torch.utils.data import DataLoader

from lightly_train._commands import common_helpers
from lightly_train._commands.benchmark_types import (
    BenchmarkObjectDetectionConfig,
    BenchmarkObjectDetectionMetricArgs,
    BenchmarkResult,
    BenchmarkStatistics,
    BenchmarkTimingResult,
    DescriptiveStatistics,
    ONNXBackendArgs,
    TensorRTBackendArgs,
)
from lightly_train._configs import validate
from lightly_train._data.task_data_args import TaskDataArgs
from lightly_train._metrics.detection.task_metric import (
    ObjectDetectionTaskMetric,
    ObjectDetectionTaskMetricArgs,
)
from lightly_train._task_models import task_model_helpers
from lightly_train._task_models.object_detection_components.utils import (
    _denormalize_xyxy_boxes,
    _yolo_to_xyxy,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train._transforms.object_detection_transform import (
    ObjectDetectionCollateFunction,
    ObjectDetectionTransform,
    ObjectDetectionTransformArgs,
)
from lightly_train._transforms.transform import ResizeArgs
from lightly_train.types import PathLike

_PreprocessFn = Callable[[list[str]], Any]
_InferFn = Callable[[Any], Any]
_PostprocessFn = Callable[[Any, float, int | None], list[dict[str, Tensor]]]


def benchmark_object_detection(
    *,
    out: PathLike,
    data: dict[str, Any] | str,
    model: TaskModel | PathLike,
    batch_size: int = 16,
    warmup_steps: int = 0,
    steps: int | None = None,
    num_workers: int | Literal["auto"] = "auto",
    overwrite: bool = False,
    debug: bool = False,
    metric_args: dict[str, Any] | None = None,
    backend_args: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Benchmark an object detection model on a validation dataset.

    Runs inference on the validation split and computes mAP and other detection
    metrics.

    Args:
        out:
            Output directory where benchmark results are saved.
        data:
            Dataset configuration dictionary (same format as train_object_detection).
        model:
            A loaded TaskModel instance or a path to an exported model file.
        batch_size:
            Number of images to process at once.
        warmup_steps:
            Number of warmup batches to run before the benchmark. Warmup
            results are discarded. The dataloader restarts from the beginning
            after warmup.
        steps:
            Maximum number of batches to process. None means process all
            validation images.
        num_workers:
            Number of workers for data loading.
        overwrite:
            Overwrite the output directory if it already exists.
        metric_args:
            Metric configuration. Supports ``detection_threshold`` (float, default
            0.6) to filter low-confidence predictions, ``top_k`` (int or None) to
            limit detections per image, ``classwise`` (bool) for per-class metrics,
            and ``map`` (dict or None) for MeanAveragePrecision configuration.
        backend_args:
            Backend configuration. Use ``format`` to select the backend:
            ``"torch"`` (default), ``"onnx"``, or ``"tensorrt"``. ONNX and
            TensorRT backends accept an optional ``export_args`` dict forwarded
            to ``model.export_onnx()``.

    Returns:
        BenchmarkResult containing metric values and timing statistics.
    """
    args = {k: v for k, v in locals().items() if v is not None}
    config = validate.pydantic_model_validate(BenchmarkObjectDetectionConfig, args)
    return _benchmark_object_detection_from_config(config=config)


def _benchmark_object_detection_from_config(
    config: BenchmarkObjectDetectionConfig,
) -> BenchmarkResult:
    # Set up output directory.
    out_dir = _get_out_dir(out=config.out, overwrite=config.overwrite)

    # Load model if a path is given.
    model: TaskModel
    if isinstance(config.model, TaskModel):
        model = config.model
    else:
        model = task_model_helpers.load_model(model=config.model)

    metric_args = config.metric_args

    backend_args = config.backend_args

    # Set up validation data.
    data_args = config.data
    num_workers = common_helpers.get_num_workers(
        num_workers=config.num_workers, num_devices_per_node=1
    )
    val_dataloader = _create_val_dataloader(
        data_args=data_args,
        batch_size=config.batch_size,
        num_workers=num_workers,
    )
    total_images = len(val_dataloader.dataset)  # type: ignore[arg-type]

    # Set up metric.
    class_names = list(data_args.included_classes.values())
    metric = _create_metric(metric_args=metric_args, class_names=class_names)

    # Set up inference pipeline based on backend.
    if isinstance(backend_args, ONNXBackendArgs):
        preprocess_fn, infer_fn, postprocess_fn = _create_onnx_pipeline(
            model=model,
            out_dir=out_dir,
            export_args=backend_args.export_args,
            batch_size=config.batch_size,
            provider=backend_args.provider,
        )
    elif isinstance(backend_args, TensorRTBackendArgs):
        preprocess_fn, infer_fn, postprocess_fn = _create_tensorrt_pipeline(
            model=model,
            out_dir=out_dir,
            export_args=backend_args.export_args,
            batch_size=config.batch_size,
        )
    else:
        preprocess_fn, infer_fn, postprocess_fn = _create_torch_pipeline(model=model)

    # Warmup.
    # Deploy model for inference (sets eval mode and enables optimized postprocessing).
    if hasattr(model, "deploy"):
        model.deploy()
    else:
        model.eval()
    if config.warmup_steps > 0:
        print(f"Running {config.warmup_steps} warmup steps...")
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                if step >= config.warmup_steps:
                    break
                preprocessed = preprocess_fn(batch["image_path"])
                raw_outputs = infer_fn(preprocessed)
                postprocess_fn(
                    raw_outputs,
                    metric_args.detection_threshold,
                    metric_args.top_k,
                )
        print("Warmup complete.")

    # Run inference in batches.
    total_batches = len(val_dataloader)
    if config.steps is not None:
        total_batches = min(total_batches, config.steps)

    print_every = max(1, min(10, total_batches // 10))

    batch_times: list[float] = []
    _debug = config.debug
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            t_load = time.perf_counter()
            if step >= total_batches:
                break
            image_paths = batch["image_path"]
            targets = batch["targets"]

            t_preprocess = time.perf_counter()
            preprocessed = preprocess_fn(image_paths)
            t_infer = time.perf_counter()
            raw_outputs = infer_fn(preprocessed)
            t_postprocess = time.perf_counter()
            batch_times.append(t_postprocess - t_infer)
            predictions = postprocess_fn(
                raw_outputs,
                metric_args.detection_threshold,
                metric_args.top_k,
            )

            t_metrics = time.perf_counter()
            predictions_cpu = _to_cpu(predictions)
            metric.update_with_predictions(predictions_cpu, targets)
            t_end = time.perf_counter()

            if _debug:
                load_time = t_preprocess - t_load
                preprocess_time = t_infer - t_preprocess
                infer_time = t_postprocess - t_infer
                postprocess_time = t_metrics - t_postprocess
                metrics_time = t_end - t_metrics
                total_time = t_end - t_load
                print(
                    f"[DEBUG] Batch {step} timing: "
                    f"load={load_time:.3f}s ({100*load_time/total_time:.1f}%) | "
                    f"preprocess={preprocess_time:.3f}s ({100*preprocess_time/total_time:.1f}%) | "
                    f"infer={infer_time:.3f}s ({100*infer_time/total_time:.1f}%) | "
                    f"postprocess={postprocess_time:.3f}s ({100*postprocess_time/total_time:.1f}%) | "
                    f"metrics={metrics_time:.3f}s ({100*metrics_time/total_time:.1f}%) | "
                    f"total={total_time:.3f}s",
                    flush=True,
                )

            if step % print_every == 0 or step == total_batches - 1:
                processed = min((step + 1) * config.batch_size, total_images)
                num_preds = sum(len(p["labels"]) for p in predictions_cpu)
                num_targets = sum(len(t["labels"]) for t in targets)
                print(
                    f"Step {step + 1}/{total_batches} "
                    f"({processed}/{total_images} images) "
                    f"- {num_preds} predictions, {num_targets} ground truth boxes",
                    flush=True,
                )

    aggregated = metric.compute_aggregated_values()
    metric_values = aggregated.metric_values

    bs = config.batch_size
    image_times = [t / bs for t in batch_times]
    tput_img = [bs / t for t in batch_times if t > 0]
    tput_batch = [1.0 / t for t in batch_times if t > 0]
    timing = BenchmarkTimingResult(
        batch_times_s=batch_times,
        total_s=sum(batch_times),
        statistics=BenchmarkStatistics(
            latency_batch_s=_compute_statistics(batch_times),
            latency_image_s=_compute_statistics(image_times),
            throughput_img_s=_compute_statistics(tput_img),
            throughput_batch_s=_compute_statistics(tput_batch),
        ),
    )

    model_name: str
    if isinstance(config.model, TaskModel):
        model_name = type(config.model).__name__
    else:
        model_name = str(config.model)

    result = BenchmarkResult(
        out=str(out_dir),
        model_name=model_name,
        backend_args=backend_args,
        dataset_format=data_args.format,
        num_images=total_images,
        batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        steps=config.steps,
        metric_values=metric_values,
        timing=timing,
    )

    # Save results.
    results_path = out_dir / "benchmark_results.json"
    results_path.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path = out_dir / "benchmark_summary.md"
    summary_path.write_text(result.to_markdown() + "\n", encoding="utf-8")

    result.print()

    return result


class _BenchmarkTransformArgs(ObjectDetectionTransformArgs):
    """Minimal transform args: only resize so images can be stacked by the collate
    function. The image tensor itself is not used for inference (predict_batch
    loads images from paths), but the DataLoader requires a uniform size."""

    channel_drop: None = None
    num_channels: int = 3
    photometric_distort: None = None
    random_zoom_out: None = None
    random_iou_crop: None = None
    random_flip: None = None
    random_rotate_90: None = None
    random_rotate: None = None
    image_size: tuple[int, int] = (64, 64)
    resize: ResizeArgs = Field(default_factory=lambda: ResizeArgs(height=64, width=64))
    normalize: None = None
    bbox_params: None = None


class _BenchmarkValBatch(TypedDict):
    image_path: list[str]
    targets: list[dict[str, Tensor]]


def _create_val_dataloader(
    data_args: TaskDataArgs,
    batch_size: int,
    num_workers: int,
) -> DataLoader[_BenchmarkValBatch]:
    val_dataset_args = data_args.get_val_args()
    dataset_cls = val_dataset_args.get_dataset_cls()
    image_info = list(val_dataset_args.list_image_info())
    transform = ObjectDetectionTransform(transform_args=_BenchmarkTransformArgs())
    dataset = dataset_cls(
        dataset_args=val_dataset_args,
        image_info=image_info,
        transform=transform,
    )
    inner_collate = ObjectDetectionCollateFunction(
        split="val",
        transform_args=_BenchmarkTransformArgs(),
    )

    def collate_fn(
        batch: list[Any],
    ) -> _BenchmarkValBatch:
        od_batch = inner_collate(batch)
        boxes_xyxy = _yolo_to_xyxy(od_batch["bboxes"])
        boxes_denorm = _denormalize_xyxy_boxes(boxes_xyxy, od_batch["original_size"])
        targets = [
            {"boxes": boxes, "labels": classes}
            for boxes, classes in zip(boxes_denorm, od_batch["classes"])
        ]
        return _BenchmarkValBatch(
            image_path=od_batch["image_path"],
            targets=targets,
        )

    return DataLoader(
        dataset=dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )


def _create_torch_pipeline(
    model: TaskModel,
) -> tuple[_PreprocessFn, _InferFn, _PostprocessFn]:
    """Create a preprocess/infer/postprocess pipeline using the PyTorch model."""

    def preprocess_fn(image_paths: list[str]) -> dict[str, Any]:
        tensors: list[Tensor] = []
        metadata: list[dict[str, Any]] = []
        for path in image_paths:
            x, meta = model.preprocess_image(path)
            tensors.append(x)
            metadata.append(meta)
        batch = torch.stack(tensors, dim=0)
        batch = model.preprocess_batch(batch)
        return {"batch": batch, "metadata": metadata}

    def infer_fn(preprocessed: dict[str, Any]) -> dict[str, Any]:
        raw = model.forward_backend(preprocessed["batch"])
        preprocessed["raw"] = raw
        return preprocessed

    def postprocess_fn(
        result: dict[str, Any],
        detection_threshold: float,
        top_k: int | None,
    ) -> list[dict[str, Tensor]]:
        preds = model.postprocess(
            result["raw"],
            result["metadata"],
            threshold=detection_threshold,
        )
        if top_k is not None:
            preds = [_filter_top_k(p, top_k) for p in preds]
        return [
            {
                "boxes": p["bboxes"],
                "scores": p["scores"],
                "labels": p["labels"],
            }
            for p in preds
        ]

    return preprocess_fn, infer_fn, postprocess_fn


_ONNX_PROVIDERS: dict[str, list[str]] = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
}


def _create_onnx_pipeline(
    model: TaskModel,
    out_dir: Path,
    export_args: dict[str, Any] | None,
    batch_size: int,
    provider: str = "cpu",
) -> tuple[_PreprocessFn, _InferFn, _PostprocessFn]:
    """Export the model to ONNX and create a preprocess/infer/postprocess pipeline."""
    import onnxruntime as ort

    onnx_path = out_dir / "model.onnx"
    export_kwargs = dict(export_args) if export_args else {}
    export_kwargs.setdefault("dynamic_batch_size", True)
    export_kwargs.setdefault("simplify", True)
    export_kwargs.setdefault("verify", False)
    model.export_onnx(out=onnx_path, **export_kwargs)  # type: ignore[operator]

    providers = _ONNX_PROVIDERS.get(provider, _ONNX_PROVIDERS["cpu"])
    available = set(ort.get_available_providers())
    missing = [p for p in providers if p not in available]
    if missing:
        raise RuntimeError(
            f"ONNX provider '{provider}' requires {missing} but only "
            f"{sorted(available)} are available."
        )
    provider_options = [
        {"trt_detailed_build_log": True} if p == "TensorrtExecutionProvider" else {}
        for p in providers
    ]
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0  # 0=Verbose, 1=Info, 2=Warning, 3=Error
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=providers,
        provider_options=provider_options,
    )

    # Verify the requested provider is actually being used (not silently fallen back).
    active_providers = session.get_providers()
    expected_provider = providers[0]
    if expected_provider not in active_providers:
        raise RuntimeError(
            f"ONNX provider '{provider}' failed to initialize. "
            f"Requested {expected_provider} but session is using {active_providers}. "
            f"Please verify that the required libraries are installed and configured."
        )

    def preprocess_fn(
        image_paths: list[str],
    ) -> dict[str, Any]:
        tensors: list[Tensor] = []
        metadata: list[dict[str, Any]] = []
        for path in image_paths:
            x, meta = model.preprocess_image(path)
            tensors.append(x)
            metadata.append(meta)
        batch = torch.stack(tensors, dim=0)
        batch = model.preprocess_batch(batch)
        return {
            "input_feed": {"images": batch.cpu().numpy()},
            "metadata": metadata,
            "model_h": batch.shape[-2],
            "model_w": batch.shape[-1],
        }

    def infer_fn(preprocessed: dict[str, Any]) -> dict[str, Any]:
        outputs = session.run(None, preprocessed["input_feed"])
        preprocessed["outputs"] = outputs
        return preprocessed

    def postprocess_fn(
        raw: dict[str, Any],
        detection_threshold: float,
        top_k: int | None,
    ) -> list[dict[str, Tensor]]:
        outputs = raw["outputs"]
        metadata = raw["metadata"]
        model_h = raw["model_h"]
        model_w = raw["model_w"]

        labels_batch = torch.from_numpy(outputs[0])
        boxes_batch = torch.from_numpy(outputs[1])
        scores_batch = torch.from_numpy(outputs[2])

        # The ONNX forward() rescales boxes to the model input size when
        # orig_target_size is not provided. Rescale to original image
        # coordinates.
        results: list[dict[str, Tensor]] = []
        for i in range(len(metadata)):
            orig_w = metadata[i]["orig_w"]
            orig_h = metadata[i]["orig_h"]
            boxes = boxes_batch[i].clone()
            boxes[:, 0] *= orig_w / model_w
            boxes[:, 1] *= orig_h / model_h
            boxes[:, 2] *= orig_w / model_w
            boxes[:, 3] *= orig_h / model_h

            scores = scores_batch[i]
            keep = scores > detection_threshold
            pred: dict[str, Tensor] = {
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels_batch[i][keep],
            }
            if top_k is not None:
                pred = _filter_top_k(pred, top_k)
            results.append(pred)
        return results

    return preprocess_fn, infer_fn, postprocess_fn


def _create_tensorrt_pipeline(
    model: TaskModel,
    out_dir: Path,
    export_args: dict[str, Any] | None,
    batch_size: int,
) -> tuple[_PreprocessFn, _InferFn, _PostprocessFn]:
    """Export the model to TensorRT and create a preprocess/infer/postprocess pipeline."""
    import tensorrt as trt

    engine_path = out_dir / "model.engine"
    export_kwargs = dict(export_args) if export_args else {}
    export_kwargs.setdefault("max_batchsize", batch_size)
    export_kwargs.setdefault("opt_batchsize", batch_size)
    export_kwargs.setdefault("min_batchsize", 1)
    export_kwargs.setdefault("verbose", False)
    # Disable ONNX verification by default to avoid numerical precision mismatches.
    onnx_args = export_kwargs.pop("onnx_args", {})
    onnx_args.setdefault("verify", False)
    export_kwargs["onnx_args"] = onnx_args
    model.export_tensorrt(out=engine_path, **export_kwargs)

    # Load TensorRT engine.
    trt_logger = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Get input/output tensor info (TensorRT 10.x API).
    input_name = "images"
    output_names = ["labels", "boxes", "scores"]

    # Get input shape (may have dynamic batch dimension = -1).
    input_shape = list(engine.get_tensor_shape(input_name))

    # Create CUDA stream for async execution.
    stream = torch.cuda.Stream()

    def preprocess_fn(image_paths: list[str]) -> dict[str, Any]:
        tensors: list[Tensor] = []
        metadata: list[dict[str, Any]] = []
        for path in image_paths:
            x, meta = model.preprocess_image(path)
            tensors.append(x)
            metadata.append(meta)
        batch = torch.stack(tensors, dim=0)
        batch = model.preprocess_batch(batch)
        return {
            "batch": batch.cuda().contiguous(),
            "metadata": metadata,
            "model_h": batch.shape[-2],
            "model_w": batch.shape[-1],
        }

    def infer_fn(preprocessed: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        batch = preprocessed["batch"]
        current_batch_size = batch.shape[0]

        # Set input shape for dynamic batch dimension.
        actual_input_shape = [current_batch_size] + input_shape[1:]
        context.set_input_shape(input_name, actual_input_shape)

        # Set input tensor address.
        context.set_tensor_address(input_name, batch.data_ptr())

        # Allocate output buffers and set their addresses.
        outputs: dict[str, Tensor] = {}
        for name in output_names:
            shape = list(context.get_tensor_shape(name))
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            torch_dtype = torch.from_numpy(np.zeros(1, dtype=dtype)).dtype
            outputs[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")
            context.set_tensor_address(name, outputs[name].data_ptr())

        # Run inference.
        context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        preprocessed["outputs"] = outputs
        return preprocessed

    def postprocess_fn(
        raw: dict[str, Any],
        detection_threshold: float,
        top_k: int | None,
    ) -> list[dict[str, Tensor]]:
        outputs = raw["outputs"]
        metadata = raw["metadata"]
        model_h = raw["model_h"]
        model_w = raw["model_w"]

        labels_batch = outputs["labels"].cpu()
        boxes_batch = outputs["boxes"].cpu()
        scores_batch = outputs["scores"].cpu()

        results: list[dict[str, Tensor]] = []
        for i in range(len(metadata)):
            orig_w = metadata[i]["orig_w"]
            orig_h = metadata[i]["orig_h"]
            boxes = boxes_batch[i].clone()
            boxes[:, 0] *= orig_w / model_w
            boxes[:, 1] *= orig_h / model_h
            boxes[:, 2] *= orig_w / model_w
            boxes[:, 3] *= orig_h / model_h

            scores = scores_batch[i]
            keep = scores > detection_threshold
            pred: dict[str, Tensor] = {
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels_batch[i][keep],
            }
            if top_k is not None:
                pred = _filter_top_k(pred, top_k)
            results.append(pred)
        return results

    return preprocess_fn, infer_fn, postprocess_fn


def _filter_by_threshold(
    pred: dict[str, Tensor], threshold: float
) -> dict[str, Tensor]:
    keep = pred["scores"] >= threshold
    return {k: v[keep] for k, v in pred.items()}


def _filter_top_k(pred: dict[str, Tensor], top_k: int) -> dict[str, Tensor]:
    scores = pred.get("scores", pred.get("bboxes"))
    if scores is None or len(scores) <= top_k:
        return pred
    _, topk_indices = pred["scores"].topk(min(top_k, len(pred["scores"])))
    return {k: v[topk_indices] for k, v in pred.items()}


def _to_cpu(
    predictions: list[dict[str, Tensor]],
) -> list[dict[str, Tensor]]:
    return [{k: v.detach().cpu() for k, v in p.items()} for p in predictions]


def _create_metric(
    *,
    metric_args: BenchmarkObjectDetectionMetricArgs,
    class_names: Sequence[str],
) -> ObjectDetectionTaskMetric:
    task_metric_args = ObjectDetectionTaskMetricArgs(
        classwise=metric_args.classwise,
        map=metric_args.map,
    )
    return ObjectDetectionTaskMetric(
        task_metric_args=task_metric_args,
        split="val",
        class_names=class_names,
        box_format="xyxy",
        loss_names=[],
    )


def _compute_statistics(values: list[float]) -> DescriptiveStatistics:
    if not values:
        return DescriptiveStatistics(min=0.0, max=0.0, mean=0.0, median=0.0, std=0.0)
    return DescriptiveStatistics(
        min=min(values),
        max=max(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        std=statistics.stdev(values) if len(values) >= 2 else 0.0,
    )


def _get_out_dir(out: PathLike, overwrite: bool) -> Path:
    out_dir = Path(out).resolve()
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"Output '{out_dir}' is not a directory!")
        if any(out_dir.iterdir()) and not overwrite:
            raise ValueError(
                f"Output '{out_dir}' is not empty! Set overwrite=True to overwrite "
                "the directory."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
