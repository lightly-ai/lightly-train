#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import gc
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from lightly_train._commands import common_helpers
from lightly_train._commands.benchmark_backends import (
    ObjectDetectionBackend,
    ONNXBackend,
    TensorRTBackend,
    TorchBackend,
)
from lightly_train._commands.benchmark_types import (
    BenchmarkBackendArgs,
    BenchmarkObjectDetectionConfig,
    BenchmarkResult,
    BenchmarkStatistics,
    BenchmarkTimingResult,
    CpuDeviceInfo,
    CudaDeviceInfo,
    DescriptiveStatistics,
    DeviceInfo,
    ONNXBackendArgs,
    TensorRTBackendArgs,
    TorchBackendArgs,
)
from lightly_train._configs import validate
from lightly_train._data import data_helpers as data_arg_helpers
from lightly_train._data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._metrics.detection.task_metric import (
    ObjectDetectionTaskMetric,
    ObjectDetectionTaskMetricArgs,
)
from lightly_train._pre_post_processing.object_detection import (
    ObjectDetectionPreprocessor,
    targets_to_torchmetrics,
)
from lightly_train._task_models import task_model_helpers
from lightly_train._task_models.task_model import TaskModel
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
)
from lightly_train.types import (
    ObjectDetectionBatch,
    ObjectDetectionDatasetItem,
    PathLike,
)


def benchmark_object_detection(
    *,
    out: PathLike,
    dataset_name: str,
    data: dict[str, Any] | PathLike,
    model: TaskModel | PathLike,
    batch_size: int = 1,
    threshold: float = 0.0,
    warmup_steps: int = 0,
    steps: int | None = None,
    num_workers: int | Literal["auto"] = "auto",
    overwrite: bool = False,
    device: str | None = None,
    backend_args: dict[str, Any] | BenchmarkBackendArgs | None = None,
) -> BenchmarkResult:
    """Benchmark an object detection model on a validation dataset.

    Runs inference on the validation split and computes mAP and other detection
    metrics.

    Args:
        out:
            Output directory where benchmark results are saved.
        dataset_name:
            Human-readable name for the dataset (e.g. ``"COCO 2017"``).
            Included in the benchmark report.
        data:
            The dataset configuration or path to a YAML file with the configuration
            (same format as train_object_detection). See the documentation for more information:
            https://docs.lightly.ai/train/stable/object_detection.html#data
        model:
            A loaded TaskModel instance or a path to an exported model file.
        batch_size:
            Number of images to process at once.
        threshold:
            Score threshold for filtering detections. Predictions with scores
            at or below this value are discarded.
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
        device:
            Device to run inference on (e.g. ``"cpu"``, ``"cuda"``). If
            ``None``, the device is auto-detected based on the backend
            configuration.
        backend_args:
            Backend configuration. Use ``format`` to select the backend:
            ``"torch"`` (default), ``"onnx"``, or ``"tensorrt"``. ONNX and
            TensorRT backends accept an optional ``export_args`` dict
            forwarded to ``model.export_onnx()``.

    Returns:
        BenchmarkResult containing metric values and timing statistics.
    """
    if backend_args is None:
        backend_args = {"format": "torch"}
    config = validate.pydantic_model_validate(BenchmarkObjectDetectionConfig, locals())
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

    backend_args = config.backend_args

    # Benchmark through the model's own preprocessing so that the reported metrics
    # describe the pipeline that runs at deployment.
    preprocessor = _get_preprocessor(model)

    # Set up validation data.
    data_arg_helpers.resolve_data_paths(config.data)
    data_args = config.data
    num_workers = common_helpers.get_num_workers(
        num_workers=config.num_workers, num_devices_per_node=1
    )
    val_dataloader = _create_val_dataloader(
        data_args=data_args,
        batch_size=config.batch_size,
        num_workers=num_workers,
        preprocessor=preprocessor,
    )
    num_batches = len(val_dataloader)
    if num_batches == 0:
        raise ValueError(
            f"Not enough images in the dataset for batch_size={config.batch_size}. "
            f"The dataset has {len(val_dataloader.dataset)} images."  # type: ignore[arg-type]
        )
    dataset_size = len(val_dataloader.dataset)  # type: ignore[arg-type]
    total_images = min(num_batches, config.steps or num_batches) * config.batch_size

    # Set up metric.
    class_names = list(data_args.included_classes.values())
    metric = _create_metric(class_names=class_names)

    device = _resolve_device(device=config.device, backend_args=backend_args)
    backend: ObjectDetectionBackend
    if isinstance(backend_args, ONNXBackendArgs):
        backend = ONNXBackend(
            model=model,
            backend_args=backend_args,
            batch_size=config.batch_size,
            out_dir=out_dir,
            device=str(device),
            preprocessor=preprocessor,
            threshold=config.threshold,
        )
    elif isinstance(backend_args, TensorRTBackendArgs):
        backend = TensorRTBackend(
            model=model,
            backend_args=backend_args,
            batch_size=config.batch_size,
            out_dir=out_dir,
            device=str(device),
            preprocessor=preprocessor,
            threshold=config.threshold,
        )
    elif isinstance(backend_args, TorchBackendArgs):
        backend = TorchBackend(
            model=model,
            backend_args=backend_args,
            device=device,
            preprocessor=preprocessor,
            threshold=config.threshold,
        )
    else:
        raise ValueError(f"Unsupported backend: {type(backend_args).__name__}")

    # Warmup. Cycle through the dataloader if warmup_steps exceeds the
    # number of batches.
    if config.warmup_steps > 0:
        print(f"Running {config.warmup_steps} warmup steps...")
        step = 0
        with torch.no_grad():
            while step < config.warmup_steps:
                for batch in val_dataloader:
                    if step >= config.warmup_steps:
                        break
                    backend.run_batch(batch)
                    step += 1
        print("Warmup complete.")

    # Free cached memory before the timed benchmark loop so warmup
    # allocations don't skew memory or timing measurements.
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Run inference in batches.
    total_batches = num_batches
    if config.steps is not None:
        total_batches = min(total_batches, config.steps)

    print_every = max(1, min(10, total_batches // 10))

    batch_times: list[float] = []
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            if step >= total_batches:
                break

            predictions, t_infer = backend.run_batch(batch=batch)
            batch_times.append(t_infer)

            metric_preds = [
                prediction.to(device="cpu").to_torchmetrics()
                for prediction in predictions
            ]
            targets = targets_to_torchmetrics(
                bboxes=batch["bboxes"],
                classes=batch["classes"],
                original_sizes=batch["original_size"],
            )
            metric.update_with_predictions(metric_preds, targets)

            if step % print_every == 0 or step == total_batches - 1:
                processed = min((step + 1) * config.batch_size, total_images)
                print(
                    f"Step {step + 1}/{total_batches} "
                    f"({processed}/{total_images} images)",
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

    model_name: str | None = getattr(model, "model_name", None)
    model_class = type(model).__name__

    device_info = _get_device_info(device=device)

    result = BenchmarkResult(
        out=str(out_dir),
        model_name=model_name,
        model_class=model_class,
        backend_args=backend_args,
        device_info=device_info,
        dataset_name=config.dataset_name,
        dataset_size=dataset_size,
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


def _get_preprocessor(model: TaskModel) -> ObjectDetectionPreprocessor:
    """Return the preprocessor the model uses for inference.

    Benchmarking through the model's own preprocessor is what makes the reported
    metrics describe the deployed pipeline rather than a re-derived approximation
    of it.
    """
    preprocessor = getattr(model, "preprocessor", None)
    if not isinstance(preprocessor, ObjectDetectionPreprocessor):
        raise ValueError(
            f"Model '{type(model).__name__}' does not expose an "
            "ObjectDetectionPreprocessor and cannot be benchmarked."
        )
    return preprocessor


class _BenchmarkTransformArgs(TaskTransformArgs):
    pass


class _BenchmarkTransform(TaskTransform):
    """Decode-only transform. All model preprocessing happens in the collate function.

    Ground truth boxes are passed through in normalized YOLO coordinates. They are
    independent of the model input size and are denormalized to the original image
    size by :func:`targets_to_torchmetrics` right before the metric update, so no
    box transform is needed here.
    """

    transform_args_cls = _BenchmarkTransformArgs

    def __init__(self) -> None:
        super().__init__(transform_args=_BenchmarkTransformArgs())

    def __call__(self, input: dict[str, Any]) -> dict[str, Any]:
        image = torch.from_numpy(np.ascontiguousarray(input["image"]))
        if image.ndim == 2:
            image = image.unsqueeze(-1)
        return {
            "image": image.permute(2, 0, 1),
            "bboxes": input["bboxes"],
            "class_labels": input["class_labels"],
        }


class _BenchmarkCollateFunction:
    """Run the model's per-image preprocessing and stack the batch.

    Mirrors the host-side/device-side split of ``TaskModel.predict_batch``:
    ``preprocess_image`` runs here (in the dataloader workers), while
    ``preprocess_batch`` runs on the device in the backend.
    """

    def __init__(self, preprocessor: ObjectDetectionPreprocessor) -> None:
        self.preprocessor = preprocessor

    def __call__(self, batch: list[ObjectDetectionDatasetItem]) -> ObjectDetectionBatch:
        images = [
            self.preprocessor.preprocess_image(
                # ObjectDetectionDatasetItem declares "image" as a numpy array, but
                # it holds whatever the transform returned, here a (C, H, W) tensor.
                cast(Tensor, item["image"]),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )[0]
            for item in batch
        ]
        return ObjectDetectionBatch(
            image_path=[item["image_path"] for item in batch],
            # preprocess_image returns a (1, C, H, W) stack per image.
            image=torch.cat(images),
            bboxes=[
                torch.from_numpy(item["bboxes"]).float().reshape(-1, 4)
                for item in batch
            ],
            classes=[torch.from_numpy(item["classes"]).long() for item in batch],
            original_size=[item["original_size"] for item in batch],
        )


def _create_val_dataloader(
    data_args: COCOObjectDetectionDataArgs | YOLOObjectDetectionDataArgs,
    batch_size: int,
    num_workers: int,
    preprocessor: ObjectDetectionPreprocessor,
) -> DataLoader[ObjectDetectionDatasetItem]:
    val_dataset_args = data_args.get_val_args()
    dataset_cls = val_dataset_args.get_dataset_cls()
    image_info = list(val_dataset_args.list_image_info())
    dataset = dataset_cls(
        dataset_args=val_dataset_args,
        image_info=image_info,
        transform=_BenchmarkTransform(),
    )

    return DataLoader(
        # ObjectDetectionDataset inherits Dataset[TaskDatasetItem] from
        # TaskDataset, so the type checker sees a type mismatch even though the
        # dataset actually yields ObjectDetectionDatasetItem at runtime.
        dataset=dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=_BenchmarkCollateFunction(preprocessor=preprocessor),
    )


def _create_metric(
    *,
    class_names: Sequence[str],
) -> ObjectDetectionTaskMetric:
    return ObjectDetectionTaskMetric(
        task_metric_args=ObjectDetectionTaskMetricArgs(),
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


def _resolve_device(
    *, device: str | torch.device | None, backend_args: BenchmarkBackendArgs
) -> torch.device:
    """Determine and validate the device for the given backend configuration.

    Args:
        device:
            Explicit device from the user, or None for auto-detection.
        backend_args:
            Backend configuration to validate against.

    Returns:
        Resolved torch device.

    Raises:
        ValueError: If the device is incompatible with the backend configuration.
    """
    requires_cuda = False
    reason = ""

    if isinstance(backend_args, TensorRTBackendArgs):
        requires_cuda = True
        reason = "TensorRT backend requires a CUDA device."
    elif isinstance(backend_args, ONNXBackendArgs):
        if backend_args.provider in ("cuda", "tensorrt"):
            requires_cuda = True
            reason = (
                f"ONNX backend with provider '{backend_args.provider}' requires a "
                "CUDA device."
            )
    elif isinstance(backend_args, TorchBackendArgs):
        if backend_args.precision in ("fp16", "bf16"):
            requires_cuda = True
            reason = (
                f"Torch backend with precision '{backend_args.precision}' requires a "
                "CUDA device."
            )

    if device is not None:
        resolved = torch.device(device)
        if requires_cuda and resolved.type != "cuda":
            raise ValueError(
                f"Device '{resolved}' is incompatible with the backend "
                f"configuration. {reason}"
            )
        return resolved

    # Auto-detect device.
    if requires_cuda:
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA is not available but is required. {reason}")
        return torch.device("cuda")

    # Default: prefer CUDA if available.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_device_info(device: torch.device) -> DeviceInfo:
    """Collect information about the device used for benchmarking."""
    import os
    import platform

    cpu_model = platform.processor() or None
    cpu_threads = os.cpu_count()

    ram_gb: float | None = None
    try:
        import psutil  # type: ignore[import-untyped]

        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    if device.type == "cuda":
        device_index = device.index or 0
        gpu_name = torch.cuda.get_device_name(device_index)
        gpu_memory_gb = torch.cuda.get_device_properties(device_index).total_memory / (
            1024**3
        )
        cuda_version = torch.version.cuda
        cudnn_version = (
            str(torch.backends.cudnn.version())  # type: ignore[no-untyped-call]
            if torch.backends.cudnn.is_available()  # type: ignore[no-untyped-call]
            else None
        )
        return CudaDeviceInfo(
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            cpu_model=cpu_model,
            cpu_threads=cpu_threads,
            ram_gb=ram_gb,
        )
    else:
        return CpuDeviceInfo(
            cpu_model=cpu_model,
            cpu_threads=cpu_threads,
            ram_gb=ram_gb,
        )
