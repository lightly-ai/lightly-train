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
from typing import Any, Literal

import torch
from albumentations import BboxParams
from lightning_utilities.core.imports import RequirementCache
from pydantic import Field
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
    ObjectDetectionPrediction,
    ONNXBackendArgs,
    TensorRTBackendArgs,
    TorchBackendArgs,
)
from lightly_train._configs import validate
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
from lightly_train._task_models import task_model_helpers
from lightly_train._task_models.object_detection_components.utils import (
    _denormalize_xyxy_boxes,
    _yolo_to_xyxy,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train._transforms.ltdetr_transforms.object_detection import (
    LTDETRObjectDetectionCollateFunction,
    LTDETRObjectDetectionTransform,
    LTDETRObjectDetectionTransformArgs,
)
from lightly_train._transforms.transform import NormalizeArgs, ResizeArgs
from lightly_train.types import (
    ObjectDetectionDatasetItem,
    PathLike,
)

_ALBUMENTATIONS_GE_1_4_5 = RequirementCache("albumentations>=1.4.5")
_ALBUMENTATIONS_GE_2_0_1 = RequirementCache("albumentations>=2.0.1")


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

    # Build val transform args from the model.
    transform_args = _build_val_transform_args(model=model)

    # Set up validation data.
    data_args = config.data
    num_workers = common_helpers.get_num_workers(
        num_workers=config.num_workers, num_devices_per_node=1
    )
    val_dataloader = _create_val_dataloader(
        data_args=data_args,
        batch_size=config.batch_size,
        num_workers=num_workers,
        transform_args=transform_args,
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
            threshold=config.threshold,
        )
    elif isinstance(backend_args, TensorRTBackendArgs):
        backend = TensorRTBackend(
            model=model,
            backend_args=backend_args,
            batch_size=config.batch_size,
            out_dir=out_dir,
            device=str(device),
            threshold=config.threshold,
        )
    elif isinstance(backend_args, TorchBackendArgs):
        backend = TorchBackend(
            model=model,
            backend_args=backend_args,
            device=device,
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
            predictions_cpu = _to_cpu(predictions)

            # Convert predictions from "bboxes" to "boxes" for torchmetrics.
            metric_preds: list[dict[str, Tensor]] = [
                {
                    "boxes": p["bboxes"],
                    "scores": p["scores"],
                    "labels": p["labels"],
                }
                for p in predictions_cpu
            ]
            # Convert ground truth boxes from YOLO format to denormalized xyxy.
            boxes_xyxy = _yolo_to_xyxy(batch["bboxes"])
            boxes_denorm = _denormalize_xyxy_boxes(boxes_xyxy, batch["original_size"])
            targets = [
                {"boxes": boxes, "labels": classes}
                for boxes, classes in zip(boxes_denorm, batch["classes"])
            ]
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


def _make_val_bbox_params() -> BboxParams:
    """Create standard YOLO bbox params matching training val transforms."""
    return BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_width=0.0,
        min_height=0.0,
        **(dict(filter_invalid_bboxes=True) if _ALBUMENTATIONS_GE_2_0_1 else {}),
        **(dict(clip=True) if _ALBUMENTATIONS_GE_1_4_5 else {}),
    )


class _BenchmarkValTransformArgs(LTDETRObjectDetectionTransformArgs):
    """Val transform args that mirror the training validation pipeline.

    All augmentations are disabled. Only resize and normalize (plus bbox params)
    are applied so that the collate function produces images at the model's
    expected size with the correct normalization.
    """

    channel_drop: None = None
    num_channels: int = 3
    photometric_distort: None = None
    random_zoom_out: None = None
    random_iou_crop: None = None
    random_flip: None = None
    random_rotate_90: None = None
    random_rotate: None = None
    image_size: tuple[int, int] = (640, 640)
    resize: ResizeArgs = Field(
        default_factory=lambda: ResizeArgs(height=640, width=640)
    )
    normalize: NormalizeArgs | None = None
    bbox_params: BboxParams = Field(default_factory=_make_val_bbox_params)


def _build_val_transform_args(model: TaskModel) -> _BenchmarkValTransformArgs:
    """Build val transform args from the model's ``init_args``.

    Uses ``image_size`` and ``image_normalize`` stored in the model to
    construct a transform that matches the training validation pipeline.

    Args:
        model: A loaded task model instance.

    Returns:
        Transform args configured for the model.
    """
    init_args = model.init_args
    if "image_size" not in init_args:
        raise ValueError(
            "Model does not specify 'image_size' in init_args. Cannot build "
            "validation transforms without a known image size."
        )
    image_size: tuple[int, int] = tuple(init_args["image_size"])  # type: ignore[assignment]
    height, width = image_size

    # Resolve normalize the same way training val transforms do.
    normalize: NormalizeArgs | None
    raw_normalize = init_args.get("image_normalize", "none")
    if raw_normalize is None:
        normalize = None
    elif raw_normalize == "none":
        normalize = NormalizeArgs()
    else:
        if not isinstance(raw_normalize, dict):
            raise ValueError(
                f"Expected 'image_normalize' to be a dict, got {type(raw_normalize).__name__}."
            )
        normalize = NormalizeArgs.from_dict(raw_normalize)

    num_channels = 3 if normalize is None else len(normalize.mean)

    return _BenchmarkValTransformArgs(
        image_size=image_size,
        resize=ResizeArgs(height=height, width=width),
        normalize=normalize,
        num_channels=num_channels,
    )


def _create_val_dataloader(
    data_args: COCOObjectDetectionDataArgs | YOLOObjectDetectionDataArgs,
    batch_size: int,
    num_workers: int,
    transform_args: _BenchmarkValTransformArgs,
) -> DataLoader[ObjectDetectionDatasetItem]:
    val_dataset_args = data_args.get_val_args()
    dataset_cls = val_dataset_args.get_dataset_cls()
    image_info = list(val_dataset_args.list_image_info())
    transform = LTDETRObjectDetectionTransform(transform_args=transform_args)
    dataset = dataset_cls(
        dataset_args=val_dataset_args,
        image_info=image_info,
        transform=transform,
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
        collate_fn=LTDETRObjectDetectionCollateFunction(
            split="val",
            transform_args=transform_args,
        ),
    )


def _to_cpu(
    predictions: list[ObjectDetectionPrediction],
) -> list[ObjectDetectionPrediction]:
    return [{k: v.detach().cpu() for k, v in p.items()} for p in predictions]  # type: ignore[attr-defined,misc]


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
