#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Literal, Sequence, TypedDict

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
    BenchmarkObjectDetectionConfig,
    BenchmarkObjectDetectionMetricArgs,
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
from lightly_train._transforms.transform import NormalizeArgs, ResizeArgs
from lightly_train.types import PathLike

_ALBUMENTATIONS_GE_1_4_5 = RequirementCache("albumentations>=1.4.5")
_ALBUMENTATIONS_GE_2_0_1 = RequirementCache("albumentations>=2.0.1")


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
    device: str | None = None,
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
    total_images = len(val_dataloader.dataset)  # type: ignore[arg-type]

    # Set up metric.
    class_names = list(data_args.included_classes.values())
    metric = _create_metric(metric_args=metric_args, class_names=class_names)

    # Determine device from backend if not specified.
    device = config.device
    if device is None:
        if isinstance(backend_args, ONNXBackendArgs):
            device = "cpu" if backend_args.provider == "cpu" else "cuda"
        elif isinstance(backend_args, TensorRTBackendArgs):
            device = "cuda"
        else:
            # Torch backend: default to cuda if available.
            device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up inference pipeline based on backend.
    torch_device = torch.device(device)
    backend: ObjectDetectionBackend
    if isinstance(backend_args, ONNXBackendArgs):
        backend = ONNXBackend(
            model=model,
            backend_args=backend_args,
            batch_size=config.batch_size,
            out_dir=out_dir,
            device=device,
        )
    elif isinstance(backend_args, TensorRTBackendArgs):
        backend = TensorRTBackend(
            model=model,
            backend_args=backend_args,
            batch_size=config.batch_size,
            out_dir=out_dir,
            device=device,
        )
    elif isinstance(backend_args, TorchBackendArgs):
        backend = TorchBackend(
            model=model, backend_args=backend_args, device=torch_device
        )
    else:
        raise ValueError(f"Unsupported backend: {type(backend_args).__name__}")

    # Warmup.
    if config.warmup_steps > 0:
        print(f"Running {config.warmup_steps} warmup steps...")
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                if step >= config.warmup_steps:
                    break
                backend.run_batch(batch)
        print("Warmup complete.")

    # Free cached memory before the timed benchmark loop so warmup
    # allocations don't skew memory or timing measurements.
    import gc

    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Run inference in batches.
    total_batches = len(val_dataloader)
    if config.steps is not None:
        total_batches = min(total_batches, config.steps)

    print_every = max(1, min(10, total_batches // 10))

    batch_times: list[float] = []
    _debug = config.debug
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            if step >= total_batches:
                break

            targets = batch["targets"]

            predictions, t_infer = backend.run_batch(batch=batch)
            batch_times.append(t_infer)
            predictions_cpu = _to_cpu(predictions)
            # Need to relabel for torchmetrics
            predictions_cpu = [
                {
                    "boxes": p["bboxes"],
                    "scores": p["scores"],
                    "labels": p["labels"],
                }
                for p in predictions_cpu
            ]
            metric.update_with_predictions(predictions_cpu, targets)

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

    model_name: str | None
    model_class: str
    if isinstance(config.model, TaskModel):
        model_name = getattr(config.model, "model_name", None)
        model_class = type(config.model).__name__
    else:
        model_name = str(config.model)
        model_class = "N/A"

    device_info = _get_device_info(device)

    result = BenchmarkResult(
        out=str(out_dir),
        model_name=model_name,
        model_class=model_class,
        backend_args=backend_args,
        device_info=device_info,
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


class _BenchmarkValTransformArgs(ObjectDetectionTransformArgs):
    """Val transform args that mirror the training validation pipeline.

    All augmentations are disabled. Only resize and normalize (plus bbox params)
    are applied so that the collate function produces images at the model's
    expected size with the correct normalisation.
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
    image_size: tuple[int, int] = tuple(init_args.get("image_size", (64, 64)))  # type: ignore[assignment]
    height, width = image_size

    # Resolve normalize the same way training val transforms do.
    normalize: NormalizeArgs | None
    raw_normalize = init_args.get("image_normalize", "none")
    if raw_normalize is None:
        normalize = None
    elif raw_normalize == "none":
        normalize = NormalizeArgs()
    else:
        assert isinstance(raw_normalize, dict)
        normalize = NormalizeArgs.from_dict(raw_normalize)

    num_channels = 3 if normalize is None else len(normalize.mean)

    return _BenchmarkValTransformArgs(
        image_size=image_size,
        resize=ResizeArgs(height=height, width=width),
        normalize=normalize,
        num_channels=num_channels,
    )


class _BenchmarkValBatch(TypedDict):
    image: Tensor
    image_path: list[str]
    original_size: list[tuple[int, int]]
    targets: list[dict[str, Tensor]]


class _BenchmarkCollateFunction:
    """Picklable collate function for benchmark dataloader."""

    def __init__(self, transform_args: _BenchmarkValTransformArgs) -> None:
        self._inner_collate = ObjectDetectionCollateFunction(
            split="val",
            transform_args=transform_args,
        )

    def __call__(self, batch: list[Any]) -> _BenchmarkValBatch:
        od_batch = self._inner_collate(batch)
        boxes_xyxy = _yolo_to_xyxy(od_batch["bboxes"])
        boxes_denorm = _denormalize_xyxy_boxes(boxes_xyxy, od_batch["original_size"])
        targets = [
            {"boxes": boxes, "labels": classes}
            for boxes, classes in zip(boxes_denorm, od_batch["classes"])
        ]
        return _BenchmarkValBatch(
            image=od_batch["image"],
            image_path=od_batch["image_path"],
            original_size=od_batch["original_size"],
            targets=targets,
        )


def _create_val_dataloader(
    data_args: TaskDataArgs,
    batch_size: int,
    num_workers: int,
    transform_args: _BenchmarkValTransformArgs,
) -> DataLoader[_BenchmarkValBatch]:
    val_dataset_args = data_args.get_val_args()
    dataset_cls = val_dataset_args.get_dataset_cls()
    image_info = list(val_dataset_args.list_image_info())
    transform = ObjectDetectionTransform(transform_args=transform_args)
    dataset = dataset_cls(
        dataset_args=val_dataset_args,
        image_info=image_info,
        transform=transform,
    )

    return DataLoader(
        dataset=dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=_BenchmarkCollateFunction(transform_args=transform_args),
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


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


def _get_device_info(device: str) -> DeviceInfo:
    """Collect information about the device used for benchmarking."""
    import os

    cpu_model = None
    cpu_cores = None
    cpu_threads = os.cpu_count()
    smt_enabled = None
    ram_gb = None

    # Try to get CPU model name and physical core count from /proc/cpuinfo (Linux)
    try:
        physical_ids = set()
        core_ids = set()
        with open("/proc/cpuinfo") as f:
            current_physical_id = None
            for line in f:
                if line.startswith("model name") and cpu_model is None:
                    cpu_model = line.split(":")[1].strip()
                elif line.startswith("physical id"):
                    current_physical_id = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":")[1].strip()
                    if current_physical_id is not None:
                        physical_ids.add(current_physical_id)
                        core_ids.add((current_physical_id, core_id))
        if core_ids:
            cpu_cores = len(core_ids)
            smt_enabled = cpu_threads is not None and cpu_threads > cpu_cores
    except Exception:
        pass

    # Try to check SMT status directly (Linux)
    if smt_enabled is None:
        try:
            with open("/sys/devices/system/cpu/smt/active") as f:
                smt_enabled = f.read().strip() == "1"
        except Exception:
            pass

    # Try to get total RAM from /proc/meminfo (Linux)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_kb = int(line.split()[1])
                    ram_gb = mem_kb / (1024**2)
                    break
    except Exception:
        pass

    if device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            smt_enabled=smt_enabled,
            ram_gb=ram_gb,
        )
    else:
        return CpuDeviceInfo(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            smt_enabled=smt_enabled,
            ram_gb=ram_gb,
        )
