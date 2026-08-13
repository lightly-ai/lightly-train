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
    BenchmarkObjectDetectionBatch,
    BenchmarkObjectDetectionConfig,
    BenchmarkResult,
    BenchmarkSAHIArgs,
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
    ObjectDetectionPostprocessor,
    ObjectDetectionPreprocessor,
    ObjectDetectionSAHIConfig,
    targets_to_torchmetrics,
)
from lightly_train._task_models import task_model_helpers
from lightly_train._task_models.task_model import TaskModel
from lightly_train._transforms.task_transform import (
    TaskTransform,
    TaskTransformArgs,
)
from lightly_train.types import (
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
    sahi_args: dict[str, Any] | None = None,
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
            at or below this value are discarded. With ``sahi_args`` a non-zero threshold
            is strongly recommended: merging tiles runs non-maximum suppression over
            the surviving detections of every tile, which is slow when nothing is
            filtered out first.
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
        sahi_args:
            If given, benchmark Slicing Aided Hyper Inference instead of plain
            inference: every image larger than a single tile is tiled and the tile
            predictions are merged back the same way ``model.predict_sahi()`` does.
            Accepts ``tile_size``, ``overlap``, ``nms_iou_threshold``, and
            ``global_local_iou_threshold``, defaulting to the values ``predict_sahi()``
            uses, so ``{}`` enables tiling with those defaults. ``None`` (default)
            disables tiling. Note that ``tile_size`` defaults to half the model's input
            size, so tiles are magnified 2x and an image contributes several times more
            rows than it would with tiles cut at the model input size.

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
    _validate_sahi_backend(sahi_args=config.sahi_args, backend_args=backend_args)

    # Benchmark through the model's own pre- and postprocessing so that the reported
    # metrics describe the pipeline that runs at deployment.
    preprocessor, postprocessor = _get_pre_post_processors(model)
    sahi_config = (
        None if config.sahi_args is None else config.sahi_args.to_sahi_config()
    )

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
        sahi_config=sahi_config,
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
            postprocessor=postprocessor,
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
            postprocessor=postprocessor,
            threshold=config.threshold,
        )
    elif isinstance(backend_args, TorchBackendArgs):
        backend = TorchBackend(
            model=model,
            backend_args=backend_args,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
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
            # batch["classes"] holds internal, contiguous class ids (see
            # COCOObjectDetectionDatasetArgs.list_image_info), while predictions are
            # already mapped to user-facing class ids by the postprocessor. Map targets
            # into the same space so update_with_predictions compares like with like.
            internal_class_to_class = postprocessor.internal_class_to_class.cpu()
            targets = targets_to_torchmetrics(
                bboxes=batch["bboxes"],
                classes=[internal_class_to_class[c] for c in batch["classes"]],
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
        sahi_args=config.sahi_args,
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


def _get_pre_post_processors(
    model: TaskModel,
) -> tuple[ObjectDetectionPreprocessor, ObjectDetectionPostprocessor]:
    """Return the pre- and postprocessor the model uses for inference.

    Benchmarking through the model's own pair is what makes the reported metrics
    describe the deployed pipeline rather than a re-derived approximation of it: the
    dataloader prepares images exactly as ``predict()`` does, and every backend decodes
    its raw outputs exactly as ``predict()`` does.
    """
    preprocessor = getattr(model, "preprocessor", None)
    postprocessor = getattr(model, "postprocessor", None)
    if not isinstance(preprocessor, ObjectDetectionPreprocessor) or not isinstance(
        postprocessor, ObjectDetectionPostprocessor
    ):
        raise ValueError(
            f"Model '{type(model).__name__}' does not expose an "
            "ObjectDetectionPreprocessor and an ObjectDetectionPostprocessor and "
            "cannot be benchmarked."
        )
    return preprocessor, postprocessor


def _validate_sahi_backend(
    *, sahi_args: BenchmarkSAHIArgs | None, backend_args: BenchmarkBackendArgs
) -> None:
    """Reject SAHI on backends exported with a fixed batch size.

    With tiling an image occupies ``1 + num_tiles`` model input rows and the tile count
    depends on the image size, so the number of rows per batch varies from batch to
    batch. Only a graph with a dynamic batch dimension, and for TensorRT a profile whose
    upper bound covers the largest batch, can run that.

    Raises:
        ValueError: If the backend cannot run a varying number of input rows.
    """
    if sahi_args is None:
        return

    export_args = getattr(backend_args, "export_args", None) or {}

    if isinstance(backend_args, ONNXBackendArgs):
        if not export_args.get("dynamic_batch_size", True):
            raise ValueError(
                "sahi_args requires the ONNX backend to keep a dynamic batch "
                "dimension, but export_args sets dynamic_batch_size=False. Tiling "
                "makes the number of model input rows differ from batch to batch."
            )
    elif isinstance(backend_args, TensorRTBackendArgs):
        onnx_args = export_args.get("onnx_args") or {}
        if not onnx_args.get("dynamic_batch_size", True):
            raise ValueError(
                "sahi_args requires the TensorRT backend to keep a dynamic batch "
                "dimension, but export_args['onnx_args'] sets "
                "dynamic_batch_size=False. Tiling makes the number of model input "
                "rows differ from batch to batch."
            )
        if "max_batchsize" not in export_args:
            raise ValueError(
                "sahi_args requires an explicit export_args['max_batchsize'] for the "
                "TensorRT backend. It otherwise defaults to batch_size, which is an "
                "upper bound only without tiling: with tiling each image contributes "
                "one row plus one row per tile. Set it to at least "
                "batch_size * (1 + the largest tile count in the dataset)."
            )


class _BenchmarkTransformArgs(TaskTransformArgs):
    pass


class _BenchmarkTransform(TaskTransform):
    """Decode-only transform. All model preprocessing happens in the collate function.

    Ground truth boxes are passed through in normalized YOLO coordinates. They are
    independent of the model input size and are denormalized to the original image
    size by :func:`targets_to_torchmetrics` right before the metric update, so no
    rescale is needed here. Clipping to the image canvas and dropping degenerate
    boxes also happens there, not here.
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

    The metadata ``preprocess_image`` returns is kept on the batch instead of being
    discarded: it is what the backends hand to the postprocessor, and with tiling it is
    the only record of how many rows of ``image`` belong to which input image.
    """

    def __init__(
        self,
        preprocessor: ObjectDetectionPreprocessor,
        sahi_config: ObjectDetectionSAHIConfig | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.sahi_config = sahi_config

    def __call__(
        self, batch: list[ObjectDetectionDatasetItem]
    ) -> BenchmarkObjectDetectionBatch:
        prepared = [
            self.preprocessor.preprocess_image(
                # ObjectDetectionDatasetItem declares "image" as a numpy array, but
                # it holds whatever the transform returned, here a (C, H, W) tensor.
                cast(Tensor, item["image"]),
                device=torch.device("cpu"),
                dtype=torch.float32,
                sahi_config=self.sahi_config,
            )
            for item in batch
        ]
        return BenchmarkObjectDetectionBatch(
            image_path=[item["image_path"] for item in batch],
            # preprocess_image returns a (metadata.num_rows, C, H, W) stack per image:
            # a single row without tiling, one global row plus one row per tile with.
            image=torch.cat([rows for rows, _ in prepared]),
            metadata=[metadata for _, metadata in prepared],
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
    sahi_config: ObjectDetectionSAHIConfig | None = None,
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
        collate_fn=_BenchmarkCollateFunction(
            preprocessor=preprocessor, sahi_config=sahi_config
        ),
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
