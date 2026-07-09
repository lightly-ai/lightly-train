#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, TypedDict, Union

from pydantic import ConfigDict, Field, field_validator
from rich.console import Console
from rich.markdown import Markdown
from torch import Tensor
from typing_extensions import Annotated

from lightly_train._commands import data_helpers
from lightly_train._configs.config import PydanticConfig
from lightly_train._data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike


class ObjectDetectionPrediction(TypedDict):
    """Per-image object detection prediction."""

    bboxes: Tensor
    scores: Tensor
    labels: Tensor


class CpuDeviceInfo(PydanticConfig):
    """Information about CPU device used for benchmarking."""

    device_type: Literal["cpu"] = "cpu"
    cpu_model: str | None = None
    cpu_threads: int | None = None
    ram_gb: float | None = None


class CudaDeviceInfo(PydanticConfig):
    """Information about CUDA device used for benchmarking."""

    device_type: Literal["cuda"] = "cuda"
    # GPU fields
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None
    # System fields
    cpu_model: str | None = None
    cpu_threads: int | None = None
    ram_gb: float | None = None


DeviceInfo = Union[CpuDeviceInfo, CudaDeviceInfo]


class DescriptiveStatistics(PydanticConfig):
    """Descriptive statistics for a series of measurements."""

    min: float
    max: float
    mean: float
    median: float
    std: float


class BenchmarkStatistics(PydanticConfig):
    """All computed statistics from a benchmark run."""

    latency_batch_s: DescriptiveStatistics
    latency_image_s: DescriptiveStatistics
    throughput_img_s: DescriptiveStatistics
    throughput_batch_s: DescriptiveStatistics


class BenchmarkTimingResult(PydanticConfig):
    """Timing results from a benchmark run."""

    batch_times_s: list[float]
    total_s: float
    statistics: BenchmarkStatistics


class BenchmarkResult(PydanticConfig):
    """Result of a benchmark run."""

    out: str
    model_name: str | None
    model_class: str
    backend_args: BenchmarkBackendArgs
    device_info: Annotated[DeviceInfo, Field(discriminator="device_type")]
    dataset_name: str
    dataset_size: int
    num_images: int
    batch_size: int
    warmup_steps: int
    steps: int | None
    metric_values: dict[str, float]
    timing: BenchmarkTimingResult

    def to_markdown(self) -> str:
        """Return the benchmark report as a markdown string."""
        run_name = Path(self.out).name
        lines: list[str] = []

        lines.append(f"# Benchmark Report — {run_name}")
        lines.append("")

        # Run Config.
        lines.append("## Run Config")
        lines.append("")
        lines.append(f"- **Model**: {self.model_name or 'N/A'}")
        lines.append(f"- **Model class**: {self.model_class}")
        ba = self.backend_args
        backend_str: str = ba.format
        provider = getattr(ba, "provider", None)
        if ba.format == "onnx" and provider:
            backend_str = f"{ba.format} ({provider})"
        if getattr(ba, "compile", False):
            backend_str += ", compiled"
        # Show precision for all backends.
        precision = getattr(ba, "precision", None)
        if precision:
            backend_str += f", {precision}"
        lines.append(f"- **Backend**: {backend_str}")
        lines.append(
            f"- **Dataset**: {self.dataset_name} "
            f"({self.num_images}/{self.dataset_size} images)"
        )
        lines.append(f"- **Batch Size**: {self.batch_size}")
        lines.append(f"- **Warmup Steps**: {self.warmup_steps}")
        steps_str = str(self.steps) if self.steps is not None else "all"
        lines.append(f"- **Steps**: {steps_str}")
        lines.append("")

        # Device Info.
        lines.append("## Device Info")
        lines.append("")
        di = self.device_info
        if isinstance(di, CudaDeviceInfo):
            lines.append(f"- **GPU**: {di.gpu_name}")
            if di.gpu_memory_gb is not None:
                lines.append(f"- **GPU Memory**: {di.gpu_memory_gb:.1f} GB")
            if di.cuda_version:
                lines.append(f"- **CUDA**: {di.cuda_version}")
            if di.cudnn_version:
                lines.append(f"- **cuDNN**: {di.cudnn_version}")
        if di.cpu_model:
            lines.append(f"- **CPU**: {di.cpu_model}")
        if di.cpu_threads:
            lines.append(f"- **CPU Threads**: {di.cpu_threads}")
        if di.ram_gb is not None:
            lines.append(f"- **RAM**: {di.ram_gb:.1f} GB")
        lines.append("")

        # Performance Metrics.
        lines.append("## Performance Metrics")
        lines.append("")

        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for key, value in self.metric_values.items():
            if key.startswith("val_metric_classwise/"):
                continue
            label = _metric_key_to_label(key)
            lines.append(f"| {label} | {value:.4f} |")
        lines.append("")

        classwise = {
            k: v
            for k, v in self.metric_values.items()
            if k.startswith("val_metric_classwise/")
        }
        if classwise:
            lines.append("### Per-Class mAP")
            lines.append("")
            lines.append("| Class | mAP |")
            lines.append("| --- | ---: |")
            sorted_classes = sorted(classwise.items(), key=lambda x: x[1], reverse=True)
            for key, value in sorted_classes:
                class_name = key.split("/", 1)[1].removeprefix("map_")
                lines.append(f"| {class_name} | {value:.4f} |")
            lines.append("")

        # Throughput & Latency.
        lines.append("## Throughput & Latency")
        lines.append("")

        def _fmt(v: float, sig: int = 2) -> str:
            if v == 0:
                return f"{v:.2f}"
            decimals = max(2, sig - 1 - math.floor(math.log10(abs(v))))
            return f"{v:.{decimals}f}"

        def _stats_row(label: str, s: DescriptiveStatistics, scale: float = 1.0) -> str:
            return (
                f"| {label} "
                f"| {_fmt(s.min * scale)} "
                f"| {_fmt(s.max * scale)} "
                f"| {_fmt(s.mean * scale)} "
                f"| {_fmt(s.median * scale)} "
                f"| {_fmt(s.std * scale)} |"
            )

        lines.append("| | min | max | mean | median | std |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        s = self.timing.statistics
        lines.append(_stats_row("Throughput (img/s)", s.throughput_img_s))
        lines.append(_stats_row("Throughput (batch/s)", s.throughput_batch_s))
        lines.append(_stats_row("Latency (ms/img)", s.latency_image_s, scale=1000))
        lines.append(_stats_row("Latency (ms/batch)", s.latency_batch_s, scale=1000))
        lines.append("")
        lines.append(f"**Total time**: {self.timing.total_s:.2f}s")
        lines.append("")

        # Footer.
        results_path = Path(self.out) / "benchmark_results.json"
        lines.append("---")
        lines.append(f"Full report: `{results_path}`")

        return "\n".join(lines)

    def print(self) -> None:
        console = Console()
        console.print(Markdown(self.to_markdown()))


class TorchBackendArgs(PydanticConfig):
    """Backend arguments for PyTorch inference."""

    format: Literal["torch"] = "torch"
    compile: bool = False
    precision: Literal["fp32", "fp16-mixed", "bf16-mixed"] = "fp32"


class ONNXBackendArgs(PydanticConfig):
    """Backend arguments for ONNX inference."""

    format: Literal["onnx"] = "onnx"
    provider: Literal["cpu", "cuda", "tensorrt"] = "cpu"
    precision: Literal["fp32", "fp16"] = "fp32"
    export_args: dict[str, Any] | None = None


class TensorRTBackendArgs(PydanticConfig):
    """Backend arguments for TensorRT inference."""

    format: Literal["tensorrt"] = "tensorrt"
    precision: Literal["fp32", "fp16"] = "fp32"
    export_args: dict[str, Any] | None = None


BenchmarkBackendArgs = Union[TorchBackendArgs, ONNXBackendArgs, TensorRTBackendArgs]


class BenchmarkObjectDetectionConfig(PydanticConfig):
    out: PathLike
    dataset_name: str
    data: Annotated[
        Union[YOLOObjectDetectionDataArgs, COCOObjectDetectionDataArgs],
        Field(discriminator="format"),
    ]
    model: TaskModel | PathLike
    batch_size: int = Field(ge=1)
    threshold: float
    warmup_steps: int
    steps: int | None = Field(ge=1)
    num_workers: int | Literal["auto"]
    overwrite: bool
    device: str | None
    backend_args: Annotated[
        BenchmarkBackendArgs,
        Field(discriminator="format"),
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data", mode="before")
    @classmethod
    def _prepare_data(cls, v: Any) -> Any:
        # Keep data config handling consistent with train_object_detection.
        return data_helpers.prepare_object_detection_data(
            v, cls.model_fields["data"].annotation
        )


_SIZE_SUFFIXES = {"small", "medium", "large"}


def _metric_key_to_label(key: str) -> str:
    """Derive a human-readable label from a metric key.

    Examples:
        "val_metric/map"       -> "mAP@0.5:0.95"
        "val_metric/map_50"    -> "mAP@0.50"
        "val_metric/map_small" -> "mAP (small)"
        "val_metric/mar_100"   -> "mAR@100"
        "val_metric/mar_large" -> "mAR (large)"
    """
    name = key.split("/", 1)[-1]

    if name.startswith("map"):
        prefix = "mAP"
        suffix = name[3:].lstrip("_")
    elif name.startswith("mar"):
        prefix = "mAR"
        suffix = name[3:].lstrip("_")
    else:
        return name

    if not suffix:
        return f"{prefix}@0.5:0.95"
    if suffix in _SIZE_SUFFIXES:
        return f"{prefix} ({suffix})"
    if suffix.isdigit():
        iou = int(suffix)
        if iou <= 100 and prefix == "mAP":
            return f"{prefix}@0.{iou}"
        return f"{prefix}@{iou}"
    return f"{prefix}_{suffix}"
