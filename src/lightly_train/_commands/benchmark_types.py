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
from typing import Any, Literal, Union

from pydantic import ConfigDict, Field
from typing_extensions import Annotated

from lightly_train._configs.config import PydanticConfig
from lightly_train._data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._metrics.mean_average_precision import MeanAveragePrecisionArgs
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike


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
    model_name: str
    backend_args: BenchmarkBackendArgs
    dataset_format: str
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
        lines.append(f"- **Model**: {self.model_name}")
        ba = self.backend_args
        backend_str: str = ba.format
        if isinstance(ba, ONNXBackendArgs):
            backend_str = f"{ba.format} ({ba.provider})"
        lines.append(f"- **Backend**: {backend_str}")
        lines.append(f"- **Dataset**: {self.dataset_format} ({self.num_images} images)")
        lines.append(f"- **Batch Size**: {self.batch_size}")
        lines.append(f"- **Warmup Steps**: {self.warmup_steps}")
        steps_str = str(self.steps) if self.steps is not None else "all"
        lines.append(f"- **Steps**: {steps_str}")
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
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        console.print(Markdown(self.to_markdown()))


class BenchmarkObjectDetectionMetricArgs(PydanticConfig):
    """Metric arguments for object detection benchmarking."""

    detection_threshold: float = 0.6
    top_k: int | None = None
    map: MeanAveragePrecisionArgs | None = Field(
        default_factory=MeanAveragePrecisionArgs
    )
    # TODO check what this does
    classwise: bool = False


class TorchBackendArgs(PydanticConfig):
    """Backend arguments for PyTorch inference."""

    format: Literal["torch"] = "torch"


class ONNXBackendArgs(PydanticConfig):
    """Backend arguments for ONNX inference."""

    format: Literal["onnx"] = "onnx"
    provider: Literal["cpu", "cuda", "tensorrt"] = "cpu"
    export_args: dict[str, Any] | None = None


class TensorRTBackendArgs(PydanticConfig):
    """Backend arguments for TensorRT inference."""

    format: Literal["tensorrt"] = "tensorrt"
    export_args: dict[str, Any] | None = None


BenchmarkBackendArgs = Union[TorchBackendArgs, ONNXBackendArgs, TensorRTBackendArgs]


class BenchmarkObjectDetectionConfig(PydanticConfig):
    out: PathLike
    data: Annotated[
        Union[YOLOObjectDetectionDataArgs, COCOObjectDetectionDataArgs],
        Field(discriminator="format"),
    ]
    model: TaskModel | PathLike
    batch_size: int = 16
    warmup_steps: int = 0
    steps: int | None = None
    num_workers: int | Literal["auto"] = "auto"
    overwrite: bool = False
    metric_args: BenchmarkObjectDetectionMetricArgs = Field(
        default_factory=BenchmarkObjectDetectionMetricArgs
    )
    backend_args: Annotated[
        BenchmarkBackendArgs,
        Field(discriminator="format"),
    ] = Field(default_factory=TorchBackendArgs)

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
