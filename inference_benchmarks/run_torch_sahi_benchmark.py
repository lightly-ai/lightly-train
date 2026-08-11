#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Benchmark the ltdetrv2 detection checkpoints with the compiled torch backend in
bf16-mixed precision on CUDA, under three amounts of SAHI tiling.

Produces a latency vs. mAP@0.5:0.95 plot with one curve per SAHI setting, each curve
running through the small, medium, and large model.

The dataset should have large images: SAHI tiles are cut at the model's image_size (640)
from the original image, so on a dataset whose images are not much larger than a tile
(COCO val2017, for example) every setting collapses to the same one or two tiles and the
curves become indistinguishable. VisDrone2019-DET is a good fit. Note that the class ids
of the data config must line up with the class ids of the COCO-pretrained checkpoints,
as the benchmark matches ground truth against predictions by id and does not remap them.

Usage:
    python run_torch_sahi_benchmark.py --data /path/to/visdrone/data.yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import lightly_train

MODEL_NAMES = [
    "ltdetrv2-s-coco",
    "ltdetrv2-m-coco",
    "ltdetrv2-l-coco",
]

# SAHI tiles are always cut at the model's image_size, so the overlap is the only knob
# controlling how many tiles an image is split into. On a 2000x1500 image with 640x640
# tiles this is roughly 0, 12, and 35 tiles per image.
SAHI_SETTINGS: list[tuple[str, dict[str, Any] | None]] = [
    ("no-sahi", None),
    ("sahi-0.2", {"overlap": 0.2}),
    ("sahi-0.6", {"overlap": 0.6}),
]

# Same threshold for every run so that the mAP values stay comparable across curves. A
# small non-zero value keeps the cross-tile merge tractable: at 0.0 the heaviest SAHI
# setting pushes ~10k boxes per image through NMS.
THRESHOLD = 0.05


@dataclass(frozen=True)
class BenchmarkPoint:
    """One (model, SAHI setting) run, reduced to the two plotted quantities."""

    model_name: str
    sahi_setting: str
    map_5095: float
    latency_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the data YAML file, in either yolo or coco format. Passed to "
        "benchmark_object_detection as is.",
    )
    parser.add_argument(
        "--dataset-name",
        default="VisDrone2019-DET val",
        help="Dataset name recorded in the benchmark reports.",
    )
    parser.add_argument(
        "--out",
        default="out/torch_sahi_benchmark",
        help="Root output directory for benchmark results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for inference. Note that with SAHI every image in the "
        "batch contributes one global row plus one row per tile.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        # With SAHI the number of tiles differs between images, so the compiled graph is
        # re-traced until torch.compile settles on a dynamic batch dimension. Warm up
        # generously to keep those recompilations out of the measured window.
        default=20,
        help="Number of warmup batches run before measuring.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Limit the number of batches processed per run. Defaults to the full val "
        "split. Useful for smoke-testing the sweep.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution of the latency vs. mAP plot.",
    )
    return parser.parse_args()


def sanitize(model_name: str) -> str:
    return model_name.replace("/", "_")


def short_label(model_name: str) -> str:
    """Return the model's size suffix, used to label points in the plot.

    Every curve passes through the same models, so the full name is repeated three
    times and the labels start overlapping.
    """
    return model_name.removeprefix("ltdetrv2-").removesuffix("-coco")


def format_summary_row(point: BenchmarkPoint, throughput: float) -> str:
    return (
        f"| {point.model_name} | {point.sahi_setting} | {point.map_5095:.4f} "
        f"| {point.latency_ms:.2f} | {throughput:.2f} |"
    )


def write_plot(*, path: Path, points: Sequence[BenchmarkPoint], dpi: int) -> None:
    """Write the latency vs. mAP plot, with one curve per SAHI setting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for index, (setting_name, _) in enumerate(SAHI_SETTINGS):
        # Sorted by latency so that the line reads from left to right. The models are
        # ordered by size, which is the same order in practice, but a heavier model is
        # not guaranteed to be slower.
        setting_points = sorted(
            (point for point in points if point.sahi_setting == setting_name),
            key=lambda point: point.latency_ms,
        )
        if not setting_points:
            continue
        color = f"C{index}"
        ax.plot(
            [point.latency_ms for point in setting_points],
            [point.map_5095 for point in setting_points],
            marker="o",
            color=color,
            label=setting_name,
        )
        for point in setting_points:
            ax.annotate(
                short_label(point.model_name),
                xy=(point.latency_ms, point.map_5095),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )

    ax.set_xlabel("Latency (ms/img)")
    ax.set_ylabel("Val mAP@0.5:0.95")
    ax.set_title("torch.compile, bf16-mixed, CUDA")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)

    points: list[BenchmarkPoint] = []
    summary_rows: list[str] = []
    for model_name in MODEL_NAMES:
        for setting_name, sahi_args in SAHI_SETTINGS:
            out_dir = out_root / sanitize(model_name) / setting_name
            print(
                f"\n=== Benchmarking {model_name} (torch, bf16-mixed, {setting_name}) ==="
            )
            try:
                result = lightly_train.benchmark_object_detection(
                    out=str(out_dir),
                    dataset_name=args.dataset_name,
                    model=model_name,
                    data=args.data,
                    batch_size=args.batch_size,
                    threshold=THRESHOLD,
                    warmup_steps=args.warmup_steps,
                    steps=args.steps,
                    device="cuda",
                    overwrite=True,
                    backend_args={
                        "format": "torch",
                        "compile": True,
                        "precision": "bf16-mixed",
                    },
                    sahi_args=sahi_args,
                )
            except Exception as e:
                print(f"FAILED: {model_name} ({setting_name}): {e}")
                continue
            point = BenchmarkPoint(
                model_name=model_name,
                sahi_setting=setting_name,
                map_5095=result.metric_values.get("val_metric/map", float("nan")),
                latency_ms=result.timing.statistics.latency_image_s.mean * 1000,
            )
            points.append(point)
            summary_rows.append(
                format_summary_row(
                    point, throughput=result.timing.statistics.throughput_img_s.mean
                )
            )

    summary_lines = [
        "# Torch SAHI Object Detection Benchmark Summary",
        "",
        f"Backend: torch, compiled, bf16-mixed, CUDA. Score threshold: {THRESHOLD}.",
        "",
        "| Model | SAHI | Val mAP@0.5:0.95 | Latency (ms/img) | Throughput (img/s) |",
        "| --- | --- | ---: | ---: | ---: |",
        *summary_rows,
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote summary to {summary_path}")

    if points:
        plot_path = out_root / "latency_vs_map.png"
        write_plot(path=plot_path, points=points, dpi=args.dpi)
        print(f"Wrote plot to {plot_path}")
    else:
        print("No successful runs, skipping the plot.")


if __name__ == "__main__":
    main()
