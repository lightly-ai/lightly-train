#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Benchmark the ltdetrv2 detection checkpoints with the compiled torch backend in
fp32 precision on CUDA, under several amounts of SAHI tiling.

Produces a latency vs. mAP@0.5:0.95 plot with one curve per SAHI setting, each curve
running through the small, medium, and large model.

Two knobs control the tiling. ``tile_size`` is the region of the original image a tile
covers; the tile is resized to the model's image_size (640) before the forward pass, so a
tile_size below that magnifies the tile, which is where the gain on small objects comes
from. ``overlap`` controls how densely the tiles are laid down. Both increase the number
of tiles, and therefore the latency, so the sweep varies them separately.

The dataset must have images larger than a tile, otherwise an image is not tiled at all
and the setting collapses onto the untiled baseline. Pick ``--tile-sizes`` accordingly:
on COCO val2017, whose images are around 640x480, a 640 tile does nothing at all and only
sub-640 tiles tile anything.

For a yolo/coco data YAML the class ids must line up with the class ids of the
COCO-pretrained checkpoints, as the benchmark matches ground truth against predictions by
id and does not remap them. ``--coco-root`` builds the config for a COCO 2017 tree
directly and needs no remapping.

Usage:
    python run_torch_sahi_benchmark.py --data /path/to/visdrone/data.yaml
    python run_torch_sahi_benchmark.py --coco-root /path/to/coco --tile-sizes 320,160
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

# Swept per tile size. A tile is resized to the model's 640x640 input before the forward
# pass, so a 640 tile is a native-resolution crop that magnifies nothing (and does not
# tile an image at all unless it is larger than 640), while 320 magnifies every tile 2x.
DEFAULT_TILE_SIZES = [640, 320]
OVERLAPS = [0.2, 0.6]


def build_sahi_settings(
    tile_sizes: Sequence[int],
) -> list[tuple[str, dict[str, Any] | None]]:
    """Return the (name, sahi_args) sweep, untiled baseline first."""
    settings: list[tuple[str, dict[str, Any] | None]] = [("no-sahi", None)]
    for tile_size in tile_sizes:
        for overlap in OVERLAPS:
            settings.append(
                (
                    f"sahi-{tile_size}-{overlap}",
                    {"tile_size": (tile_size, tile_size), "overlap": overlap},
                )
            )
    return settings


# Same threshold for every run so that the mAP values stay comparable across curves. A
# small non-zero value keeps the cross-tile merge tractable: at 0.0 the heaviest SAHI
# setting pushes ~10k boxes per image through NMS.
THRESHOLD = 0.05


@dataclass(frozen=True)
class BenchmarkPoint:
    """One (model, SAHI setting) run, reduced to the quantities that get reported."""

    model_name: str
    sahi_setting: str
    map_5095: float
    latency_ms: float
    # Not plotted, but reported: tiling exists for the small objects, so the overall mAP
    # alone hides most of what a setting actually did.
    map_small: float
    map_large: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data",
        help="Path to the data YAML file, in either yolo or coco format. Passed to "
        "benchmark_object_detection as is.",
    )
    source.add_argument(
        "--coco-root",
        help="Root directory of a COCO 2017 tree, containing annotations/ and the "
        "train2017/val2017 image directories. The data config is built from it, so no "
        "data YAML and no class id remapping are needed.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name recorded in the benchmark reports. Defaults to "
        "'COCO val2017' with --coco-root and 'VisDrone2019-DET val' otherwise.",
    )
    parser.add_argument(
        "--tile-sizes",
        default=",".join(str(size) for size in DEFAULT_TILE_SIZES),
        help="Comma-separated square tile sizes to sweep, in pixels of the original "
        "image. Each is swept at every overlap in OVERLAPS. A tile size at or above the "
        "model input (640) magnifies nothing and only tiles images larger than it, so "
        f"on COCO val2017 use something below 640. Default: "
        f"{','.join(str(size) for size in DEFAULT_TILE_SIZES)}.",
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


def build_coco_data_config(coco_root: str) -> dict[str, Any]:
    """Return the data config for a COCO 2017 tree.

    Accepts both the ``<root>/images/val2017`` and the ``<root>/val2017`` layout, since
    the official zips extract to the latter.
    """
    root = Path(coco_root)
    images = root / "images" if (root / "images" / "val2017").is_dir() else root
    return {
        "format": "coco",
        "train": {
            "annotations": str(root / "annotations" / "instances_train2017.json"),
            "images": str(images / "train2017"),
        },
        "val": {
            "annotations": str(root / "annotations" / "instances_val2017.json"),
            "images": str(images / "val2017"),
        },
    }


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
        f"| {point.map_small:.4f} | {point.map_large:.4f} "
        f"| {point.latency_ms:.2f} | {throughput:.2f} |"
    )


def write_plot(
    *,
    path: Path,
    points: Sequence[BenchmarkPoint],
    sahi_settings: Sequence[tuple[str, dict[str, Any] | None]],
    dpi: int,
    metric: str = "map_5095",
    ylabel: str = "Val mAP@0.5:0.95",
) -> None:
    """Write a latency vs. metric plot, with one curve per SAHI setting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for index, (setting_name, _) in enumerate(sahi_settings):
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
            [getattr(point, metric) for point in setting_points],
            marker="o",
            color=color,
            label=setting_name,
        )
        for point in setting_points:
            ax.annotate(
                short_label(point.model_name),
                xy=(point.latency_ms, getattr(point, metric)),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )

    # The settings span more than an order of magnitude in latency (an untiled run
    # against a densely tiled, magnified one), which on a linear axis squashes every
    # cheap setting into the left edge.
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms/img, log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title("torch.compile, fp32, CUDA")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    if args.coco_root is not None:
        data: Any = build_coco_data_config(args.coco_root)
        dataset_name = args.dataset_name or "COCO val2017"
    else:
        data = args.data
        dataset_name = args.dataset_name or "VisDrone2019-DET val"
    tile_sizes = [int(size) for size in args.tile_sizes.split(",") if size.strip()]
    sahi_settings = build_sahi_settings(tile_sizes)

    points: list[BenchmarkPoint] = []
    summary_rows: list[str] = []
    for model_name in MODEL_NAMES:
        for setting_name, sahi_args in sahi_settings:
            out_dir = out_root / sanitize(model_name) / setting_name
            print(f"\n=== Benchmarking {model_name} (torch, fp32, {setting_name}) ===")
            try:
                result = lightly_train.benchmark_object_detection(
                    out=str(out_dir),
                    dataset_name=dataset_name,
                    model=model_name,
                    data=data,
                    batch_size=args.batch_size,
                    threshold=THRESHOLD,
                    warmup_steps=args.warmup_steps,
                    steps=args.steps,
                    device="cuda",
                    overwrite=True,
                    backend_args={
                        "format": "torch",
                        "compile": True,
                        "precision": "fp32",
                    },
                    sahi_args=sahi_args,
                )
            except Exception as e:
                print(f"FAILED: {model_name} ({setting_name}): {e}")
                continue
            metrics = result.metric_values
            point = BenchmarkPoint(
                model_name=model_name,
                sahi_setting=setting_name,
                map_5095=metrics.get("val_metric/map", float("nan")),
                latency_ms=result.timing.statistics.latency_image_s.mean * 1000,
                map_small=metrics.get("val_metric/map_small", float("nan")),
                map_large=metrics.get("val_metric/map_large", float("nan")),
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
        f"Dataset: {dataset_name}. Backend: torch, compiled, fp32, CUDA. "
        f"Score threshold: {THRESHOLD}.",
        "",
        "| Model | SAHI | Val mAP@0.5:0.95 | mAP small | mAP large "
        "| Latency (ms/img) | Throughput (img/s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *summary_rows,
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote summary to {summary_path}")

    if points:
        plot_path = out_root / "latency_vs_map.png"
        write_plot(
            path=plot_path,
            points=points,
            sahi_settings=sahi_settings,
            dpi=args.dpi,
        )
        print(f"Wrote plot to {plot_path}")
        # Tiling exists for the small objects, and the overall mAP averages that effect
        # away against the medium and large ones.
        small_plot_path = out_root / "latency_vs_map_small.png"
        write_plot(
            path=small_plot_path,
            points=points,
            sahi_settings=sahi_settings,
            dpi=args.dpi,
            metric="map_small",
            ylabel="Val mAP@0.5:0.95, small objects",
        )
        print(f"Wrote plot to {small_plot_path}")
    else:
        print("No successful runs, skipping the plot.")


if __name__ == "__main__":
    main()
