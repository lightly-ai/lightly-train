#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Select TensorRT resolutions that maximize the detection Pareto frontier.

This script analyzes results produced by ``run_tensorrt_resolution_benchmark.py``.
It does not run inference or rebuild TensorRT engines.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

LatencyStatistic = Literal["min", "median", "mean"]

MODEL_RESOLUTIONS: dict[str, tuple[int, ...]] = {
    "dinov3/vitt16-ltdetr-coco": (480, 512, 544, 576, 608, 640),
    "dinov3/vitt16plus-ltdetr-coco": (512, 544, 576, 608, 640, 672),
    "dinov3/convnext-tiny-ltdetr-coco": (544, 576, 608, 640, 672, 704),
    "dinov3/convnext-small-ltdetr-coco": (576, 608, 640, 672, 704, 736),
    "dinov3/convnext-base-ltdetr-coco": (608, 640, 672, 704, 736, 768),
    "dinov3/convnext-large-ltdetr-coco": (640, 672, 704, 736, 768, 800),
}
BASELINE_RESOLUTION = 640
_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class BenchmarkPoint:
    """One model-resolution benchmark measurement."""

    model_name: str
    resolution: int
    map_5095: float
    latency_ms: float


@dataclass(frozen=True)
class OptimizationResult:
    """The best replacement configuration and its area improvement."""

    selected_points: tuple[BenchmarkPoint, ...]
    baseline_auc: float
    selected_auc: float
    latency_interval_ms: tuple[float, float]

    @property
    def auc_gain(self) -> float:
        return self.selected_auc - self.baseline_auc

    @property
    def average_map_gain(self) -> float:
        start, end = self.latency_interval_ms
        return self.auc_gain / (end - start)


def sanitize(model_name: str) -> str:
    return model_name.replace("/", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="out/tensorrt_resolution_benchmark",
        help="Root output directory created by the resolution benchmark.",
    )
    parser.add_argument(
        "--analysis-out",
        default=None,
        help="Directory for analysis artifacts. Defaults to <out>/pareto_analysis.",
    )
    parser.add_argument(
        "--latency-stat",
        choices=("min", "median", "mean"),
        default="min",
        help="Latency statistic used for the x-axis and optimization.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    return parser.parse_args()


def load_benchmark_points(
    out_root: Path, latency_stat: LatencyStatistic
) -> dict[str, tuple[BenchmarkPoint, ...]]:
    """Load and validate every expected benchmark result."""
    points_by_model: dict[str, tuple[BenchmarkPoint, ...]] = {}
    missing_paths: list[Path] = []
    invalid_paths: list[str] = []
    for model_name, resolutions in MODEL_RESOLUTIONS.items():
        points: list[BenchmarkPoint] = []
        for resolution in resolutions:
            path = (
                out_root
                / sanitize(model_name)
                / f"{resolution}x{resolution}"
                / "fp16"
                / "benchmark_results.json"
            )
            if not path.is_file():
                missing_paths.append(path)
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                map_5095 = float(payload["metric_values"]["val_metric/map"])
                latency_s = float(
                    payload["timing"]["statistics"]["latency_image_s"][latency_stat]
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                invalid_paths.append(f"{path}: {error}")
                continue
            if (
                not math.isfinite(map_5095)
                or not math.isfinite(latency_s)
                or latency_s <= 0
            ):
                invalid_paths.append(
                    f"{path}: mAP and latency must be finite and latency must be positive"
                )
                continue
            points.append(
                BenchmarkPoint(model_name, resolution, map_5095, latency_s * 1000)
            )
        points_by_model[model_name] = tuple(points)
    if missing_paths or invalid_paths:
        messages = ["Resolution Pareto analysis requires every expected result."]
        if missing_paths:
            messages.append("Missing results:\n" + "\n".join(map(str, missing_paths)))
        if invalid_paths:
            messages.append("Invalid results:\n" + "\n".join(invalid_paths))
        raise ValueError("\n\n".join(messages))
    return points_by_model


def dominates(left: BenchmarkPoint, right: BenchmarkPoint) -> bool:
    """Return whether ``left`` is at least as fast and accurate as ``right``."""
    return (
        left.latency_ms <= right.latency_ms
        and left.map_5095 >= right.map_5095
        and (left.latency_ms < right.latency_ms or left.map_5095 > right.map_5095)
    )


def pareto_frontier(points: Sequence[BenchmarkPoint]) -> list[BenchmarkPoint]:
    """Return non-dominated points ordered by increasing latency."""
    return sorted(
        (
            point
            for point in points
            if not any(dominates(other, point) for other in points)
        ),
        key=lambda point: (point.latency_ms, -point.map_5095, point.model_name),
    )


def best_map_at_latency(points: Sequence[BenchmarkPoint], latency_ms: float) -> float:
    """Return the best feasible mAP for a maximum latency budget."""
    return max(
        (point.map_5095 for point in points if point.latency_ms <= latency_ms),
        default=0.0,
    )


def envelope_auc(
    points: Sequence[BenchmarkPoint], interval_ms: tuple[float, float]
) -> float:
    """Integrate the best-feasible-mAP step envelope over a latency interval."""
    start, end = interval_ms
    if end <= start:
        raise ValueError("Latency interval must have positive width.")
    breakpoints = sorted(
        {point.latency_ms for point in points if start < point.latency_ms < end}
    )
    current = best_map_at_latency(points, start)
    previous = start
    area = 0.0
    for breakpoint in breakpoints:
        area += current * (breakpoint - previous)
        current = best_map_at_latency(points, breakpoint)
        previous = breakpoint
    return area + current * (end - previous)


def optimize_resolution_selection(
    points_by_model: dict[str, tuple[BenchmarkPoint, ...]],
) -> OptimizationResult:
    """Find the one-resolution-per-model selection with maximum frontier AUC."""
    model_names = tuple(MODEL_RESOLUTIONS)
    baseline = tuple(
        next(
            point
            for point in points_by_model[model_name]
            if point.resolution == BASELINE_RESOLUTION
        )
        for model_name in model_names
    )
    interval_ms = (
        min(point.latency_ms for point in baseline),
        max(point.latency_ms for point in baseline),
    )
    baseline_auc = envelope_auc(baseline, interval_ms)
    best_selection: tuple[BenchmarkPoint, ...] | None = None
    best_auc = -math.inf
    best_total_latency = math.inf
    for selection in itertools.product(
        *(points_by_model[name] for name in model_names)
    ):
        selected_auc = envelope_auc(selection, interval_ms)
        total_latency = sum(point.latency_ms for point in selection)
        if selected_auc > best_auc + _TIE_TOLERANCE:
            best_selection, best_auc, best_total_latency = (
                selection,
                selected_auc,
                total_latency,
            )
        elif math.isclose(selected_auc, best_auc, abs_tol=_TIE_TOLERANCE):
            assert best_selection is not None
            candidate_key = (
                total_latency,
                tuple(point.resolution for point in selection),
            )
            best_key = (
                best_total_latency,
                tuple(point.resolution for point in best_selection),
            )
            if candidate_key < best_key:
                best_selection, best_auc, best_total_latency = (
                    selection,
                    selected_auc,
                    total_latency,
                )
    assert best_selection is not None
    return OptimizationResult(best_selection, baseline_auc, best_auc, interval_ms)


def _step_coordinates(
    points: Sequence[BenchmarkPoint], interval_ms: tuple[float, float]
) -> tuple[list[float], list[float]]:
    start, end = interval_ms
    x_values = [start]
    x_values.extend(
        sorted({point.latency_ms for point in points if start < point.latency_ms < end})
    )
    x_values.append(end)
    return x_values, [best_map_at_latency(points, value) for value in x_values]


def write_plot(
    *,
    path: Path,
    baseline: Sequence[BenchmarkPoint],
    selected: Sequence[BenchmarkPoint],
    score_interval_ms: tuple[float, float],
    latency_stat: LatencyStatistic,
    result: OptimizationResult,
    dpi: int,
) -> None:
    """Write the baseline and optimized replacement frontier plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_points = [*baseline, *selected]
    plot_interval = (
        min(point.latency_ms for point in all_points),
        max(point.latency_ms for point in all_points),
    )
    baseline_x, baseline_y = _step_coordinates(baseline, plot_interval)
    selected_x, selected_y = _step_coordinates(selected, plot_interval)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axvspan(*score_interval_ms, color="0.93", zorder=0, label="AUC scoring range")
    ax.scatter(
        [point.latency_ms for point in baseline],
        [point.map_5095 for point in baseline],
        color="0.35",
        zorder=3,
        label="Native 640×640 points",
    )
    ax.step(
        baseline_x,
        baseline_y,
        where="post",
        color="0.35",
        linestyle="--",
        linewidth=2,
        label="Native 640×640 frontier",
    )
    ax.scatter(
        [point.latency_ms for point in selected],
        [point.map_5095 for point in selected],
        color="C0",
        marker="s",
        zorder=4,
        label="Selected-resolution points",
    )
    ax.step(
        selected_x,
        selected_y,
        where="post",
        color="C0",
        linewidth=2.5,
        label="Selected-resolution frontier",
    )
    for point in baseline:
        ax.annotate(
            point.model_name.rsplit("/", maxsplit=1)[-1],
            (point.latency_ms, point.map_5095),
            xytext=(4, 5),
            textcoords="offset points",
            color="0.35",
            fontsize=8,
        )
    for point in selected:
        ax.annotate(
            f"{point.model_name.rsplit('/', maxsplit=1)[-1]} ({point.resolution})",
            (point.latency_ms, point.map_5095),
            xytext=(4, -11),
            textcoords="offset points",
            color="C0",
            fontsize=8,
        )
    ax.set_xlabel(f"Latency (ms/image, {latency_stat})")
    ax.set_ylabel("COCO val mAP@0.5:0.95")
    ax.set_title(
        "TensorRT resolution selection: "
        f"ΔAUC = {result.auc_gain:.4f} mAP·ms "
        f"({result.average_map_gain:+.4f} average mAP)"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_artifacts(
    *,
    analysis_out: Path,
    points_by_model: dict[str, tuple[BenchmarkPoint, ...]],
    latency_stat: LatencyStatistic,
    result: OptimizationResult,
    dpi: int,
) -> None:
    """Write the selection report, data exports, and comparison plot."""
    analysis_out.mkdir(parents=True, exist_ok=True)
    baseline_by_model = {
        model_name: next(
            point
            for point in points_by_model[model_name]
            if point.resolution == BASELINE_RESOLUTION
        )
        for model_name in MODEL_RESOLUTIONS
    }
    selected_frontier = {
        (point.model_name, point.resolution)
        for point in pareto_frontier(result.selected_points)
    }
    rows: list[dict[str, object]] = []
    for selected in result.selected_points:
        baseline = baseline_by_model[selected.model_name]
        rows.append(
            {
                "model_name": selected.model_name,
                "baseline_resolution": BASELINE_RESOLUTION,
                "baseline_map_5095": baseline.map_5095,
                "baseline_latency_ms": baseline.latency_ms,
                "selected_resolution": selected.resolution,
                "selected_map_5095": selected.map_5095,
                "selected_latency_ms": selected.latency_ms,
                "map_5095_delta": selected.map_5095 - baseline.map_5095,
                "latency_ms_delta": selected.latency_ms - baseline.latency_ms,
                "on_selected_frontier": (selected.model_name, selected.resolution)
                in selected_frontier,
            }
        )
    payload = {
        "latency_statistic": latency_stat,
        "baseline_resolution": BASELINE_RESOLUTION,
        "score_latency_interval_ms": list(result.latency_interval_ms),
        "baseline_auc_map_ms": result.baseline_auc,
        "selected_auc_map_ms": result.selected_auc,
        "auc_gain_map_ms": result.auc_gain,
        "average_map_gain": result.average_map_gain,
        "selections": rows,
    }
    (analysis_out / "resolution_pareto_decision.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (analysis_out / "resolution_pareto_decision.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_rows = [
        "# TensorRT Resolution Pareto Decision",
        "",
        f"Latency statistic: `{latency_stat}`. The score compares replacement "
        f"frontiers over {result.latency_interval_ms[0]:.3f}–"
        f"{result.latency_interval_ms[1]:.3f} ms/image.",
        "",
        f"- Baseline AUC: {result.baseline_auc:.6f} mAP·ms",
        f"- Selected AUC: {result.selected_auc:.6f} mAP·ms",
        f"- Gain: {result.auc_gain:+.6f} mAP·ms "
        f"({result.average_map_gain:+.6f} average mAP)",
        "",
        "| Model | Native 640 mAP | Native 640 latency (ms) | Selected resolution | "
        "Selected mAP | Selected latency (ms) | ΔmAP | Δlatency (ms) | New frontier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        markdown_rows.append(
            f"| {row['model_name']} | {row['baseline_map_5095']:.4f} | "
            f"{row['baseline_latency_ms']:.3f} | {row['selected_resolution']}×"
            f"{row['selected_resolution']} | {row['selected_map_5095']:.4f} | "
            f"{row['selected_latency_ms']:.3f} | {row['map_5095_delta']:+.4f} | "
            f"{row['latency_ms_delta']:+.3f} | "
            f"{'yes' if row['on_selected_frontier'] else 'no'} |"
        )
    (analysis_out / "resolution_pareto_decision.md").write_text(
        "\n".join(markdown_rows) + "\n", encoding="utf-8"
    )
    write_plot(
        path=analysis_out / "resolution_pareto.png",
        baseline=tuple(baseline_by_model.values()),
        selected=result.selected_points,
        score_interval_ms=result.latency_interval_ms,
        latency_stat=latency_stat,
        result=result,
        dpi=dpi,
    )


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    latency_stat: LatencyStatistic = args.latency_stat
    points_by_model = load_benchmark_points(out_root, latency_stat)
    result = optimize_resolution_selection(points_by_model)
    analysis_out = (
        Path(args.analysis_out) if args.analysis_out else out_root / "pareto_analysis"
    )
    write_artifacts(
        analysis_out=analysis_out,
        points_by_model=points_by_model,
        latency_stat=latency_stat,
        result=result,
        dpi=args.dpi,
    )
    print(f"Wrote Pareto decision and plot to {analysis_out}")


if __name__ == "__main__":
    main()
