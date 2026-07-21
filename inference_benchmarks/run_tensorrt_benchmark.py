"""Benchmark all COCO-pretrained object detection checkpoints with the TensorRT
backend, in both fp32 and fp16, on the COCO val2017 split.

Usage:
    python run_tensorrt_benchmark.py --coco-root /path/to/coco
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import lightly_train

MODEL_NAMES = [
    "picodet-s-coco",
    "picodet-l-coco",
    "ltdetrv2-s-coco",
    "dinov3/vitt16-ltdetr-coco",
    "dinov3/vitt16plus-ltdetr-coco",
    "dinov3/vits16-ltdetr-coco",
    "dinov3/convnext-tiny-ltdetr-coco",
    "dinov3/convnext-small-ltdetr-coco",
    "dinov3/convnext-base-ltdetr-coco",
    "dinov3/convnext-large-ltdetr-coco",
    "dinov2/vits14-noreg-ltdetr-coco",
]

PRECISIONS: list[Literal["fp32", "fp16"]] = ["fp32", "fp16"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco-root",
        default="coco",
        help="Root directory of the COCO 2017 dataset, containing "
        "annotations/instances_{train,val}2017.json and the train2017/ and "
        "val2017/ image directories.",
    )
    parser.add_argument(
        "--out",
        default="out/tensorrt_benchmark",
        help="Root output directory for benchmark results.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used for inference.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup batches run before measuring.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Limit the number of batches processed per run. Defaults to the "
        "full val2017 split. Useful for smoke-testing the sweep.",
    )
    return parser.parse_args()


def build_data_config(coco_root: str) -> dict[str, Any]:
    root = Path(coco_root)
    return {
        "format": "coco",
        "train": {
            "annotations": str(root / "annotations" / "instances_train2017.json"),
            "images": str(root / "images" / "train2017"),
        },
        "val": {
            "annotations": str(root / "annotations" / "instances_val2017.json"),
            "images": str(root / "images" / "val2017"),
        },
    }


def sanitize(model_name: str) -> str:
    return model_name.replace("/", "_")


def format_summary_row(
    model_name: str, precision: str, result: lightly_train.BenchmarkResult
) -> str:
    map_5095 = result.metric_values.get("val_metric/map", float("nan"))
    latency_ms = result.timing.statistics.latency_image_s.mean * 1000
    throughput = result.timing.statistics.throughput_img_s.mean
    return (
        f"| {model_name} | {precision} | {map_5095:.4f} | {latency_ms:.2f} "
        f"| {throughput:.2f} |"
    )


def main() -> None:
    args = parse_args()
    data = build_data_config(args.coco_root)
    out_root = Path(args.out)

    summary_rows: list[str] = []
    for model_name in MODEL_NAMES:
        for precision in PRECISIONS:
            out_dir = out_root / sanitize(model_name) / precision
            print(f"\n=== Benchmarking {model_name} (tensorrt, {precision}) ===")
            try:
                result = lightly_train.benchmark_object_detection(
                    out=str(out_dir),
                    dataset_name="COCO val2017",
                    model=model_name,
                    data=data,
                    batch_size=args.batch_size,
                    warmup_steps=args.warmup_steps,
                    steps=args.steps,
                    device="cuda",
                    overwrite=True,
                    backend_args={"format": "tensorrt", "precision": precision},
                )
            except Exception as e:
                print(f"FAILED: {model_name} ({precision}): {e}")
                continue
            summary_rows.append(format_summary_row(model_name, precision, result))

    summary_lines = [
        "# TensorRT Object Detection Benchmark Summary",
        "",
        "| Model | Precision | Val mAP@0.5:0.95 | Latency (ms/img) | Throughput (img/s) |",
        "| --- | --- | ---: | ---: | ---: |",
        *summary_rows,
    ]
    summary_path = out_root / "summary.md"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
