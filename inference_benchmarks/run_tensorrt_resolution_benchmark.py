#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Benchmark selected DINOv3 LTDETR checkpoints across trained resolutions.

Every run uses TensorRT FP16, batch size 1 by default, and COCO val2017. The
resolution windows are ordered by model capacity so smaller models are measured
at smaller input sizes and larger models at larger input sizes.

Usage:
    python run_tensorrt_resolution_benchmark.py --coco-root /path/to/coco
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

import lightly_train
from lightly_train._task_models import task_model_helpers
from lightly_train._task_models.task_model import TaskModel

MODEL_RESOLUTIONS: dict[str, tuple[int, ...]] = {
    "dinov3/vitt16-ltdetr-coco": (480, 512, 544, 576, 608, 640),
    "dinov3/vitt16plus-ltdetr-coco": (512, 544, 576, 608, 640, 672),
    "dinov3/convnext-tiny-ltdetr-coco": (544, 576, 608, 640, 672, 704),
    "dinov3/convnext-small-ltdetr-coco": (576, 608, 640, 672, 704, 736),
    "dinov3/convnext-base-ltdetr-coco": (608, 640, 672, 704, 736, 768),
    "dinov3/convnext-large-ltdetr-coco": (640, 672, 704, 736, 768, 800),
}


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
        default="out/tensorrt_resolution_benchmark",
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


def load_model_at_resolution(
    model_name: str, resolution: int, device: torch.device
) -> TaskModel:
    """Load a checkpoint and rebuild it with a different input resolution.

    The trained weights don't depend on ``image_size`` (decoder anchors and
    position embeddings are recomputed for the given size), so this loads the
    same weights into a model reconstructed with the target resolution.
    """
    ckpt_path = task_model_helpers.download_checkpoint(checkpoint=model_name)
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    checkpoint["model_init_args"]["image_size"] = (resolution, resolution)
    return task_model_helpers.init_model_from_checkpoint(
        checkpoint=checkpoint, device=device
    )


def format_summary_row(
    model_name: str, resolution: int, result: lightly_train.BenchmarkResult
) -> str:
    map_5095 = result.metric_values.get("val_metric/map", float("nan"))
    latency_ms = result.timing.statistics.latency_image_s.mean * 1000
    throughput = result.timing.statistics.throughput_img_s.mean
    return (
        f"| {model_name} | {resolution}x{resolution} | fp16 | {map_5095:.4f} | "
        f"{latency_ms:.2f} | {throughput:.2f} |"
    )


def main() -> None:
    args = parse_args()
    data = build_data_config(args.coco_root)
    out_root = Path(args.out)
    device = torch.device("cuda")

    summary_rows: list[str] = []
    for model_name, resolutions in MODEL_RESOLUTIONS.items():
        for resolution in resolutions:
            out_dir = (
                out_root / sanitize(model_name) / f"{resolution}x{resolution}" / "fp16"
            )
            print(
                f"\n=== Benchmarking {model_name} at {resolution}x{resolution} "
                "(tensorrt, fp16) ==="
            )
            try:
                model = load_model_at_resolution(
                    model_name=model_name, resolution=resolution, device=device
                )
                result = lightly_train.benchmark_object_detection(
                    out=str(out_dir),
                    dataset_name="COCO val2017",
                    model=model,
                    data=data,
                    batch_size=args.batch_size,
                    warmup_steps=args.warmup_steps,
                    steps=args.steps,
                    device="cuda",
                    overwrite=True,
                    backend_args={"format": "tensorrt", "precision": "fp16"},
                )
            except Exception as error:
                print(
                    f"FAILED: {model_name} ({resolution}x{resolution}, fp16): {error}"
                )
                continue
            summary_rows.append(format_summary_row(model_name, resolution, result))

    summary_lines = [
        "# TensorRT FP16 Multi-resolution Object Detection Benchmark Summary",
        "",
        "| Model | Input size | Precision | Val mAP@0.5:0.95 | Latency (ms/img) | Throughput (img/s) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
        *summary_rows,
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
