#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from PIL.Image import Image as PILImage
from torch import Tensor

import lightly_train
from lightly_train._commands.benchmark_task import (
    _BenchmarkValTransformArgs,
    _create_val_dataloader,
    _to_cpu,
    benchmark_object_detection,
)
from lightly_train._commands.benchmark_types import (
    BenchmarkObjectDetectionConfig,
    BenchmarkResult,
    ObjectDetectionPrediction,
    TorchBackendArgs,
)
from lightly_train._data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._task_models.task_model import TaskModel
from lightly_train.types import PathLike

from .. import helpers


def _create_coco_data_dict(tmp_path: Path) -> dict[str, Any]:
    """Create a COCO dataset and return the data config dict."""
    helpers.create_coco_object_detection_dataset(
        tmp_path / "dataset",
        num_files=2,
        height=128,
        width=128,
        num_classes=2,
        annotations_per_image=[
            [
                {"category_id": 0, "bbox": [10, 10, 30, 40]},
                {"category_id": 1, "bbox": [50, 50, 20, 30]},
            ],
            [
                {"category_id": 0, "bbox": [5, 5, 25, 35]},
            ],
        ],
    )
    return {
        "format": "coco",
        "train": {
            "annotations": str(tmp_path / "dataset" / "train.json"),
            "images": "train",
        },
        "val": {
            "annotations": str(tmp_path / "dataset" / "val.json"),
            "images": "val",
        },
    }


class _FakeObjectDetectionModel(TaskModel):
    """Minimal TaskModel subclass that returns fixed predictions."""

    model_suffix = ".pt"
    _PRED = {
        "labels": torch.tensor([0]),
        "bboxes": torch.tensor([[10.0, 10.0, 40.0, 50.0]]),
        "scores": torch.tensor([0.9]),
    }

    def __init__(self) -> None:
        super().__init__(
            init_args={
                "self": self,
                "__class__": type(self),
                "image_size": (64, 64),
            },
        )

    def preprocess_image(
        self, image: PathLike | PILImage | Tensor
    ) -> tuple[Tensor, dict[str, Any]]:
        return torch.zeros(3, 64, 64), {"orig_h": 128, "orig_w": 128}

    def preprocess_batch(self, batch: Tensor) -> Tensor:
        return batch

    def forward_backend(self, x: Tensor) -> Any:
        return x

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Tensor]]:
        return [dict(self._PRED) for _ in metadata]


class TestValDataloader:
    def test_loads_data(self, tmp_path: Path) -> None:
        data_dict = _create_coco_data_dict(tmp_path)
        data_args = COCOObjectDetectionDataArgs.model_validate(data_dict)
        dataloader = _create_val_dataloader(
            data_args=data_args,
            batch_size=2,
            num_workers=0,
            transform_args=_BenchmarkValTransformArgs(),
        )

        batches = list(dataloader)
        assert len(batches) == 1
        batch = batches[0]
        assert len(batch["image_path"]) == 2
        assert all(Path(p).exists() for p in batch["image_path"])

        # First image has 2 annotations.
        assert batch["bboxes"][0].shape == (2, 4)
        assert batch["classes"][0].shape == (2,)

        # Second image has 1 annotation.
        assert batch["bboxes"][1].shape == (1, 4)
        assert batch["classes"][1].shape == (1,)

    def test_respects_ignore_classes(self, tmp_path: Path) -> None:
        data_dict = _create_coco_data_dict(tmp_path)
        data_dict["ignore_classes"] = [1]
        data_args = COCOObjectDetectionDataArgs.model_validate(data_dict)
        dataloader = _create_val_dataloader(
            data_args=data_args,
            batch_size=2,
            num_workers=0,
            transform_args=_BenchmarkValTransformArgs(),
        )

        batch = next(iter(dataloader))

        # First image: category_id=1 should be filtered, only category_id=0 remains.
        assert batch["bboxes"][0].shape == (1, 4)
        assert batch["classes"][0].tolist() == [0]

    def test_empty_annotations(self, tmp_path: Path) -> None:
        helpers.create_coco_object_detection_dataset(
            tmp_path / "dataset",
            num_files=2,
            height=128,
            width=128,
            annotations_per_image=[[], []],
        )
        data_dict = {
            "format": "coco",
            "train": {
                "annotations": str(tmp_path / "dataset" / "train.json"),
                "images": "train",
            },
            "val": {
                "annotations": str(tmp_path / "dataset" / "val.json"),
                "images": "val",
            },
        }
        data_args = COCOObjectDetectionDataArgs.model_validate(data_dict)
        dataloader = _create_val_dataloader(
            data_args=data_args,
            batch_size=2,
            num_workers=0,
            transform_args=_BenchmarkValTransformArgs(),
        )

        batch = next(iter(dataloader))
        assert batch["image"].shape[0] == 2
        assert len(batch["bboxes"]) == 2
        assert len(batch["classes"]) == 2


class TestToCpu:
    def test_moves_to_cpu(self) -> None:
        preds: list[ObjectDetectionPrediction] = [
            {
                "bboxes": torch.tensor([[1, 2, 3, 4]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        result = _to_cpu(preds)
        r = result[0]
        assert r["bboxes"].device.type == "cpu"
        assert r["scores"].device.type == "cpu"
        assert r["labels"].device.type == "cpu"


class TestBenchmarkObjectDetectionConfig:
    def test_validates_coco_data(self, tmp_path: Path) -> None:
        data_dict = _create_coco_data_dict(tmp_path)
        config = BenchmarkObjectDetectionConfig(
            out=str(tmp_path / "out"),
            dataset_name="test-coco",
            data=data_dict,  # type: ignore[arg-type]
            model=_FakeObjectDetectionModel(),
            batch_size=1,
            threshold=0.0,
            warmup_steps=0,
            steps=None,
            num_workers="auto",
            overwrite=False,
            device=None,
            backend_args=TorchBackendArgs(),
        )
        assert isinstance(config.data, COCOObjectDetectionDataArgs)

    def test_rejects_extra_fields(self, tmp_path: Path) -> None:
        data_dict = _create_coco_data_dict(tmp_path)
        with pytest.raises(Exception):
            BenchmarkObjectDetectionConfig(
                out=str(tmp_path / "out"),
                dataset_name="test-coco",
                data=data_dict,  # type: ignore[arg-type]
                model=_FakeObjectDetectionModel(),
                batch_size=1,
                threshold=0.0,
                warmup_steps=0,
                steps=None,
                num_workers="auto",
                overwrite=False,
                device=None,
                backend_args=TorchBackendArgs(),
                unknown_field="value",  # type: ignore[call-arg]
            )

    def test_defaults_format_to_yolo(self, tmp_path: Path) -> None:
        # A data dict without an explicit "format" defaults to "yolo", matching
        # train_object_detection.
        config = BenchmarkObjectDetectionConfig(
            out=str(tmp_path / "out"),
            dataset_name="test-yolo",
            data={  # type: ignore[arg-type]
                "path": str(tmp_path),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "class_a"},
            },
            model=_FakeObjectDetectionModel(),
            batch_size=1,
            threshold=0.0,
            warmup_steps=0,
            steps=None,
            num_workers="auto",
            overwrite=False,
            device=None,
            backend_args=TorchBackendArgs(),
        )
        assert isinstance(config.data, YOLOObjectDetectionDataArgs)
        assert config.data.format == "yolo"

    def test_loads_data_from_yaml_path(self, tmp_path: Path) -> None:
        # A path to a YAML file is loaded automatically and unknown keys are ignored.
        yaml_path = tmp_path / "data.yaml"
        with yaml_path.open("w") as file:
            yaml.safe_dump(
                {
                    "path": str(tmp_path),
                    "train": "images/train",
                    "val": "images/val",
                    "names": {0: "class_a"},
                    "unknown_key": "ignored",
                },
                file,
            )
        config = BenchmarkObjectDetectionConfig(
            out=str(tmp_path / "out"),
            dataset_name="test-yaml",
            data=str(yaml_path),  # type: ignore[arg-type]
            model=_FakeObjectDetectionModel(),
            batch_size=1,
            threshold=0.0,
            warmup_steps=0,
            steps=None,
            num_workers="auto",
            overwrite=False,
            device=None,
            backend_args=TorchBackendArgs(),
        )
        assert isinstance(config.data, YOLOObjectDetectionDataArgs)
        assert config.data.format == "yolo"
        assert config.data.names == {0: "class_a"}


class TestBenchmarkObjectDetectionE2E:
    def test_benchmark_with_fake_model(self, tmp_path: Path) -> None:
        data_dict = _create_coco_data_dict(tmp_path)
        model = _FakeObjectDetectionModel()

        result = benchmark_object_detection(
            out=str(tmp_path / "out"),
            dataset_name="test-coco",
            data=data_dict,
            model=model,
            batch_size=2,
            overwrite=True,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.model_name is None
        assert result.model_class == "_FakeObjectDetectionModel"
        assert result.backend_args.format == "torch"
        assert result.dataset_name == "test-coco"
        assert result.num_images == 2
        assert result.batch_size == 2
        assert result.warmup_steps == 0
        assert result.steps is None
        assert "val_metric/map" in result.metric_values
        assert isinstance(result.metric_values["val_metric/map"], float)

        # Check inference timing.
        timing = result.timing
        assert isinstance(timing.batch_times_s, list)
        assert all(t > 0 for t in timing.batch_times_s)
        assert timing.total_s > 0
        assert timing.statistics.latency_image_s.mean > 0

        # Check latency and throughput statistics.
        stats = timing.statistics
        assert stats.latency_batch_s.mean > 0
        assert stats.latency_batch_s.min > 0
        assert stats.latency_image_s.mean > 0
        assert stats.latency_image_s.min > 0
        assert stats.throughput_img_s.mean > 0
        assert stats.throughput_batch_s.mean > 0

        # Check results file was saved.
        results_path = tmp_path / "out" / "benchmark_results.json"
        assert results_path.exists()
        saved = json.loads(results_path.read_text())
        assert "metric_values" in saved
        assert "val_metric/map" in saved["metric_values"]
        assert "timing" in saved
        assert "batch_times_s" in saved["timing"]
        assert "statistics" in saved["timing"]
        assert "latency_batch_s" in saved["timing"]["statistics"]
        assert "throughput_img_s" in saved["timing"]["statistics"]

        # Check markdown summary was saved.
        summary_path = tmp_path / "out" / "benchmark_summary.md"
        assert summary_path.exists()
        summary = summary_path.read_text()
        assert "# Benchmark Report" in summary
        assert "mAP" in summary

    def test_output_dir_not_empty_raises(self, tmp_path: Path) -> None:
        data_dict = _create_coco_data_dict(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "existing_file.txt").write_text("content")

        with pytest.raises(ValueError, match="not empty"):
            benchmark_object_detection(
                out=str(out_dir),
                dataset_name="test-coco",
                data=data_dict,
                model=_FakeObjectDetectionModel(),
            )

    def test_overwrite_allows_non_empty_dir(self, tmp_path: Path) -> None:
        data_dict = _create_coco_data_dict(tmp_path)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "existing_file.txt").write_text("content")

        result = benchmark_object_detection(
            out=str(out_dir),
            dataset_name="test-coco",
            data=data_dict,
            model=_FakeObjectDetectionModel(),
            batch_size=2,
            overwrite=True,
        )
        assert "val_metric/map" in result.metric_values

    def test_benchmark_with_warmup(self, tmp_path: Path) -> None:
        data_dict = _create_coco_data_dict(tmp_path)
        model = _FakeObjectDetectionModel()

        result = benchmark_object_detection(
            out=str(tmp_path / "out"),
            dataset_name="test-coco",
            data=data_dict,
            model=model,
            batch_size=2,
            warmup_steps=1,
            overwrite=True,
        )

        assert result.warmup_steps == 1
        assert "val_metric/map" in result.metric_values
        assert result.timing.total_s > 0

    def test_benchmark_accessible_from_lightly_train(self) -> None:
        assert hasattr(lightly_train, "benchmark_object_detection")
