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
from torch import Tensor

import lightly_train
from lightly_train._commands.benchmark_backends import TorchBackend
from lightly_train._commands.benchmark_task import (
    _create_val_dataloader,
    benchmark_object_detection,
)
from lightly_train._commands.benchmark_types import (
    BenchmarkObjectDetectionConfig,
    BenchmarkResult,
    BenchmarkSAHIArgs,
    ONNXBackendArgs,
    TensorRTBackendArgs,
    TorchBackendArgs,
)
from lightly_train._data.coco_object_detection_dataset import (
    COCOObjectDetectionDataArgs,
)
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._pre_post_processing.object_detection import (
    ObjectDetectionBatchOutput,
    ObjectDetectionMetadata,
    ObjectDetectionPostprocessor,
    ObjectDetectionPrediction,
    ObjectDetectionPreprocessor,
    ObjectDetectionSAHIConfig,
)
from lightly_train._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)
from lightly_train._task_models.task_model import TaskModel

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


class _FakeObjectDetectionPostprocessor(ObjectDetectionPostprocessor):
    """Records the metadata it is handed and returns one fixed detection per image."""

    def __init__(self) -> None:
        super().__init__(
            num_top_queries=1, internal_class_to_class=torch.tensor([0, 1])
        )
        self.last_metadata: Sequence[ObjectDetectionMetadata] | None = None

    def postprocess(  # type: ignore[override]
        self,
        raw: Any,
        metadata: Sequence[ObjectDetectionMetadata],
        threshold: float,
    ) -> list[ObjectDetectionPrediction]:
        self.last_metadata = metadata
        return [
            ObjectDetectionPrediction(
                labels=torch.tensor([0]),
                bboxes=torch.tensor([[10.0, 10.0, 40.0, 50.0]]),
                scores=torch.tensor([0.9]),
            )
            for _ in metadata
        ]


class _FakeObjectDetectionModel(TaskModel):
    """Minimal TaskModel subclass that returns fixed predictions."""

    model_suffix = ".pt"

    def __init__(self) -> None:
        super().__init__(
            init_args={
                "self": self,
                "__class__": type(self),
                "image_size": (64, 64),
            },
        )
        self.preprocessor = ObjectDetectionPreprocessor(
            image_size=(64, 64),
            image_normalize=None,
            expected_input_channels=3,
        )
        self.postprocessor = _FakeObjectDetectionPostprocessor()

    def forward(self, images: Tensor) -> ObjectDetectionBatchOutput:
        # The fake postprocessor ignores the outputs, they only have to be a valid
        # ObjectDetectionBatchOutput so the shared backend pipeline can carry them.
        num_rows = images.shape[0]
        return ObjectDetectionBatchOutput(
            logits=torch.zeros(num_rows, 1, 2), boxes=torch.zeros(num_rows, 1, 4)
        )


def _preprocessor() -> ObjectDetectionPreprocessor:
    return ObjectDetectionPreprocessor(
        image_size=(64, 64), image_normalize=None, expected_input_channels=3
    )


class TestValDataloader:
    def test_loads_data(self, tmp_path: Path) -> None:
        data_dict = _create_coco_data_dict(tmp_path)
        data_args = COCOObjectDetectionDataArgs.model_validate(data_dict)
        dataloader = _create_val_dataloader(
            data_args=data_args,
            batch_size=2,
            num_workers=0,
            preprocessor=_preprocessor(),
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
            preprocessor=_preprocessor(),
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
            preprocessor=_preprocessor(),
        )

        batch = next(iter(dataloader))
        assert batch["image"].shape[0] == 2
        assert len(batch["bboxes"]) == 2
        assert len(batch["classes"]) == 2

    def test_carries_preprocessing_metadata(self, tmp_path: Path) -> None:
        # The metadata the preprocessor records travels on the batch, so the backends
        # can hand it to the postprocessor instead of re-deriving it.
        data_dict = _create_coco_data_dict(tmp_path)
        data_args = COCOObjectDetectionDataArgs.model_validate(data_dict)
        dataloader = _create_val_dataloader(
            data_args=data_args,
            batch_size=2,
            num_workers=0,
            preprocessor=_preprocessor(),
        )

        batch = next(iter(dataloader))
        assert batch["metadata"] == [
            ObjectDetectionMetadata(orig_h=128, orig_w=128),
            ObjectDetectionMetadata(orig_h=128, orig_w=128),
        ]
        # Without tiling each image occupies exactly one model input row.
        assert all(item.num_rows == 1 for item in batch["metadata"])
        assert batch["image"].shape[0] == 2

    def test_sahi_tiles_images(self, tmp_path: Path) -> None:
        # With tiling an image occupies one global row plus one row per tile, and only
        # the metadata records how many.
        data_dict = _create_coco_data_dict(tmp_path)
        data_args = COCOObjectDetectionDataArgs.model_validate(data_dict)
        dataloader = _create_val_dataloader(
            data_args=data_args,
            batch_size=2,
            num_workers=0,
            preprocessor=_preprocessor(),
            sahi_config=ObjectDetectionSAHIConfig(
                overlap=0.2, nms_iou_threshold=0.3, global_local_iou_threshold=0.1
            ),
        )

        batch = next(iter(dataloader))
        assert len(batch["metadata"]) == 2
        for item in batch["metadata"]:
            assert item.tiling is not None
            assert item.tiling.num_tiles > 0
            assert item.num_rows == 1 + item.tiling.num_tiles
        assert batch["image"].shape[0] == sum(
            item.num_rows for item in batch["metadata"]
        )


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
        assert result.sahi_args is None
        # The metadata the backend hands to the postprocessor is the one the
        # preprocessor recorded in the dataloader, not a re-derived copy.
        assert model.postprocessor.last_metadata == [
            ObjectDetectionMetadata(orig_h=128, orig_w=128),
            ObjectDetectionMetadata(orig_h=128, orig_w=128),
        ]

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

    def test_benchmark_with_yaml_resolves_paths_relative_to_yaml(
        self, tmp_path: Path
    ) -> None:
        helpers.create_coco_object_detection_dataset(
            tmp_path / "dataset",
            num_files=2,
            height=128,
            width=128,
            num_classes=2,
            annotations_per_image=[
                [{"category_id": 0, "bbox": [10, 10, 30, 40]}],
                [{"category_id": 1, "bbox": [5, 5, 25, 35]}],
            ],
        )
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        data_yaml = config_dir / "data.yaml"
        with data_yaml.open("w") as file:
            yaml.safe_dump(
                {
                    "format": "coco",
                    "train": {
                        "annotations": "../dataset/train.json",
                        "images": "train",
                    },
                    "val": {
                        "annotations": "../dataset/val.json",
                        "images": "val",
                    },
                },
                file,
            )

        result = benchmark_object_detection(
            out=str(tmp_path / "out"),
            dataset_name="test-coco-yaml",
            data=data_yaml,
            model=_FakeObjectDetectionModel(),
            batch_size=2,
            overwrite=True,
        )

        assert result.num_images == 2

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

    def test_benchmark_with_sahi(self, tmp_path: Path) -> None:
        data_dict = _create_coco_data_dict(tmp_path)
        model = _FakeObjectDetectionModel()

        result = benchmark_object_detection(
            out=str(tmp_path / "out"),
            dataset_name="test-coco",
            data=data_dict,
            model=model,
            batch_size=2,
            threshold=0.5,
            sahi_args={"overlap": 0.2},
            overwrite=True,
        )

        assert result.sahi_args == BenchmarkSAHIArgs(overlap=0.2)
        # One prediction per input image, not per tile.
        assert result.num_images == 2
        metadata = model.postprocessor.last_metadata
        assert metadata is not None
        assert len(metadata) == 2
        assert all(item.tiling is not None for item in metadata)
        assert "val_metric/map" in result.metric_values

        # A tiled report must not be mistaken for an untiled one.
        summary = (tmp_path / "out" / "benchmark_summary.md").read_text()
        assert "**SAHI**: overlap 0.2" in summary

    def test_benchmark_accessible_from_lightly_train(self) -> None:
        assert hasattr(lightly_train, "benchmark_object_detection")


class TestSAHIBackendValidation:
    """SAHI needs a dynamic batch dimension: tiling makes the row count vary."""

    def _benchmark(self, tmp_path: Path, backend_args: Any) -> BenchmarkResult:
        return benchmark_object_detection(
            out=str(tmp_path / "out"),
            dataset_name="test-coco",
            data=_create_coco_data_dict(tmp_path),
            model=_FakeObjectDetectionModel(),
            batch_size=2,
            sahi_args={},
            backend_args=backend_args,
            overwrite=True,
        )

    def test_onnx_static_batch_size_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="dynamic batch dimension"):
            self._benchmark(
                tmp_path,
                ONNXBackendArgs(export_args={"dynamic_batch_size": False}),
            )

    def test_tensorrt_static_batch_size_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="dynamic batch dimension"):
            self._benchmark(
                tmp_path,
                TensorRTBackendArgs(
                    export_args={"onnx_args": {"dynamic_batch_size": False}}
                ),
            )

    def test_tensorrt_without_max_batchsize_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="max_batchsize"):
            self._benchmark(tmp_path, TensorRTBackendArgs())


class TestPrePostProcessingMatchesPredict:
    """The benchmark must measure the pipeline that runs at deployment.

    That is the point of routing it through the model's own preprocessor *and*
    postprocessor: its per-image predictions have to be identical to what the model's
    own predict methods return for the same images.
    """

    @staticmethod
    def _model() -> LTDETRObjectDetection:
        return LTDETRObjectDetection(
            model_name="dinov3/vitt16-notpretrained-ltdetr",
            classes={0: "class_0", 1: "class_1"},
            image_size=(256, 256),
            load_weights=False,
        )

    def test_benchmark_predictions_match_predict(self, tmp_path: Path) -> None:
        data_dict = _create_coco_data_dict(tmp_path)
        data_args = COCOObjectDetectionDataArgs.model_validate(data_dict)
        model = self._model()
        dataloader = _create_val_dataloader(
            data_args=data_args,
            batch_size=1,
            num_workers=0,
            preprocessor=model.preprocessor,
        )
        backend = TorchBackend(
            model=model,
            backend_args=TorchBackendArgs(),
            device=torch.device("cpu"),
            preprocessor=model.preprocessor,
            postprocessor=model.postprocessor,
            threshold=0.0,
        )

        num_compared = 0
        with torch.no_grad():
            for batch in dataloader:
                predictions, _ = backend.run_batch(batch=batch)
                expected = model.predict(batch["image_path"][0], threshold=0.0)

                torch.testing.assert_close(predictions[0].bboxes, expected.bboxes)
                torch.testing.assert_close(predictions[0].scores, expected.scores)
                torch.testing.assert_close(predictions[0].labels, expected.labels)
                num_compared += 1

        assert num_compared > 0

    def test_benchmark_sahi_predictions_match_predict_sahi(
        self, tmp_path: Path
    ) -> None:
        sahi_args = BenchmarkSAHIArgs()
        data_dict = _create_coco_data_dict(tmp_path)
        data_args = COCOObjectDetectionDataArgs.model_validate(data_dict)
        model = self._model()
        dataloader = _create_val_dataloader(
            data_args=data_args,
            batch_size=1,
            num_workers=0,
            preprocessor=model.preprocessor,
            sahi_config=sahi_args.to_sahi_config(),
        )
        backend = TorchBackend(
            model=model,
            backend_args=TorchBackendArgs(),
            device=torch.device("cpu"),
            preprocessor=model.preprocessor,
            postprocessor=model.postprocessor,
            threshold=0.5,
        )

        num_compared = 0
        with torch.no_grad():
            for batch in dataloader:
                predictions, _ = backend.run_batch(batch=batch)
                expected = model.predict_sahi(
                    batch["image_path"][0],
                    threshold=0.5,
                    overlap=sahi_args.overlap,
                    nms_iou_threshold=sahi_args.nms_iou_threshold,
                    global_local_iou_threshold=sahi_args.global_local_iou_threshold,
                )

                torch.testing.assert_close(predictions[0].bboxes, expected.bboxes)
                torch.testing.assert_close(predictions[0].scores, expected.scores)
                torch.testing.assert_close(predictions[0].labels, expected.labels)
                num_compared += 1

        assert num_compared > 0
