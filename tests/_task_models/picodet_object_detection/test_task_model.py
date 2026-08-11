#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from lightning_fabric import Fabric
from lightning_utilities.core.imports import RequirementCache
from pytest_mock import MockerFixture
from torch import nn

from lightly_train import __version__
from lightly_train._data import label_helpers
from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._export.onnx_helpers import (
    _TORCH_DIM_HINTS_AVAILABLE,
    _TORCH_DIM_HINTS_MIN_VERSION,
)
from lightly_train._license import LICENSE_INFO
from lightly_train._metrics.detection.task_metric import ObjectDetectionTaskMetricArgs
from lightly_train._metrics.mean_average_precision import MeanAveragePrecision
from lightly_train._pre_post_processing.object_detection import (
    ObjectDetectionBatchOutput,
    ObjectDetectionMetadata,
)
from lightly_train._task_models.picodet_object_detection.task_model import (
    PicoDetObjectDetection,
)
from lightly_train._task_models.picodet_object_detection.train_model import (
    PicoDetObjectDetectionTrain,
    PicoDetObjectDetectionTrainArgs,
    _decode_predictions_for_metrics,
)
from lightly_train._task_models.picodet_object_detection.transforms import (
    PicoDetObjectDetectionTrainTransformArgs,
    PicoDetObjectDetectionValTransformArgs,
)
from lightly_train._visualize import object_detection
from lightly_train.types import ObjectDetectionBatch


def test_load_train_state_dict__from_exported() -> None:
    model_args = PicoDetObjectDetectionTrainArgs()
    train_model = _create_train_model(model_args)
    task_model = train_model.model
    state_dict = train_model.get_export_state_dict()
    task_model.load_train_state_dict(state_dict)


def test_load_train_state_dict__no_ema_weights() -> None:
    model_args = PicoDetObjectDetectionTrainArgs()
    train_model = _create_train_model(model_args)
    task_model = train_model.model
    state_dict = train_model.state_dict()
    # Drop all EMA weights from the state dict. This is for backwards compatibility
    # with older checkpoints. The model should still be able to load the weights by
    # copying the non-EMA weights to the EMA model.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ema_model.")}
    task_model.load_train_state_dict(state_dict)


def test_task_model_forward_shapes() -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(416, 416),
        num_classes=80,
        classes={i: f"class_{i}" for i in range(80)},
        image_normalize=None,
        load_weights=False,
    )

    x = torch.randn(1, 3, 416, 416)
    output = model(x)

    strides = model.o2o_head.strides
    num_preds = sum(math.ceil(416 / s) ** 2 for s in strides)
    assert isinstance(output, ObjectDetectionBatchOutput)
    assert output.logits.shape == (1, num_preds, 80)
    assert output.boxes.shape == (1, num_preds, 4)
    # Boxes are normalized cxcywh relative to the model input, logits are pre-sigmoid.
    assert output.boxes.min() >= 0.0
    assert output.boxes.max() <= 1.0
    assert output.logits.min() < 0.0


def test_postprocess__scales_normalized_boxes_to_original_size() -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(100, 200),
        num_classes=1,
        classes={0: "class_0"},
        image_normalize=None,
        # max_detections feeds num_top_queries, which top-ks over (anchor, class).
        max_detections=2,
        load_weights=False,
    )
    logits = torch.tensor([[[10.0], [-10.0]]])
    # Normalized cxcywh -> xyxy (0.1, 0.05, 0.5, 0.25).
    boxes = torch.tensor([[[0.3, 0.15, 0.4, 0.2], [0.0, 0.0, 0.0, 0.0]]])

    predictions = model.postprocess(
        ObjectDetectionBatchOutput(logits=logits, boxes=boxes),
        [ObjectDetectionMetadata(orig_h=200, orig_w=400)],
        threshold=0.5,
    )

    # Boxes are normalized, so they scale by the ORIGINAL size (400, 200) alone --
    # there is no model-input size term left in the postprocessing.
    torch.testing.assert_close(
        predictions[0]["bboxes"], torch.tensor([[40.0, 10.0, 200.0, 50.0]])
    )
    assert predictions[0].num_detections == 1


def test_postprocess__accepts_typed_and_mapping_outputs(mocker: MockerFixture) -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(256, 256),
        num_classes=1,
        classes={0: "class_0"},
        load_weights=False,
    )
    postprocess = mocker.patch.object(
        model.postprocessor, "postprocess", return_value=[]
    )
    logits = torch.rand(1, 2, 1)
    boxes = torch.rand(1, 2, 4)
    metadata = [ObjectDetectionMetadata(orig_h=480, orig_w=640)]

    typed_raw = ObjectDetectionBatchOutput(logits=logits, boxes=boxes)
    model.postprocess(typed_raw, metadata, threshold=0.5)
    call = postprocess.call_args.kwargs
    assert call["raw"] is typed_raw
    assert call["metadata"] is metadata
    assert call["threshold"] == 0.5

    # The benchmark backends pass the decoder key names, so the mapping branch must
    # keep accepting them and one code path can serve both detection models.
    model.postprocess(
        {"pred_logits": logits, "pred_boxes": boxes}, metadata, threshold=0.25
    )
    call = postprocess.call_args.kwargs
    assert isinstance(call["raw"], ObjectDetectionBatchOutput)
    assert call["raw"].logits is logits
    assert call["raw"].boxes is boxes
    assert call["threshold"] == 0.25


@pytest.mark.parametrize(
    "transform_args",
    [
        PicoDetObjectDetectionTrainTransformArgs(),
        PicoDetObjectDetectionValTransformArgs(),
    ],
)
def test_transform_args_resolve_auto__uses_model_config_image_size(
    transform_args: PicoDetObjectDetectionTrainTransformArgs
    | PicoDetObjectDetectionValTransformArgs,
) -> None:
    transform_args.resolve_auto(model_init_args={"model_name": "picodet/l-640"})

    assert transform_args.image_size == (640, 640)


@pytest.mark.parametrize(
    "transform_args",
    [
        PicoDetObjectDetectionTrainTransformArgs(),
        PicoDetObjectDetectionValTransformArgs(),
    ],
)
def test_transform_args_resolve_auto__requires_config_image_size(
    transform_args: PicoDetObjectDetectionTrainTransformArgs
    | PicoDetObjectDetectionValTransformArgs,
) -> None:
    with pytest.raises(ValueError, match="requires 'model_name' in model_init_args"):
        transform_args.resolve_auto(model_init_args={})


@pytest.mark.skipif(
    not _TORCH_DIM_HINTS_AVAILABLE,
    reason=f"torch >= {_TORCH_DIM_HINTS_MIN_VERSION} required",
)
def test_model_input_spec__uses_channels_and_image_size() -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(256, 320),
        num_classes=2,
        classes={1: "person", 0: "car"},
        image_normalize={"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
        load_weights=False,
    )

    spec = model.model_input_spec

    assert list(spec.input_specs) == ["images"]
    assert spec.input_specs["images"].shape == (3, 256, 320)
    assert spec.input_specs["images"].dtype == torch.float32
    assert spec.input_specs["images"].is_batched is True


def test_onnx_export_metadata() -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(256, 320),
        num_classes=2,
        classes={1: "person", 0: "car"},
        image_normalize={"mean": (0.0, 0.0, 0.0), "std": (1.0, 1.0, 1.0)},
        load_weights=False,
    )

    metadata = model.onnx_export_metadata()

    assert metadata["lightly_train_version"] == __version__
    assert metadata["license_info"] == LICENSE_INFO
    assert json.loads(metadata["image_normalize"]) == {
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    }
    # Insertion order defines the internal class indices, so it must be preserved.
    assert list(json.loads(metadata["classes"]).items()) == [
        ("1", "person"),
        ("0", "car"),
    ]
    assert metadata["model_name"] == "picodet/s-416"
    # The image size is carried by the ONNX input shape, not the metadata.
    assert "image_size" not in metadata


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx_has_no_nms(tmp_path: Path) -> None:
    import onnx

    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(416, 416),
        num_classes=80,
        classes={i: f"class_{i}" for i in range(80)},
        load_weights=False,
    )

    out = tmp_path / "picodet.onnx"
    model.export_onnx(out=out, simplify=False, verify=True)

    onnx_model = onnx.load(out)
    op_types = {node.op_type for node in onnx_model.graph.node}
    assert "NonMaxSuppression" not in op_types
    assert "If" not in op_types


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__dynamic_batch_size(tmp_path: Path) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(416, 416),
        num_classes=80,
        classes={i: f"class_{i}" for i in range(80)},
        load_weights=False,
    )

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, simplify=False, verify=True)

    onnx_model = onnx.load(out)
    input_batch_dim = onnx_model.graph.input[0].type.tensor_type.shape.dim[0]
    assert input_batch_dim.dim_param == "batch_size"
    # The graph returns raw logits and normalized boxes; selection happens outside.
    assert [output.name for output in onnx_model.graph.output] == ["logits", "boxes"]

    inputs = np.random.randn(3, 3, 416, 416).astype(np.float32)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_boxes = (
        torch.from_numpy(o) for o in session.run(None, {"images": inputs})
    )

    with torch.no_grad():
        torch_output = model(torch.from_numpy(inputs))

    assert onnx_logits.shape == torch_output.logits.shape
    assert onnx_boxes.shape == torch_output.boxes.shape
    # Compare logits as scores rather than raw values: the o2o peak filter suppresses
    # non-peaks with a large negative sentinel, so a single anchor whose peak status
    # differs between backends would dominate a raw-logit comparison.
    close_scores = torch.isclose(
        onnx_logits.sigmoid(), torch_output.logits.sigmoid(), atol=2e-2, rtol=1e-1
    )
    assert close_scores.float().mean() > 0.95
    close_boxes = torch.isclose(onnx_boxes, torch_output.boxes, atol=2e-2, rtol=1e-1)
    assert close_boxes.float().mean() > 0.95


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__static_batch_size(tmp_path: Path) -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(416, 416),
        num_classes=80,
        classes={i: f"class_{i}" for i in range(80)},
        load_weights=False,
    )

    out = tmp_path / "model.onnx"
    model.export_onnx(
        out=out, batch_size=3, dynamic_batch_size=False, simplify=False, verify=True
    )


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__rejects_shape_overrides(tmp_path: Path) -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(416, 416),
        num_classes=2,
        classes={0: "class_0", 1: "class_1"},
        load_weights=False,
    )

    with pytest.raises(
        ValueError,
        match="shape_overrides is not supported for PicoDet object detection.",
    ):
        model.export_onnx(
            out=tmp_path / "model.onnx", shape_overrides={"images": (3, None, None)}
        )


def test_predict_batch__composes_stages_in_order(mocker: MockerFixture) -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(256, 256),
        num_classes=2,
        classes={0: "class_0", 1: "class_1"},
        load_weights=False,
    )

    preprocess_image_spy = mocker.spy(model.preprocessor, "preprocess_image")
    preprocess_batch_spy = mocker.spy(model.preprocessor, "preprocess_batch")
    forward_spy = mocker.spy(model, "forward")
    postprocess_batch_spy = mocker.spy(model.postprocessor, "postprocess_batch")
    postprocess_image_spy = mocker.spy(model.postprocessor, "postprocess_image")
    postprocess_spy = mocker.spy(model.postprocessor, "postprocess")

    images = [torch.rand(3, 480, 640), torch.rand(3, 720, 1280)]
    result = model.predict_batch(images=images)

    # Per-image host preprocessing, then a single dense batch through the model.
    assert preprocess_image_spy.call_count == 2
    assert preprocess_batch_spy.call_count == 1
    (batch_in,) = preprocess_batch_spy.call_args.args
    assert batch_in.shape == (2, 3, 256, 256)

    assert forward_spy.call_count == 1
    (forward_in,) = forward_spy.call_args.args
    assert forward_in.shape == (2, 3, 256, 256)

    # postprocess receives forward's output and per-image metadata.
    assert postprocess_spy.call_count == 1
    call = postprocess_spy.call_args.kwargs
    assert call["raw"].logits is forward_spy.spy_return.logits
    assert call["raw"].boxes is forward_spy.spy_return.boxes
    assert call["metadata"] == [
        ObjectDetectionMetadata(orig_h=480, orig_w=640),
        ObjectDetectionMetadata(orig_h=720, orig_w=1280),
    ]
    assert result is postprocess_spy.spy_return

    # Postprocessing mirrors preprocessing: one dense pass over the batch, then one
    # host-side pass per image.
    assert postprocess_batch_spy.call_count == 1
    assert postprocess_image_spy.call_count == 2


def test_predict__matches_predict_batch() -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(256, 256),
        num_classes=2,
        classes={0: "class_0", 1: "class_1"},
        load_weights=False,
    )
    image = torch.rand(3, 480, 640)

    prediction = model.predict(image, threshold=0.0)
    (batch_prediction,) = model.predict_batch([image], threshold=0.0)

    torch.testing.assert_close(prediction.bboxes, batch_prediction.bboxes)
    torch.testing.assert_close(prediction.scores, batch_prediction.scores)
    torch.testing.assert_close(prediction.labels, batch_prediction.labels)


def test_predict__caps_detections_and_returns_original_coordinates() -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(256, 256),
        num_classes=2,
        classes={0: "class_0", 1: "class_1"},
        max_detections=25,
        load_weights=False,
    )

    prediction = model.predict(torch.rand(3, 480, 640), threshold=0.0)

    assert prediction.num_detections <= 25
    # Boxes are in original-image coordinates, not the (256, 256) model input.
    assert prediction.bboxes[:, 0].max() <= 640.0
    assert prediction.bboxes[:, 1].max() <= 480.0
    assert prediction.bboxes.min() >= 0.0


def test_predict_sahi_batch__splits_raw_outputs_per_image(
    mocker: MockerFixture,
) -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(64, 64),
        num_classes=2,
        classes={0: "class_0", 1: "class_1"},
        load_weights=False,
    )
    forward_spy = mocker.spy(model, "forward")
    postprocess_image_spy = mocker.spy(model.postprocessor, "postprocess_image")

    images = [torch.rand(3, 96, 96), torch.rand(3, 96, 96)]
    predictions = model.predict_sahi_batch(images, threshold=0.0, overlap=0.0)

    # All tiles of all images go through the model in a single forward pass.
    assert forward_spy.call_count == 1
    (forward_in,) = forward_spy.call_args.args
    total_tiles = forward_in.shape[0]

    # Each image is postprocessed separately from its own slice of the decoded batch.
    assert postprocess_image_spy.call_count == 2
    assert len(predictions) == 2
    seen = 0
    for call in postprocess_image_spy.call_args_list:
        batch_prediction, metadata, _ = call.args
        assert batch_prediction.labels.shape[0] == metadata.num_rows
        seen += metadata.num_rows
    assert seen == total_tiles


def test_predict_sahi_batch__matches_predict_sahi_per_image() -> None:
    """Batching images must not change the result of any single image."""
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(64, 64),
        num_classes=2,
        classes={0: "class_0", 1: "class_1"},
        load_weights=False,
    )
    # Different sizes, so the images contribute a different number of tiles each.
    images = [torch.rand(3, 96, 96), torch.rand(3, 150, 70)]

    batched = model.predict_sahi_batch(images, threshold=0.0, overlap=0.2)
    separate = [
        model.predict_sahi(image, threshold=0.0, overlap=0.2) for image in images
    ]

    assert len(batched) == 2
    for actual, expected in zip(batched, separate):
        torch.testing.assert_close(actual.labels, expected.labels)
        torch.testing.assert_close(actual.bboxes, expected.bboxes)
        torch.testing.assert_close(actual.scores, expected.scores)


def test_decode_predictions_for_metrics__non_contiguous_class_ids() -> None:
    """Validation metrics decode predictions into the internal class id space.

    Ground truth labels are remapped to internal contiguous ids by the dataset, and
    the metric indexes class_names by that internal id, so predictions must not be
    remapped to user-facing class ids here. See the train model's
    ``metric_class_mapping`` buffer.
    """
    # A non-contiguous user class map, as `data.names` would provide.
    classes = {0: "person", 5: "bus"}
    class_id_to_internal_class_id = (
        label_helpers.get_class_id_to_internal_class_id_mapping(
            class_ids=classes.keys(), ignore_classes=None
        )
    )
    gt_label = class_id_to_internal_class_id[5]
    assert gt_label == 1
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([gt_label]),
        }
    ]

    # One anchor predicts "bus" with a box exactly matching the ground truth.
    logits = torch.tensor([[[-20.0, 10.0], [-20.0, -20.0]]])
    boxes = torch.tensor([[[0.075, 0.075, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0]]])

    results = _decode_predictions_for_metrics(
        outputs=ObjectDetectionBatchOutput(logits=logits, boxes=boxes),
        orig_target_sizes=torch.tensor([[400, 400]]),
        num_top_queries=2,
        internal_class_to_class=torch.arange(len(classes)),
    )

    assert int(results[0]["labels"][0]) == gt_label

    metric = MeanAveragePrecision(
        prefix="val", class_names=list(classes.values()), class_metrics=True
    )
    metric.update(results, targets)
    computed = metric.compute()

    # A perfect prediction scores a perfect mAP, and classwise metrics resolve the
    # class name through the internal id without going out of range.
    assert float(computed["val/map"]) == 1.0
    assert float(computed["val_classwise/map_bus"]) == 1.0


def test_predict_batch__rejects_empty_input() -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(256, 256),
        num_classes=2,
        classes={0: "class_0", 1: "class_1"},
        load_weights=False,
    )

    with pytest.raises(ValueError, match="images must contain at least one image."):
        model.predict_batch([])


def test_predict_sahi_batch__rejects_empty_input() -> None:
    model = PicoDetObjectDetection(
        model_name="picodet/s-416",
        image_size=(256, 256),
        num_classes=2,
        classes={0: "class_0", 1: "class_1"},
        load_weights=False,
    )

    with pytest.raises(ValueError, match="images must contain at least one image."):
        model.predict_sahi_batch([])


def _is_module_frozen(m: nn.Module) -> bool:
    return all(not param.requires_grad for param in m.parameters())


@pytest.mark.parametrize("should_freeze", [True, False])
def test_freeze_backbone_on_set_train_mode(should_freeze: bool) -> None:
    model_args = PicoDetObjectDetectionTrainArgs(backbone_freeze=should_freeze)
    train_model = _create_train_model(model_args)
    task_model_backbone = train_model.model.backbone
    assert isinstance(task_model_backbone, nn.Module), "Backbone should be a nn.Module"

    train_model.set_train_mode()

    assert _is_module_frozen(task_model_backbone) == should_freeze, (
        f"Backbone should be frozen: {should_freeze}, but got frozen={_is_module_frozen(task_model_backbone)}"
    )
    assert not task_model_backbone.training == should_freeze, (
        "Backbone should be in eval mode after set_train_mode()"
    )


def _create_train_model(
    train_model_args: PicoDetObjectDetectionTrainArgs,
    names: dict[int, str] | None = None,
    metric_args: ObjectDetectionTaskMetricArgs | None = None,
) -> PicoDetObjectDetectionTrain:
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"} if names is None else names,
    )
    train_model_args.resolve_auto(
        total_steps=1000,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name="picodet/s-416",
        model_init_args={},
        data_args=data_args,
    )
    train_transform_args = PicoDetObjectDetectionTrainTransformArgs()
    train_transform_args.resolve_auto(model_init_args={"image_size": (416, 416)})
    val_transform_args = PicoDetObjectDetectionValTransformArgs()
    val_transform_args.resolve_auto(model_init_args={"image_size": (416, 416)})

    train_model = PicoDetObjectDetectionTrain(
        model_name="picodet/s-416",
        model_args=train_model_args,
        data_args=data_args,
        train_transform_args=train_transform_args,
        val_transform_args=val_transform_args,
        load_weights=False,
        metric_args=(
            ObjectDetectionTaskMetricArgs() if metric_args is None else metric_args
        ),
        gradient_accumulation_steps=1,
    )
    return train_model


def _detection_batch() -> ObjectDetectionBatch:
    return {
        "image_path": ["a.jpg", "b.jpg"],
        "image": torch.rand(2, 3, 416, 416),
        "bboxes": [
            torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            torch.tensor([[0.3, 0.3, 0.1, 0.1]]),
        ],
        # Ground truth labels are internal contiguous ids, as the dataset produces.
        "classes": [torch.tensor([1]), torch.tensor([0])],
        # ObjectDetectionBatch stores (width, height).
        "original_size": [(640, 480), (800, 600)],
    }


@pytest.mark.parametrize("train_metrics", [True, False])
def test_train_and_validation_step__non_contiguous_class_ids(
    train_metrics: bool,
) -> None:
    """Both steps run end to end and report metrics in original-image coordinates."""
    train_model = _create_train_model(
        PicoDetObjectDetectionTrainArgs(),
        # Non-contiguous class ids, so a wrong class remap would be visible.
        names={0: "person", 5: "bus"},
        metric_args=ObjectDetectionTaskMetricArgs(train=train_metrics),
    )
    fabric = Fabric(accelerator="cpu", devices=1)
    fabric.launch()
    batch = _detection_batch()

    train_model.set_train_mode()
    train_result = train_model.training_step(fabric=fabric, batch=batch, step=0)
    assert torch.isfinite(train_result.loss)

    train_model.eval()
    with torch.no_grad():
        val_result = train_model.validation_step(fabric=fabric, batch=batch, step=0)
    assert torch.isfinite(val_result.loss)

    visualization = val_result.visualization
    assert isinstance(
        visualization, object_detection.ObjectDetectionTaskStepVisualization
    )
    results = visualization.results
    assert results is not None
    assert len(results) == 2
    for result, (orig_w, orig_h) in zip(results, batch["original_size"]):
        # Predictions stay in the internal class id space the targets use, so they
        # index the metric's class names without going out of range.
        assert int(result["labels"].min()) >= 0
        assert int(result["labels"].max()) < 2
        # Boxes are in original-image pixels, not the (416, 416) model input.
        assert float(result["boxes"][:, 2].max()) <= orig_w + 1
        assert float(result["boxes"][:, 3].max()) <= orig_h + 1
    # The metric resolves without an out-of-range class name lookup.
    metric_values = train_model.val_metrics.compute_aggregated_values().metric_values
    assert any("map" in key for key in metric_values)
