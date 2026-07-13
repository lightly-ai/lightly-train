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

import pytest
import torch
from lightning_utilities.core.imports import RequirementCache
from torch import nn

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._metrics.detection.task_metric import ObjectDetectionTaskMetricArgs
from lightly_train._task_models.picodet_object_detection.task_model import (
    PicoDetObjectDetection,
)
from lightly_train._task_models.picodet_object_detection.train_model import (
    PicoDetObjectDetectionTrain,
    PicoDetObjectDetectionTrainArgs,
)
from lightly_train._task_models.picodet_object_detection.transforms import (
    PicoDetObjectDetectionTrainTransformArgs,
    PicoDetObjectDetectionValTransformArgs,
)


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
    boxes, obj_logits, cls_logits = model(x)

    strides = model.o2o_head.strides
    num_preds = sum(math.ceil(416 / s) ** 2 for s in strides)
    assert boxes.shape == (1, num_preds, 4)
    assert obj_logits.shape == (1, num_preds)
    assert cls_logits.shape == (1, num_preds, 80)


def test_train_args_resolve_auto__adds_config_image_size() -> None:
    model_init_args: dict[str, object] = {}
    train_args = PicoDetObjectDetectionTrainArgs()
    train_args.resolve_auto(
        total_steps=1000,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name="picodet/l-640",
        model_init_args=model_init_args,
        data_args=YOLOObjectDetectionDataArgs(
            path=Path("/tmp/data"),
            train=Path("train") / "images",
            val=Path("val") / "images",
            names={0: "class_0"},
        ),
    )

    assert model_init_args["image_size"] == (640, 640)


@pytest.mark.parametrize(
    "transform_args",
    [PicoDetObjectDetectionTrainTransformArgs(), PicoDetObjectDetectionValTransformArgs()],
)
def test_transform_args_resolve_auto__requires_config_image_size(
    transform_args: PicoDetObjectDetectionTrainTransformArgs
    | PicoDetObjectDetectionValTransformArgs,
) -> None:
    with pytest.raises(ValueError, match="requires 'image_size' in model_init_args"):
        transform_args.resolve_auto(model_init_args={})


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
    assert input_batch_dim.dim_param == "N"

    inputs = np.random.randn(3, 3, 416, 416).astype(np.float32)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": inputs})
    onnx_labels, onnx_boxes, onnx_scores = (torch.from_numpy(o) for o in onnx_outputs)

    with torch.no_grad():
        boxes, obj_logits, cls_logits = model(torch.from_numpy(inputs))
        scores = torch.sigmoid(obj_logits)

    assert onnx_labels.shape == (3, boxes.shape[1])
    close_boxes = torch.isclose(onnx_boxes, boxes, atol=2e-2, rtol=1e-1)
    assert close_boxes.float().mean() > 0.95
    close_scores = torch.isclose(onnx_scores, scores, atol=2e-2, rtol=1e-1)
    assert close_scores.float().mean() > 0.95


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
) -> PicoDetObjectDetectionTrain:
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
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
        metric_args=ObjectDetectionTaskMetricArgs(),
        gradient_accumulation_steps=1,
    )
    return train_model
