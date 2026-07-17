#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json

import torch

from lightly_train import __version__
from lightly_train._license import LICENSE_INFO
from lightly_train._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)


def _model() -> LTDETRObjectDetection:
    return LTDETRObjectDetection(
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        classes={0: "car", 1: "person"},
        image_size=(256, 320),
        image_normalize={"mean": (0.0,), "std": (1.0,)},
        backbone_args={"in_chans": 1},
        load_weights=False,
    )


def test_model_input_spec__uses_channels_and_image_size() -> None:
    spec = _model().model_input_spec
    assert list(spec.input_specs) == ["images"]
    assert spec.input_specs["images"].shape == (1, 256, 320)
    assert spec.input_specs["images"].dtype == torch.float32
    assert spec.input_specs["images"].is_batched is True


def test_onnx_export_metadata() -> None:
    metadata = _model().onnx_export_metadata()
    assert metadata["lightly_train_version"] == __version__
    assert metadata["license_info"] == LICENSE_INFO
    assert json.loads(metadata["image_normalize"]) == {
        "mean": [0.0],
        "std": [1.0],
    }
    assert json.loads(metadata["classes"]) == {"0": "car", "1": "person"}
    assert metadata["model_name"] == "dinov3/vitt16-notpretrained-ltdetr"
    assert "image_size" not in metadata
