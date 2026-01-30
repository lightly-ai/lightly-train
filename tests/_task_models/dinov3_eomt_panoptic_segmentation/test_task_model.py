#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path

import pytest
from lightning_utilities.core.imports import RequirementCache

from lightly_train._task_models.dinov3_eomt_panoptic_segmentation.task_model import (
    DINOv3EoMTPanopticSegmentation,
)


@pytest.fixture()
def model() -> DINOv3EoMTPanopticSegmentation:
    return DINOv3EoMTPanopticSegmentation(
        model_name="dinov3/_vittest16-eomt",
        thing_classes={0: "car", 1: "person"},
        stuff_classes={2: "sky", 3: "road"},
        image_size=(16, 16),
        image_normalize={"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        num_queries=2,
        num_joint_blocks=1,
        load_weights=False,
    )


@pytest.mark.xfail(strict=True, reason="dinov3 ONNX export shape mismatch")
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx(model: DINOv3EoMTPanopticSegmentation, tmp_path: Path) -> None:
    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, simplify=False, verify=True)


@pytest.mark.xfail(strict=True, reason="dinov3 ONNX export shape mismatch")
@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__custom_height_width(
    model: DINOv3EoMTPanopticSegmentation, tmp_path: Path
) -> None:
    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, height=32, width=48, simplify=False, verify=True)
