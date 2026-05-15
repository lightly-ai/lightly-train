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

from lightly_train._task_models.dinov2_eomt_panoptic_segmentation.task_model import (
    DINOv2EoMTPanopticSegmentation,
)


@pytest.fixture()
def model() -> DINOv2EoMTPanopticSegmentation:
    return DINOv2EoMTPanopticSegmentation(
        model_name="dinov2/_vittest14-eomt",
        thing_classes={0: "car", 1: "person"},
        stuff_classes={2: "sky", 3: "road"},
        image_size=(14, 14),
        image_normalize={"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        num_queries=2,
        num_joint_blocks=1,
        load_weights=False,
    )


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx(model: DINOv2EoMTPanopticSegmentation, tmp_path: Path) -> None:
    import onnx

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, simplify=False, verify=True)

    onnx_model = onnx.load(out)
    output_names = [o.name for o in onnx_model.graph.output]
    assert "masks" in output_names
    assert "segment_ids" in output_names
    assert "scores" in output_names
