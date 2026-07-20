#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from lightning_utilities.core.imports import RequirementCache
from PIL import Image

from lightly_train._task_models.depth_estimation import task_model
from lightly_train._task_models.depth_estimation.task_model import (
    DepthAnythingDepthEstimation,
)


# Tiny per-variant model args so inference runs at a small resolution and tests stay
# fast. DAv3 variants carry `use_sky_head` inside `model_args` exactly as the exported
# checkpoints do; DAv2 metric variants carry `max_depth`.
def _tiny_model_args(model_name: str) -> dict[str, Any]:
    patch_size = 16 if model_name.startswith("dinov3/") else 14
    args: dict[str, Any] = {
        "out_layers": (0, 1, 2, 3),
        "image_size": patch_size * 4,
        "patch_size": patch_size,
        "features": 16,
        "out_channels": (8, 16, 32, 32),
    }
    if "metric" in model_name and "dav2" in model_name:
        args["max_depth"] = 20.0
    if "dav3" in model_name:
        args["use_sky_head"] = True
    return args


def _build(model_name: str, **overrides: Any) -> DepthAnythingDepthEstimation:
    model_args = {**_tiny_model_args(model_name), **overrides}
    model = DepthAnythingDepthEstimation(
        model_name=model_name,
        model_args=model_args,
        load_weights=False,
    )
    # Production fixes the image size per model; override it here so inference runs
    # at a tiny resolution and the tests stay fast.
    model.image_size = int(model_args["image_size"])
    return model


# ONNX export runs the backbone several times (export trace, onnxslim and the
# fp32-reference verify pass), so the export tests use the small (vits14, ~22M) backbone
# instead of the large (~303M) one to stay fast. The sky head is a decoder feature
# toggled via `model_args`, so a single small backbone covers both export output sets.
def _build_for_export(*, use_sky_head: bool) -> DepthAnythingDepthEstimation:
    model_args = _tiny_model_args("dinov2/dav2-relative-small")
    model_args["use_sky_head"] = use_sky_head
    model = DepthAnythingDepthEstimation(
        model_name="dinov2/dav2-relative-small",
        model_args=model_args,
        load_weights=False,
    )
    model.image_size = 56
    return model


# Model names that do not take intrinsics (relative + DAv2 metric).
_NON_FOCAL_NAMES = [
    "dinov2/dav2-relative-large",
    "dinov2/dav2-metric-large-hypersim",
    "dinov2/dav3-relative-large",
    "dinov3/dav3-relative-tiny",
    "dinov3/dav3-relative-tiny-plus",
]
# Model names with a sky head (DAv3).
_SKY_NAMES = [
    "dinov2/dav3-relative-large",
    "dinov2/dav3-metric-large",
    "dinov3/dav3-relative-tiny",
    "dinov3/dav3-relative-tiny-plus",
]
_NO_SKY_NAMES = ["dinov2/dav2-relative-large", "dinov2/dav2-metric-large-hypersim"]


class TestDepthAnythingDepthEstimation:
    @pytest.mark.parametrize("model_name", _NON_FOCAL_NAMES)
    def test_predict__returns_original_resolution(self, model_name: str) -> None:
        model = _build(model_name)
        image = Image.new("RGB", (80, 64), color=(32, 64, 128))

        depth = model.predict(image)

        # Inference runs at the processing resolution and the depth map is resized back
        # to the original input resolution.
        assert depth.shape == (64, 80)
        assert depth.dtype == next(model.parameters()).dtype
        assert torch.isfinite(depth).all()

    def test_predict__returns_original_resolution_focal(self) -> None:
        model = _build("dinov2/dav3-metric-large")
        image = Image.new("RGB", (80, 64), color=(32, 64, 128))
        intrinsics = torch.tensor(
            [[600.0, 0.0, 40.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]]
        )

        depth = model.predict(image, intrinsics=intrinsics)

        assert depth.shape == (64, 80)
        assert depth.dtype == next(model.parameters()).dtype
        assert torch.isfinite(depth).all()

    @pytest.mark.parametrize("model_name", _NON_FOCAL_NAMES)
    def test_predict_batch__mixed_sizes(self, model_name: str) -> None:
        model = _build(model_name)
        images = [
            Image.new("RGB", (80, 64), color=(32, 64, 128)),
            Image.new("RGB", (56, 56), color=(128, 64, 32)),
        ]

        depths = model.predict_batch(images)

        # The processed sizes differ, so the batch is center-cropped to the smallest
        # processed size before inference; each depth map is still resized back to its
        # image's original resolution.
        assert len(depths) == 2
        assert depths[0].shape == (64, 80)
        assert depths[1].shape == (56, 56)
        for depth in depths:
            assert torch.isfinite(depth).all()

    def test_predict__max_depth_scales_output(self) -> None:
        # The metric depth scales linearly with max_depth: with identical weights,
        # doubling max_depth must exactly double the predicted depth.
        model_20 = _build("dinov2/dav2-metric-large-hypersim", max_depth=20.0)
        model_40 = _build("dinov2/dav2-metric-large-hypersim", max_depth=40.0)
        model_40.load_state_dict(model_20.state_dict())
        image = Image.new("RGB", (56, 56), color=(32, 64, 128))

        depth_20 = model_20.predict(image)
        depth_40 = model_40.predict(image)

        assert torch.equal(depth_40, depth_20 * 2.0)

    def test_predict__intrinsics_scale_depth_by_focal(self) -> None:
        model = _build("dinov2/dav3-metric-large")
        # A 56x56 image is processed at its own size, so focal lengths are not rescaled.
        # fx = fy = 600 → scale 600/300 = 2; fx = fy = 1200 → scale 1200/300 = 4.
        # Doubling the focal should exactly double the metric depth.
        image = Image.new("RGB", (56, 56), color=(32, 64, 128))
        intrinsics_600 = torch.tensor(
            [[600.0, 0.0, 28.0], [0.0, 600.0, 28.0], [0.0, 0.0, 1.0]]
        )
        intrinsics_1200 = torch.tensor(
            [[1200.0, 0.0, 28.0], [0.0, 1200.0, 28.0], [0.0, 0.0, 1.0]]
        )

        depth_600 = model.predict(image, intrinsics=intrinsics_600)
        depth_1200 = model.predict(image, intrinsics=intrinsics_1200)

        assert torch.equal(depth_1200, depth_600 * 2.0)

    def test_predict_batch__intrinsics_per_image(self) -> None:
        model = _build("dinov2/dav3-metric-large")
        images = [
            Image.new("RGB", (56, 56), color=(32, 64, 128)),
            Image.new("RGB", (56, 56), color=(128, 64, 32)),
        ]
        intrinsics = [
            torch.tensor([[600.0, 0.0, 28.0], [0.0, 600.0, 28.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[1200.0, 0.0, 28.0], [0.0, 1200.0, 28.0], [0.0, 0.0, 1.0]]),
        ]
        # Scale each image's focal by 2 to verify per-image scaling.
        intrinsics_doubled = [
            torch.tensor([[1200.0, 0.0, 28.0], [0.0, 1200.0, 28.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[2400.0, 0.0, 28.0], [0.0, 2400.0, 28.0], [0.0, 0.0, 1.0]]),
        ]

        metric = model.predict_batch(images, intrinsics=intrinsics)
        metric_doubled = model.predict_batch(images, intrinsics=intrinsics_doubled)

        assert torch.equal(metric_doubled[0], metric[0] * 2.0)
        assert torch.equal(metric_doubled[1], metric[1] * 2.0)

    def test_list_model_names__includes_metric_small(self) -> None:
        # The trainable metric student must be registered so it routes to the depth
        # train model and is a valid `model` for `train_depth_estimation`.
        assert (
            "dinov2/dav3-metric-small"
            in DepthAnythingDepthEstimation.list_model_names()
        )

    def test_predict__intrinsics_required_for_metric_small(self) -> None:
        # The metric-small student is `scale_mode="focal"`, so it requires intrinsics
        # exactly like the metric-large model.
        model = _build("dinov2/dav3-metric-small")
        image = Image.new("RGB", (56, 56), color=(32, 64, 128))

        with pytest.raises(ValueError, match="requires per-image camera intrinsics"):
            model.predict(image)

    def test_predict__intrinsics_required_for_focal_model(self) -> None:
        model = _build("dinov2/dav3-metric-large")
        image = Image.new("RGB", (56, 56), color=(32, 64, 128))

        with pytest.raises(ValueError, match="requires per-image camera intrinsics"):
            model.predict(image)

    @pytest.mark.parametrize("model_name", _NON_FOCAL_NAMES)
    def test_predict__intrinsics_rejected_for_nonfocal_model(
        self, model_name: str
    ) -> None:
        model = _build(model_name)
        image = Image.new("RGB", (56, 56), color=(32, 64, 128))
        intrinsics = torch.tensor(
            [[600.0, 0.0, 28.0], [0.0, 600.0, 28.0], [0.0, 0.0, 1.0]]
        )

        with pytest.raises(ValueError, match="does not accept intrinsics"):
            model.predict(image, intrinsics=intrinsics)

    @pytest.mark.parametrize("model_name", _NO_SKY_NAMES)
    def test_forward__returns_depth(self, model_name: str) -> None:
        model = _build(model_name)
        patch_size = model.patch_size
        x = torch.randn(2, 3, patch_size * 4, patch_size * 5)

        out = model(x)

        # Depth Anything V2 has no sky head, so the forward output is (depth,) only.
        assert len(out) == 1
        assert out[0].shape == (2, 1, patch_size * 4, patch_size * 5)

    @pytest.mark.parametrize("model_name", _SKY_NAMES)
    def test_forward__returns_depth_and_sky(self, model_name: str) -> None:
        model = _build(model_name)
        patch_size = model.patch_size
        x = torch.randn(2, 3, patch_size * 4, patch_size * 5)

        out = model(x)

        # A model with a sky head (DAv3) returns (depth, sky) in that order.
        depth, sky = out
        assert depth.shape == (2, 1, patch_size * 4, patch_size * 5)
        assert sky.shape == (2, 1, patch_size * 4, patch_size * 5)

    @pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
    @pytest.mark.skipif(
        not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
    )
    @pytest.mark.parametrize(
        "use_sky_head, expected_output_names",
        [
            (True, ["depth", "sky"]),
            (False, ["depth"]),
        ],
    )
    def test_export_onnx__output_names_match_sky_head(
        self, use_sky_head: bool, expected_output_names: list[str], tmp_path: Path
    ) -> None:
        import onnx

        model = _build_for_export(use_sky_head=use_sky_head)
        out = tmp_path / "model.onnx"

        model.export_onnx(out=out, simplify=False, verify=True)

        # A model with a sky head (DAv3) exports both depth and sky; one without (DAv2)
        # exports depth only.
        onnx_model = onnx.load(out)
        assert [o.name for o in onnx_model.graph.output] == expected_output_names

    @pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
    @pytest.mark.skipif(
        not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
    )
    def test_export_onnx__dynamic_batch_size(self, tmp_path: Path) -> None:
        import numpy as np
        import onnx
        import onnxruntime as ort

        # A sky head exercises both ONNX outputs against PyTorch.
        model = _build_for_export(use_sky_head=True)
        out = tmp_path / "model.onnx"

        model.export_onnx(out=out, dynamic_batch_size=True, simplify=False, verify=True)

        onnx_model = onnx.load(out)
        input_batch_dim = onnx_model.graph.input[0].type.tensor_type.shape.dim[0]
        assert input_batch_dim.dim_param == "N"

        inputs = np.random.randn(3, 3, 56, 56).astype(np.float32)
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        onnx_outputs = session.run(None, {"images": inputs})

        # `forward` returns (depth, sky) in the same order as the ONNX outputs.
        with torch.no_grad():
            torch_outputs = model(torch.from_numpy(inputs))

        assert len(onnx_outputs) == len(torch_outputs)
        for onnx_out, torch_out in zip(onnx_outputs, torch_outputs):
            close = torch.isclose(
                torch.from_numpy(onnx_out), torch_out, atol=2e-2, rtol=1e-1
            )
            assert close.float().mean() > 0.95

    @pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
    @pytest.mark.skipif(
        not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
    )
    def test_export_onnx__static_batch_size(self, tmp_path: Path) -> None:
        import onnx

        model = _build_for_export(use_sky_head=False)
        out = tmp_path / "model.onnx"

        model.export_onnx(
            out=out, batch_size=2, dynamic_batch_size=False, simplify=False, verify=True
        )

        # A static batch size fixes the batch dimension to the integer instead of "N".
        onnx_model = onnx.load(out)
        input_batch_dim = onnx_model.graph.input[0].type.tensor_type.shape.dim[0]
        assert input_batch_dim.dim_value == 2

    @pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
    @pytest.mark.skipif(
        not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
    )
    def test_export_onnx__custom_height_width(self, tmp_path: Path) -> None:
        import onnx

        model = _build_for_export(use_sky_head=False)
        out = tmp_path / "model.onnx"

        # Height and width must be multiples of the patch size (14).
        model.export_onnx(out=out, height=70, width=42, simplify=False, verify=True)

        # The height/width override flows into the ONNX input's spatial dimensions.
        onnx_model = onnx.load(out)
        input_dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
        assert input_dims[2].dim_value == 70
        assert input_dims[3].dim_value == 42


def test__processed_focal_length() -> None:
    intrinsics = torch.tensor(
        [[400.0, 0.0, 100.0], [0.0, 300.0, 50.0], [0.0, 0.0, 1.0]]
    )

    focal = task_model._processed_focal_length(
        intrinsics=intrinsics,
        orig_h=100,
        orig_w=200,
        proc_h=50,
        proc_w=100,
    )

    # fx scales by the width ratio (400 * 0.5 = 200), fy by the height ratio
    # (300 * 0.5 = 150), and the focal is their average.
    assert focal == 175.0
