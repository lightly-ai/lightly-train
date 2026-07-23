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
import torch
from lightning_utilities.core.imports import RequirementCache
from pytest_mock import MockerFixture
from torch import nn

from lightly_train._commands.train_task_helpers import get_steps
from lightly_train._task_models.ltdetr_instance_segmentation.task_model import (
    LTDETRInstanceSegmentation,
)
from lightly_train._task_models.ltdetr_instance_segmentation.train_model import (
    LTDETRInstanceSegmentationLargeTrainArgs,
    LTDETRInstanceSegmentationTrain,
    LTDETRInstanceSegmentationTrainArgs,
)

from ...helpers import assert_onnx_outputs_close


def _is_module_frozen(m: nn.Module) -> bool:
    return all(not param.requires_grad for param in m.parameters())


@pytest.fixture()
def model() -> LTDETRInstanceSegmentation:
    return LTDETRInstanceSegmentation(
        model_name="ltdetrv2-seg-s",
        classes={0: "background", 1: "car"},
        image_size=(256, 256),
        patch_size=16,
        image_normalize={"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        backbone_freeze=False,
        backbone_weights=None,
        backbone_args=None,
        load_weights=False,
    )


def test_task_model_init_args_roundtrip_preserves_patch_size() -> None:
    model = LTDETRInstanceSegmentation(
        model_name="ltdetrv2-seg-s",
        classes={0: "background", 1: "car"},
        image_size=(64, 64),
        patch_size=16,
        image_normalize=None,
        backbone_freeze=False,
        backbone_weights=None,
        backbone_args=None,
        load_weights=False,
    )

    assert model.init_args["patch_size"] == 16

    roundtrip_model = LTDETRInstanceSegmentation(
        **model.init_args,
        load_weights=False,
    )
    assert roundtrip_model.init_args == model.init_args
    assert roundtrip_model.init_args["patch_size"] == 16


def test_load_train_state_dict__uses_ema_weights(
    model: LTDETRInstanceSegmentation,
) -> None:
    state_dict = {
        f"ema_model.model.{name}": param.detach().clone()
        for name, param in model.state_dict().items()
    }

    incompatible = model.load_train_state_dict(state_dict)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_load_train_state_dict__falls_back_to_model_weights(
    model: LTDETRInstanceSegmentation,
) -> None:
    state_dict = {
        f"model.{name}": param.detach().clone()
        for name, param in model.state_dict().items()
    }

    incompatible = model.load_train_state_dict(state_dict)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_predict_batch__composes_stages_in_order(
    model: LTDETRInstanceSegmentation, mocker: MockerFixture
) -> None:
    # Spy (not mock) on every stage so the real backbone/encoder/decoder and
    # postprocessor run end-to-end while we assert the wiring between stages.
    preprocess_image_spy = mocker.spy(model, "preprocess_image")
    preprocess_batch_spy = mocker.spy(model, "preprocess_batch")
    forward_backend_spy = mocker.spy(model, "forward_backend")
    postprocess_spy = mocker.spy(model, "postprocess")

    images = [torch.rand(3, 24, 32), torch.rand(3, 40, 24)]
    result = model.predict_batch(images=images, threshold=0.7)

    # Each input image goes through preprocess_image once.
    assert preprocess_image_spy.call_count == 2

    # The stacked batch is preprocessed in a single call with shape (B, C, H, W).
    assert preprocess_batch_spy.call_count == 1
    (batch_in,) = preprocess_batch_spy.call_args.args
    assert batch_in.shape == (2, 3, 256, 256)

    # forward_backend receives the output of preprocess_batch.
    assert forward_backend_spy.call_count == 1
    (forward_in,) = forward_backend_spy.call_args.args
    assert forward_in is preprocess_batch_spy.spy_return

    # postprocess receives forward_backend's output and per-image metadata.
    assert postprocess_spy.call_count == 1
    raw_in, metadata = postprocess_spy.call_args.args
    assert raw_in is forward_backend_spy.spy_return
    assert len(metadata) == 2
    assert postprocess_spy.call_args.kwargs == {"threshold": 0.7}

    # predict_batch returns whatever postprocess produced: one dict per image
    # with the instance-segmentation prediction keys, masks at original size.
    assert result is postprocess_spy.spy_return
    assert len(result) == 2
    for prediction, image in zip(result, images):
        assert set(prediction) == {"labels", "bboxes", "masks", "scores"}
        assert prediction["masks"].dtype == torch.bool
        assert prediction["masks"].shape[-2:] == image.shape[-2:]


def test_predict__single_image_returns_prediction_dict(
    model: LTDETRInstanceSegmentation,
) -> None:
    # ``predict`` runs the real forward path for a single image and returns one
    # prediction dict (not a list), with masks resized back to the original
    # image size.
    image = torch.rand(3, 40, 24)
    prediction = model.predict(image, threshold=0.7)

    assert isinstance(prediction, dict)
    assert set(prediction) == {"labels", "bboxes", "masks", "scores"}
    assert prediction["masks"].dtype == torch.bool
    assert prediction["masks"].shape[-2:] == image.shape[-2:]
    # All predicted labels must be valid class IDs from ``classes``.
    assert set(prediction["labels"].tolist()).issubset(set(model.classes))


# ---------------------------------------------------------------------------
# EdgeCrafter (ECViT) backbone tests
# ---------------------------------------------------------------------------
#
# The ECViT backbones are exposed under the ``edgecrafter/`` package prefix
# (e.g. ``edgecrafter/ecvitt-ltdetr-seg``) and are dispatched inside the EdgeCrafter
# LTDETR task model. These tests verify the wiring without depending on the
# pretrained weight download.


ECVIT_LTDETR_SEG_MODEL_NAMES = [
    "edgecrafter/ecvitt-ltdetr-seg",
    "edgecrafter/ecvittplus-ltdetr-seg",
    "edgecrafter/ecvits-ltdetr-seg",
    "edgecrafter/ecvitsplus-ltdetr-seg",
]


@pytest.mark.parametrize("model_name", ECVIT_LTDETR_SEG_MODEL_NAMES)
def test_is_supported_model__ecvit(model_name: str) -> None:
    assert LTDETRInstanceSegmentation.is_supported_model(model_name) is True


@pytest.mark.parametrize("model_name", ECVIT_LTDETR_SEG_MODEL_NAMES)
def test_parse_model_name__ecvit(model_name: str) -> None:
    parsed = LTDETRInstanceSegmentation.parse_model_name(model_name)
    assert parsed["package_name"] == "edgecrafter"
    assert parsed["model_name"] == model_name
    assert parsed["backbone_name"] in {
        "ecvitt",
        "ecvittplus",
        "ecvits",
        "ecvitsplus",
    }


@pytest.mark.parametrize("model_name", ECVIT_LTDETR_SEG_MODEL_NAMES)
def test_task_model__ecvit_rejects_non_16_patch_size(model_name: str) -> None:
    # The ECViT-NN ConvPyramidPatchEmbed only supports patch_size=16. The task
    # model must reject any other value with a ValueError naming the constraint;
    # otherwise ``config.resolve_auto`` would build strides at the wrong scale
    # and anchors would be misaligned.
    with pytest.raises(ValueError, match=r"patch_size=16"):
        LTDETRInstanceSegmentation(
            model_name=model_name,
            classes={0: "background", 1: "car"},
            image_size=(256, 256),
            patch_size=32,
            load_weights=False,
        )


# ---------------------------------------------------------------------------
# Short LT-DETRv2 alias tests
# ---------------------------------------------------------------------------
#
# ``ltdetrv2-{s,m,l,x}`` is a public alias that resolves to the canonical
# EdgeCrafter (ECViT) LT-DETR object-detection model name. These tests verify
# that the alias is accepted by ``is_supported_model`` and resolves to the
# correct canonical name in ``parse_model_name``.

LTDETR_V2_SEG_ALIAS_MODEL_NAMES = [
    "ltdetrv2-seg-s",
    "ltdetrv2-seg-m",
    "ltdetrv2-seg-l",
    "ltdetrv2-seg-x",
]

LTDETR_V2_SEG_ALIAS_TO_CANONICAL: dict[str, str] = {
    "ltdetrv2-seg-s": "edgecrafter/ecvitt-ltdetr-seg",
    "ltdetrv2-seg-m": "edgecrafter/ecvittplus-ltdetr-seg",
    "ltdetrv2-seg-l": "edgecrafter/ecvits-ltdetr-seg",
    "ltdetrv2-seg-x": "edgecrafter/ecvitsplus-ltdetr-seg",
}

LTDETR_V2_SEG_DEFAULT_STEPS = {
    "edgecrafter/ecvitt-ltdetr-seg": 273_504,
    "edgecrafter/ecvittplus-ltdetr-seg": 273_504,
    "edgecrafter/ecvits-ltdetr-seg": 184_800,
    "edgecrafter/ecvitsplus-ltdetr-seg": 184_800,
}


@pytest.mark.parametrize("model_name", LTDETR_V2_SEG_ALIAS_MODEL_NAMES)
def test_is_supported_model__ltdetrv2_alias(model_name: str) -> None:
    assert LTDETRInstanceSegmentation.is_supported_model(model_name) is True


@pytest.mark.parametrize(
    ("alias", "canonical"),
    list(LTDETR_V2_SEG_ALIAS_TO_CANONICAL.items()),
)
def test_parse_model_name__ltdetrv2_alias(alias: str, canonical: str) -> None:
    parsed = LTDETRInstanceSegmentation.parse_model_name(alias)
    assert parsed["package_name"] == "edgecrafter"
    assert parsed["model_name"] == canonical
    expected_backbone_name = canonical
    if expected_backbone_name.startswith("edgecrafter/"):
        expected_backbone_name = expected_backbone_name[len("edgecrafter/") :]
    if expected_backbone_name.endswith("-ltdetr-seg"):
        expected_backbone_name = expected_backbone_name[: -len("-ltdetr-seg")]
    assert parsed["backbone_name"] == expected_backbone_name


def test_list_model_names__includes_ltdetrv2_aliases() -> None:
    names = LTDETRInstanceSegmentation.list_model_names()
    for alias in LTDETR_V2_SEG_ALIAS_MODEL_NAMES:
        assert alias in names, f"Expected alias {alias!r} in list_model_names()"


@pytest.mark.parametrize(
    "model_name",
    [*ECVIT_LTDETR_SEG_MODEL_NAMES, *LTDETR_V2_SEG_ALIAS_MODEL_NAMES],
)
def test_train_args_cls__uses_model_default_schedule(model_name: str) -> None:
    train_args_cls = LTDETRInstanceSegmentationTrain.get_train_model_args_cls(
        model_name=model_name
    )
    canonical_model_name = LTDETR_V2_SEG_ALIAS_TO_CANONICAL.get(model_name, model_name)
    expected_steps = LTDETR_V2_SEG_DEFAULT_STEPS[canonical_model_name]
    expected_cls = (
        LTDETRInstanceSegmentationLargeTrainArgs
        if expected_steps == 184_800
        else LTDETRInstanceSegmentationTrainArgs
    )

    assert train_args_cls is expected_cls
    assert train_args_cls.default_steps == expected_steps


def test_train_args_cls__explicit_steps_override_model_default() -> None:
    train_args_cls = LTDETRInstanceSegmentationTrain.get_train_model_args_cls(
        model_name="ltdetrv2-seg-l"
    )

    assert (
        get_steps(steps="auto", default_steps=train_args_cls.default_steps) == 184_800
    )
    assert get_steps(steps=12_345, default_steps=train_args_cls.default_steps) == 12_345


def test_freeze_backbone_on_init() -> None:
    model = LTDETRInstanceSegmentation(
        model_name="ltdetrv2-seg-s",
        classes={0: "background", 1: "car"},
        image_size=(64, 64),
        patch_size=16,
        image_normalize=None,
        backbone_freeze=True,
        backbone_weights=None,
        backbone_args=None,
        load_weights=False,
    )

    assert _is_module_frozen(model.backbone)
    assert not model.backbone.training


def _run_seg_onnx(path: Path, images: object, orig_target_size: object) -> list:  # type: ignore[type-arg]
    """Run the exported segmentation graph in ONNX Runtime on CPU."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs: list = session.run(  # type: ignore[type-arg]
        None, {"images": images, "orig_target_size": orig_target_size}
    )
    return outputs


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__dynamic_batch_size(
    model: LTDETRInstanceSegmentation, tmp_path: Path
) -> None:
    import numpy as np
    import onnx

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, simplify=False, verify=True)

    onnx_model = onnx.load(out)
    input_batch_dim = onnx_model.graph.input[0].type.tensor_type.shape.dim[0]
    assert input_batch_dim.dim_param == "N"

    # Use a batch size (3) different from the one used during tracing (2).
    inputs = np.random.randn(3, 3, 256, 256).astype(np.float32)
    orig_target_size = np.array([[256, 256]] * 3, dtype=np.int64)

    onnx_outputs = _run_seg_onnx(out, inputs, orig_target_size)

    with torch.no_grad():
        torch_outputs = model(
            torch.from_numpy(inputs), torch.from_numpy(orig_target_size)
        )

    assert_onnx_outputs_close(onnx_outputs, torch_outputs)


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__static_batch_size(
    model: LTDETRInstanceSegmentation, tmp_path: Path
) -> None:
    import onnx

    out = tmp_path / "model.onnx"
    model.export_onnx(
        out=out, batch_size=3, dynamic_batch_size=False, simplify=False, verify=True
    )

    onnx_model = onnx.load(out)
    input_batch_dim = onnx_model.graph.input[0].type.tensor_type.shape.dim[0]
    assert input_batch_dim.dim_value == 3


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_export_onnx__fp16(model: LTDETRInstanceSegmentation, tmp_path: Path) -> None:
    import onnx

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, precision="fp16", simplify=True, verify=True)

    model_onnx = onnx.load(str(out))
    # Verify the model has fp16 tensors.
    has_fp16 = any(
        init.data_type == onnx.TensorProto.FLOAT16
        for init in model_onnx.graph.initializer
    )
    assert has_fp16


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__simplify_matches_unsimplified(
    model: LTDETRInstanceSegmentation, tmp_path: Path
) -> None:
    # onnxslim rewrites the graph in-place; the simplified model must produce
    # the same outputs as the unsimplified one for the same input.
    import numpy as np

    simplified = tmp_path / "simplified.onnx"
    unsimplified = tmp_path / "unsimplified.onnx"
    model.export_onnx(out=simplified, simplify=True, verify=True)
    model.export_onnx(out=unsimplified, simplify=False, verify=True)

    inputs = np.random.randn(2, 3, 256, 256).astype(np.float32)
    orig_target_size = np.array([[256, 256]] * 2, dtype=np.int64)

    simplified_outputs = _run_seg_onnx(simplified, inputs, orig_target_size)
    unsimplified_outputs = _run_seg_onnx(unsimplified, inputs, orig_target_size)

    assert_onnx_outputs_close(
        simplified_outputs,
        tuple(torch.from_numpy(o) for o in unsimplified_outputs),
    )


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
@pytest.mark.parametrize("model_name", LTDETR_V2_SEG_ALIAS_MODEL_NAMES)
def test_export_onnx__aliases_and_non_square(model_name: str, tmp_path: Path) -> None:
    # Every registered alias (s/m/l/x uses a different encoder/transformer
    # config) must export at a non-square, non-default image size.
    import numpy as np

    image_size = (192, 256)
    model = LTDETRInstanceSegmentation(
        model_name=model_name,
        classes={0: "background", 1: "car"},
        image_size=image_size,
        patch_size=16,
        image_normalize={"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
        load_weights=False,
    )

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, simplify=False, verify=True)

    height, width = image_size
    inputs = np.random.randn(1, 3, height, width).astype(np.float32)
    orig_target_size = np.array([[height, width]], dtype=np.int64)

    onnx_outputs = _run_seg_onnx(out, inputs, orig_target_size)
    with torch.no_grad():
        torch_outputs = model(
            torch.from_numpy(inputs), torch.from_numpy(orig_target_size)
        )
    assert_onnx_outputs_close(onnx_outputs, torch_outputs)


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
@pytest.mark.parametrize("opset_version", [16, 17, 18, 19, 20])
def test_export_onnx__opset_version(
    model: LTDETRInstanceSegmentation, tmp_path: Path, opset_version: int
) -> None:
    # The graph uses scaled_dot_product_attention (needs opset >= 14) and
    # grid_sampler (needs opset >= 16), so opset 16 is the minimum supported.
    import numpy as np

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, opset_version=opset_version, simplify=False, verify=True)
    inputs = np.random.randn(1, 3, 256, 256).astype(np.float32)
    orig_target_size = np.array([[256, 256]], dtype=np.int64)
    onnx_outputs = _run_seg_onnx(out, inputs, orig_target_size)
    with torch.no_grad():
        torch_outputs = model(
            torch.from_numpy(inputs), torch.from_numpy(orig_target_size)
        )
    assert_onnx_outputs_close(onnx_outputs, torch_outputs)


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__matches_predict(
    model: LTDETRInstanceSegmentation, tmp_path: Path
) -> None:
    # The exported graph omits the normalization done in ``preprocess_batch`` and
    # the mask upsampling / binarization / thresholding done in ``postprocess``.
    # This test verifies that feeding a normalized input to the ONNX graph and
    # replaying that documented post-processing reproduces ``predict``.
    #
    # Top-k ordering is not stable across backends for near-tied random scores,
    # so the two prediction sets are compared order-invariantly.
    import numpy as np
    import torch.nn.functional as F
    from torchvision.transforms.v2 import functional as transforms_functional

    assert model.image_normalize is not None
    threshold = 0.0  # Keep all queries so the comparison is deterministic.

    # Reference prediction. The image already matches ``image_size`` so
    # ``predict`` performs no resize and ``orig`` equals ``image_size``.
    image = torch.rand(3, 256, 256)
    prediction = model.predict(image, threshold=threshold)

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, simplify=False, verify=False)

    # Replicate ``preprocess_batch``: the graph expects an already-normalized
    # image (``preprocess_image`` already scaled it to ``[0, 1]``).
    normalized = transforms_functional.normalize(
        image,
        mean=list(model.image_normalize["mean"]),
        std=list(model.image_normalize["std"]),
    )
    inputs = normalized.unsqueeze(0).numpy().astype(np.float32)
    orig_target_size = np.array([[256, 256]], dtype=np.int64)

    labels, boxes, masks, scores = (
        torch.from_numpy(o) for o in _run_seg_onnx(out, inputs, orig_target_size)
    )

    # Replicate ``postprocess``: threshold, upsample masks to the original size,
    # and binarize.
    keep = scores[0] > threshold
    onnx_masks = (
        F.interpolate(
            masks[0][keep].unsqueeze(1),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        > 0.0
    )

    # Same number of predictions and matching (order-invariant) score/label sets.
    assert int(keep.sum()) == prediction["scores"].numel()
    torch.testing.assert_close(
        torch.sort(scores[0][keep]).values,
        torch.sort(prediction["scores"]).values,
        atol=2e-2,
        rtol=1e-1,
    )
    assert torch.equal(
        torch.sort(labels[0][keep]).values,
        torch.sort(prediction["labels"]).values,
    )
    # Masks have the original resolution, are boolean, and cover the same area
    # (order-invariant proxy for per-instance agreement).
    assert onnx_masks.shape == prediction["masks"].shape
    assert onnx_masks.dtype == torch.bool == prediction["masks"].dtype
    onnx_area = int(onnx_masks.sum())
    predict_area = int(prediction["masks"].sum())
    assert abs(onnx_area - predict_area) <= max(1, round(0.01 * predict_area))
