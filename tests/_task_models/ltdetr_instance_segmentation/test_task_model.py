#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
import torch
from pytest_mock import MockerFixture
from torch import nn

from lightly_train._task_models.ltdetr_instance_segmentation.task_model import (
    LTDETRInstanceSegmentation,
)


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
