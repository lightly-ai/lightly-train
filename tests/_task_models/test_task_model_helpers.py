#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from pytest import MonkeyPatch
from torch.hub import download_url_to_file
from torch.nn import Conv2d, Embedding, Linear, Module

from lightly_train import load_model
from lightly_train._task_models import task_model_helpers
from lightly_train._task_models.dinov3_eomt_instance_segmentation.config import (
    DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY,
)
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.config import (
    DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY,
)
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.task_model import (
    DINOv3EoMTSemanticSegmentation,
)
from lightly_train._task_models.ltdetr_instance_segmentation.config import (
    LTDETR_SEG_MODEL_REGISTRY,
)
from lightly_train._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_decoder import (
    RTDETRTransformerv2,
)
from lightly_train._task_models.picodet_object_detection.config import (
    PICODET_OBJECT_DETECTION_MODEL_REGISTRY,
)


def test_load_model__download(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTLY_TRAIN_MODEL_CACHE_DIR", str(tmp_path))
    model_name = "dinov3/vits16-eomt-coco"
    model_file_name = r"dinov3_vits16_eomt_coco_??????_????????.pt"
    expected_model_type = DINOv3EoMTSemanticSegmentation
    expected_model_name = "dinov3/vits16-eomt"

    with patch(
        "torch.hub.download_url_to_file", wraps=download_url_to_file
    ) as spy_download_url_to_file:
        model = load_model(model_name)

        files = list(tmp_path.glob(model_file_name))
        assert len(files) == 1
        assert files[0].is_file()
        assert isinstance(model, expected_model_type)
        assert model.model_name == expected_model_name
        assert spy_download_url_to_file.call_count == 1

        # Ensure that the model is cached and not downloaded a second time
        model2 = load_model(model_name)
        assert isinstance(model2, expected_model_type)
        assert spy_download_url_to_file.call_count == 1


def test_load_model__download_invalid_model__fails() -> None:
    invalid_model_name = "definitely-not-a-valid-model-name"
    expected_error_message = (
        f"Unknown model name or checkpoint path: '{invalid_model_name}'"
    )

    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        load_model(invalid_model_name)


def test_downloadable_model__ltdetrv2_s_coco_alias() -> None:
    # ``ltdetrv2-s-coco`` is the public name for the hosted ECViT-T LTDETR COCO
    # checkpoint. It must resolve to the same (file, hash) as the canonical
    # ``edgecrafter/ecvitt-ltdetr-coco`` entry so load_model/train downloads the
    # identical file regardless of which name the user passes.
    d = task_model_helpers.DOWNLOADABLE_MODEL_URL_AND_HASH
    assert d["ltdetrv2-s-coco"] == d["edgecrafter/ecvitt-ltdetr-coco"]


def test_downloadable_model__dinov3_eomt_semantic_aliases_from_registry() -> None:
    aliases = [
        "dinov3/vits16-eomt-coco",
        "dinov3/vits32-eomt-coco",
        "dinov3/vitl16-eomt-ade20k",
    ]

    for alias in aliases:
        checkpoint = (
            DINOV3_EOMT_SEMANTIC_SEGMENTATION_MODEL_REGISTRY.get_alias_metadata(
                alias
            ).downloadable_checkpoint
        )
        assert task_model_helpers.DOWNLOADABLE_MODEL_URL_AND_HASH[alias] == (
            checkpoint.url,
            checkpoint.sha256,
        )


def test_downloadable_model__dinov3_eomt_instance_aliases_from_registry() -> None:
    aliases = [
        "dinov3/vitt16-eomt-inst-coco",
        "dinov3/vitt16plus-eomt-inst-coco",
        "dinov3/vits16-eomt-inst-coco",
        "dinov3/vitb16-eomt-inst-coco",
        "dinov3/vitl16-eomt-inst-coco",
    ]

    for alias in aliases:
        checkpoint = (
            DINOV3_EOMT_INSTANCE_SEGMENTATION_MODEL_REGISTRY.get_alias_metadata(
                alias
            ).downloadable_checkpoint
        )
        assert task_model_helpers.DOWNLOADABLE_MODEL_URL_AND_HASH[alias] == (
            checkpoint.url,
            checkpoint.sha256,
        )


def test_downloadable_model__ltdetr_seg_aliases_from_registry() -> None:
    aliases = [
        "ltdetrv2-seg-s-coco",
        "ltdetrv2-seg-m-coco",
        "ltdetrv2-seg-l-coco",
        "ltdetrv2-seg-x-coco",
        "edgecrafter/ecvitt-ltdetr-seg-coco",
        "edgecrafter/ecvittplus-ltdetr-seg-coco",
        "edgecrafter/ecvits-ltdetr-seg-coco",
        "edgecrafter/ecvitsplus-ltdetr-seg-coco",
    ]

    for alias in aliases:
        checkpoint = LTDETR_SEG_MODEL_REGISTRY.get_alias_metadata(
            alias
        ).downloadable_checkpoint
        assert task_model_helpers.DOWNLOADABLE_MODEL_URL_AND_HASH[alias] == (
            checkpoint.url,
            checkpoint.sha256,
        )


def test_downloadable_model__ltdetr_seg_coco_short_and_full_aliases_match() -> None:
    # The ``ltdetrv2-seg-<size>-coco`` short names and the ``edgecrafter/...-seg-coco``
    # full names must resolve to the same hosted (file, hash) so downloads are
    # identical regardless of which name the user passes.
    d = task_model_helpers.DOWNLOADABLE_MODEL_URL_AND_HASH
    assert d["ltdetrv2-seg-s-coco"] == d["edgecrafter/ecvitt-ltdetr-seg-coco"]
    assert d["ltdetrv2-seg-m-coco"] == d["edgecrafter/ecvittplus-ltdetr-seg-coco"]
    assert d["ltdetrv2-seg-l-coco"] == d["edgecrafter/ecvits-ltdetr-seg-coco"]
    assert d["ltdetrv2-seg-x-coco"] == d["edgecrafter/ecvitsplus-ltdetr-seg-coco"]


def test_downloadable_model__picodet_aliases_from_registry() -> None:
    aliases = ["picodet-s-coco", "picodet-l-coco"]

    for alias in aliases:
        checkpoint = PICODET_OBJECT_DETECTION_MODEL_REGISTRY.get_alias_metadata(
            alias
        ).downloadable_checkpoint
        assert task_model_helpers.DOWNLOADABLE_MODEL_URL_AND_HASH[alias] == (
            checkpoint.url,
            checkpoint.sha256,
        )


def test_download_checkpoint__non_hosted_dav2__raises_convert_guidance() -> None:
    model_name = "dinov2/dav2-relative-large"

    with pytest.raises(ValueError) as exc_info:
        task_model_helpers.download_checkpoint(model_name)

    message = str(exc_info.value)
    assert model_name in message
    assert "non-commercial" in message
    assert "convert_checkpoint_dav2" in message


def test_download_checkpoint__unknown_name__raises_generic() -> None:
    model_name = "definitely-not-a-valid-model-name"

    with pytest.raises(ValueError) as exc_info:
        task_model_helpers.download_checkpoint(model_name)

    message = str(exc_info.value)
    assert f"Unknown model name or checkpoint path: '{model_name}'" in message
    assert "convert_checkpoint_dav2" not in message


def test_init_model_from_checkpoint__legacy_dinov2_uses_registered_decoder() -> None:
    model_name = "dinov2/vits14-noreg-ltdetr"
    reference_model = LTDETRObjectDetection(
        model_name=model_name,
        classes={0: "class_0", 1: "class_1"},
        image_size=(256, 256),
        load_weights=False,
    )
    train_state_dict = {
        f"model.{name}": param for name, param in reference_model.state_dict().items()
    }

    model = task_model_helpers.init_model_from_checkpoint(
        {
            "model_class_path": (
                "lightly_train._task_models.dinov2_ltdetr_object_detection.task_model"
                ".DINOv2LTDETRObjectDetection"
            ),
            "model_init_args": {
                "model_name": model_name,
                "classes": {0: "class_0", 1: "class_1"},
                "image_size": (256, 256),
                # This matches the hosted legacy checkpoint: decoder_name was not
                # stored, so the versioned model registry defines the architecture.
            },
            "train_model": train_state_dict,
        },
        device="cpu",
    )

    assert isinstance(model, LTDETRObjectDetection)
    assert model.init_args["model_name"] == model_name
    assert model.init_args["decoder_name"] is None
    assert isinstance(model.decoder, RTDETRTransformerv2)
    for name, param in model.state_dict().items():
        torch.testing.assert_close(param, reference_model.state_dict()[name])


@pytest.mark.parametrize(
    "checkpoint_classes, expected_reinit", [(2, True), (3, False), (5, True)]
)
def test_reinit_class_heads_on_mismatch__num_classes(
    checkpoint_classes: int, expected_reinit: bool
) -> None:
    head = Linear(4, 3)
    state_dict = {
        "model.head.weight": torch.full((checkpoint_classes, 4), 0.5),
        "model.head.bias": torch.full((checkpoint_classes,), 0.5),
    }
    task_model_helpers.reinit_class_heads_on_mismatch(
        state_dict=state_dict, prefix="model.", heads={"head": head}
    )
    if expected_reinit:
        torch.testing.assert_close(state_dict["model.head.weight"], head.weight)
        torch.testing.assert_close(state_dict["model.head.bias"], head.bias)
    else:
        assert torch.all(state_dict["model.head.weight"] == 0.5)
        assert torch.all(state_dict["model.head.bias"] == 0.5)


def test_reinit_class_heads_on_mismatch__copy() -> None:
    head = Linear(4, 3, dtype=torch.float64)
    state_dict = {"head.weight": torch.zeros(5, 4), "head.bias": torch.zeros(5)}
    task_model_helpers.reinit_class_heads_on_mismatch(
        state_dict=state_dict, prefix="", heads={"head": head}
    )
    weight = state_dict["head.weight"]
    assert weight.dtype == torch.float64
    assert not weight.requires_grad
    assert weight.data_ptr() != head.weight.data_ptr()


def test_reinit_class_heads_on_mismatch__module_without_bias() -> None:
    head = Linear(4, 3, bias=False)
    state_dict = {"head.weight": torch.zeros(5, 4), "head.bias": torch.zeros(5)}
    task_model_helpers.reinit_class_heads_on_mismatch(
        state_dict=state_dict, prefix="", heads={"head": head}
    )
    torch.testing.assert_close(state_dict["head.weight"], head.weight)
    # Left as is so that load_state_dict reports the unexpected key.
    assert state_dict["head.bias"].shape == (5,)


@pytest.mark.parametrize(
    "head, checkpoint_weight_shape",
    [
        (Conv2d(4, 3, kernel_size=1), (5, 4, 1, 1)),
        (Embedding(3, 4), (5, 4)),
    ],
)
def test_reinit_class_heads_on_mismatch__module_types(
    head: Module, checkpoint_weight_shape: tuple[int, ...]
) -> None:
    state_dict = {"head.weight": torch.zeros(checkpoint_weight_shape)}
    task_model_helpers.reinit_class_heads_on_mismatch(
        state_dict=state_dict, prefix="", heads={"head": head}
    )
    torch.testing.assert_close(state_dict["head.weight"], head.weight)


def test_reinit_class_heads_on_mismatch__missing_key() -> None:
    state_dict = {"other.weight": torch.zeros(1)}
    task_model_helpers.reinit_class_heads_on_mismatch(
        state_dict=state_dict, prefix="", heads={"head": Linear(4, 3)}
    )
    assert list(state_dict) == ["other.weight"]


def test_reinit_class_heads_on_mismatch__input_dim_differs() -> None:
    model = torch.nn.Sequential(Linear(4, 3))
    state_dict = {"0.weight": torch.zeros(5, 8), "0.bias": torch.zeros(5)}
    task_model_helpers.reinit_class_heads_on_mismatch(
        state_dict=state_dict, prefix="", heads={"0": model[0]}
    )
    assert state_dict["0.weight"].shape == (5, 8)
    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_state_dict(state_dict)


def test_reinit_class_heads_on_mismatch__log(caplog: pytest.LogCaptureFixture) -> None:
    heads = {"head.0": Linear(4, 3), "head.1": Linear(4, 3)}
    state_dict = {
        "head.0.weight": torch.zeros(5, 4),
        "head.1.weight": torch.zeros(5, 4),
    }
    with caplog.at_level(level="INFO"):
        task_model_helpers.reinit_class_heads_on_mismatch(
            state_dict=state_dict, prefix="", heads=heads
        )
    assert len(caplog.records) == 1
    assert "head.0, head.1" in caplog.records[0].getMessage()
