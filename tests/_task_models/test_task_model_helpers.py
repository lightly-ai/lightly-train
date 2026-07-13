#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from packaging import version
from pytest import MonkeyPatch
from torch.hub import download_url_to_file

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
from lightly_train._task_models.ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
)


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.2.0"),
    reason="Model loading currently fails for PyTorch < 2.2.0. See https://github.com/lightly-ai/lightly-train/issues/323",
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


def test_init_model_from_checkpoint__legacy_dinov2_ltdetr_reroutes_to_generic() -> None:
    reference_model = LTDETRObjectDetection(
        model_name="dinov3/vitt16-notpretrained-ltdetr",
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
                "model_name": "dinov3/vitt16-notpretrained-ltdetr",
                "classes": {0: "class_0", 1: "class_1"},
                "image_size": (256, 256),
            },
            "train_model": train_state_dict,
        },
        device="cpu",
    )

    assert isinstance(model, LTDETRObjectDetection)
    assert model.init_args["model_name"] == "dinov3/vitt16-notpretrained-ltdetr"
    assert model.init_args["decoder_name"] == "rtdetrv2"
    for name, param in model.state_dict().items():
        assert torch.equal(param, reference_model.state_dict()[name])
