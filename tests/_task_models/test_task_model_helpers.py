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
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.task_model import (
    DINOv3EoMTSemanticSegmentation,
)
from lightly_train._task_models.ltdetr_object_detection.config import (
    LTDETR_MODEL_REGISTRY,
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
    d = task_model_helpers.DOWNLOADABLE_MODEL_URL_AND_HASH
    assert "ltdetrv2-s-coco" not in d
    assert "edgecrafter/ecvitt-ltdetr-coco" not in d

    checkpoint = LTDETR_MODEL_REGISTRY.get_downloadable_checkpoint(
        name="ltdetrv2-s-coco"
    )
    legacy_checkpoint = LTDETR_MODEL_REGISTRY.get_downloadable_checkpoint(
        name="edgecrafter/ecvitt-ltdetr-coco"
    )
    # The modern and legacy names are registered as separate aliases, each with
    # its own DownloadableCheckpoint, but they point at the same weights.
    assert checkpoint.url == legacy_checkpoint.url
    assert checkpoint.sha256 == legacy_checkpoint.sha256
    assert checkpoint.url == "edgecrafter_ecvitt_ltdetr_coco_260624_f8aefe49.pt"
    assert (
        checkpoint.sha256
        == "f8aefe499be1579c55bfcb288f623399ea5f4efef0c5a5f00960663efeda4f49"
    )


def test_download_checkpoint__ltdetrv2_s_coco_uses_registry(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("LIGHTLY_TRAIN_MODEL_CACHE_DIR", str(tmp_path))
    checkpoint = LTDETR_MODEL_REGISTRY.get_downloadable_checkpoint(
        name="ltdetrv2-s-coco"
    )
    downloaded_urls: list[str] = []

    def fake_download_url_to_file(url: str, dst: str) -> None:
        downloaded_urls.append(url)
        Path(dst).write_bytes(b"checkpoint")

    monkeypatch.setattr(
        task_model_helpers.torch.hub,
        "download_url_to_file",
        fake_download_url_to_file,
    )
    monkeypatch.setattr(
        task_model_helpers,
        "checkpoint_hash",
        lambda path: checkpoint.sha256,
    )

    local_path = task_model_helpers.download_checkpoint("ltdetrv2-s-coco")

    assert local_path == tmp_path / checkpoint.url
    assert local_path.read_bytes() == b"checkpoint"
    assert downloaded_urls == [
        f"{task_model_helpers.DOWNLOADABLE_MODEL_BASE_URL}/{checkpoint.url}"
    ]


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
