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
from lightly_train._task_models.dinov3_eomt_semantic_segmentation.task_model import (
    DINOv3EoMTSemanticSegmentation,
)


@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.2.0"),
    reason="Model loading currently fails for PyTorch < 2.2.0. See https://github.com/lightly-ai/lightly-train/issues/323",
)
def test_load_model__download(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LIGHTLY_TRAIN_MODEL_CACHE_DIR", str(tmp_path))
    model_name = "dinov3/vits16-eomt-coco"
    model_file_name = "lightlytrain_dinov3_eomt_vits16_cocostuff.pt"
    expected_model_type = DINOv3EoMTSemanticSegmentation
    expected_model_name = "dinov3/vits16-eomt"

    with patch(
        "torch.hub.download_url_to_file", wraps=download_url_to_file
    ) as spy_download_url_to_file:
        model = load_model(model_name)

        assert (tmp_path / model_file_name).is_file()
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
