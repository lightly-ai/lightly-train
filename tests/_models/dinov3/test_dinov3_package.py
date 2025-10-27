#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from pathlib import Path

import pytest
import torch

from lightly_train._models.dinov3.dinov3_package import DINOv3Package
from lightly_train._models.dinov3.dinov3_vit import DINOv3ViTModelWrapper
from lightly_train._models.dinov3.dinov3_src.models.convnext import ConvNeXt
from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer,
)

from ...helpers import DummyCustomModel


class TestDINOv3Package:
    @pytest.mark.parametrize(
        "model_name, listed",
        [
            ("dinov3/vits16", True),
            ("dinov3/vits16plus", True),
            ("dinov3/vitb16", True),
            ("dinov3/vitl16", True),
            ("dinov3/vitl16-sat493m", True),
            ("dinov3/vit7b16", True),
            ("dinov3/vit7b16-sat493m", True),
            ("dinov3/convnext-tiny", True),
            ("dinov3/convnext-small", True),
            ("dinov3/convnext-base", True),
            ("dinov3/convnext-large", True),
        ],
    )
    def test_list_model_names(self, model_name: str, listed: bool) -> None:
        model_names = DINOv3Package.list_model_names()
        assert (model_name in model_names) is listed

    def test_export_model__unsupported_model(self, tmp_path: Path) -> None:
        model = DummyCustomModel().get_model()
        out_path = tmp_path / "model.pt"
        with pytest.raises(ValueError):
            DINOv3Package.export_model(model=model, out=out_path)