#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pytest

from lightly_train._models.torchvision.torchvision_package import TorchvisionPackage


class TestTorchvisionPackage:
    @pytest.mark.parametrize(
        "model_name", ["torchvision/resnet18", "torchvision/convnext_small"]
    )
    def test_list_model_names(self, model_name: str) -> None:
        assert model_name in TorchvisionPackage.list_model_names()
