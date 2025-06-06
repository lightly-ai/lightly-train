#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest

try:
    from super_gradients.training import models
except ImportError:
    # We do not use pytest.importorskip on module level because it makes mypy unhappy.
    pytest.skip("super_gradients is not installed", allow_module_level=True)

from super_gradients.training.models import (
    PPLiteSegT,
    YoloNAS_S,
)

from lightly_train._models.super_gradients.customizable_detector import (
    CustomizableDetectorModelWrapper,
)
from lightly_train._models.super_gradients.segmentation_module import (
    SegmentationModuleModelWrapper,
)
from lightly_train._models.super_gradients.super_gradients_package import (
    SuperGradientsPackage,
)

from ...helpers import DummyCustomModel


class TestSuperGradientsPackage:
    def test_list_model_names(self) -> None:
        model_names = SuperGradientsPackage.list_model_names()
        assert "super_gradients/yolo_nas_s" in model_names
        assert "super_gradients/yolox_n" not in model_names
        assert "super_gradients/pp_lite_t_seg50" in model_names
        assert "super_gradients/ddrnet_23" not in model_names

    @pytest.mark.parametrize(
        "model_name, is_supported",
        [
            ("yolo_nas_s", True),
            ("yolox_n", False),
            ("pp_lite_t_seg50", True),
            ("ddrnet_23", False),
        ],
    )
    def test_is_supported_model__model(
        self, model_name: str, is_supported: bool
    ) -> None:
        model = models.get(model_name, num_classes=2)
        assert SuperGradientsPackage.is_supported_model(model) is is_supported

    @pytest.mark.parametrize(
        "model_name, is_supported",
        [
            ("yolo_nas_s", True),
            ("pp_lite_t_seg50", True),
        ],
    )
    def test_is_supported_model__wrapped_model(
        self, model_name: str, is_supported: bool
    ) -> None:
        model = models.get(model_name, num_classes=2)
        wrapped_model = SuperGradientsPackage.get_model_wrapper(model=model)
        assert SuperGradientsPackage.is_supported_model(wrapped_model)

    def test_is_supported_model__unsupported_model(self) -> None:
        model = DummyCustomModel()
        assert not SuperGradientsPackage.is_supported_model(model)
        assert not SuperGradientsPackage.is_supported_model(model.get_model())

    @pytest.mark.parametrize(
        "model_name, expected_cls",
        [("yolo_nas_s", YoloNAS_S), ("pp_lite_t_seg50", PPLiteSegT)],
    )
    def test_get_model(self, model_name: str, expected_cls: type) -> None:
        model = SuperGradientsPackage.get_model(model_name)
        assert isinstance(model, expected_cls)

    @pytest.mark.parametrize(
        "model_name, expected_cls",
        [
            ("yolo_nas_s", CustomizableDetectorModelWrapper),
            ("pp_lite_t_seg50", SegmentationModuleModelWrapper),
        ],
    )
    def test_get_model_wrapper(self, model_name: str, expected_cls: type) -> None:
        model = models.get(model_name, num_classes=2)
        fe = SuperGradientsPackage.get_model_wrapper(model)
        assert isinstance(fe, expected_cls)
