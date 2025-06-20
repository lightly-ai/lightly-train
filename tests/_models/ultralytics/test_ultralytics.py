#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from importlib import util as importlib_util

import pytest
import torch
from torch.nn import Identity

from lightly_train._models.ultralytics import ultralytics
from lightly_train._models.ultralytics.ultralytics import (
    YOLOV11_AVAILABLE,
    YOLOV12_ORIGINAL_AVAILABLE,
    YOLOV12_ULTRALYTICS_AVAILABLE,
    UltralyticsModelWrapper,
)

if importlib_util.find_spec("ultralytics") is None:
    pytest.skip("ultralytics is not installed", allow_module_level=True)

from ultralytics import YOLO
from ultralytics.nn.modules.block import SPPF, C2f


class TestUltralyticsModelWrapper:
    @pytest.mark.parametrize(
        "model_name", ["yolov8s.yaml", "yolov8s.pt", "yolov8s-cls.yaml"]
    )
    def test_init(self, model_name: str) -> None:
        model = YOLO(model_name)
        feature_extractor = UltralyticsModelWrapper(model=model)
        for name, param in feature_extractor.named_parameters():
            if ".dfl" in name:
                assert not param.requires_grad, name
            else:
                assert param.requires_grad, name

        for name, module in feature_extractor.named_modules():
            assert module.training, name

    @pytest.mark.parametrize("model_name", ["yolov8s.yaml", "yolov8s-cls.yaml"])
    def test_feature_dim(self, model_name: str) -> None:
        model = YOLO(model_name)
        feature_extractor = UltralyticsModelWrapper(model=model)
        assert feature_extractor.feature_dim() == 512

    @pytest.mark.parametrize(
        "model_name",
        [
            "yolov8s.yaml",
            "yolov8s-cls.yaml",
            pytest.param(
                "yolo11s.yaml",
                marks=pytest.mark.skipif(
                    not YOLOV11_AVAILABLE,
                    reason="YOLOv11 requires ultralytics>=8.3.0",
                ),
            ),
            pytest.param(
                "yolo11s-cls.yaml",
                marks=pytest.mark.skipif(
                    not YOLOV11_AVAILABLE,
                    reason="YOLOv11 requires ultralytics>=8.3.0",
                ),
            ),
            pytest.param(
                "yolo12s.yaml",
                marks=pytest.mark.skipif(
                    not YOLOV12_ULTRALYTICS_AVAILABLE,
                    reason="YOLOv12 requires ultralytics>=8.3.78 from the official source",
                ),
            ),
            pytest.param(
                "yolov12s.yaml",
                marks=pytest.mark.skipif(
                    not YOLOV12_ORIGINAL_AVAILABLE,
                    reason="YOLOv12 from the custom source",
                ),
            ),
        ],
    )
    def test_forward_features(self, model_name: str) -> None:
        model = YOLO(model_name)
        feature_extractor = UltralyticsModelWrapper(model=model)
        x = torch.rand(1, 3, 224, 224)
        features = feature_extractor.forward_features(x)["features"]
        assert features.shape == (1, 512, 7, 7)

    @pytest.mark.parametrize("model_name", ["yolov8s.yaml", "yolov8s-cls.yaml"])
    def test_forward_pool(self, model_name: str) -> None:
        model = YOLO(model_name)
        feature_extractor = UltralyticsModelWrapper(model=model)
        x = torch.rand(1, 512, 7, 7)
        pool = feature_extractor.forward_pool({"features": x})["pooled_features"]
        assert pool.shape == (1, 512, 1, 1)

    @pytest.mark.parametrize("model_name", ["yolov8s.yaml", "yolov8s-cls.yaml"])
    def test_get_model(self, model_name: str) -> None:
        model = YOLO(model_name)
        feature_extractor = UltralyticsModelWrapper(model=model)
        assert feature_extractor.get_model() is model

    def test__device(self) -> None:
        # If this test fails it means the wrapped model doesn't move all required
        # modules to the correct device. This happens if not all required modules
        # are registered as attributes of the class.
        model = YOLO("yolov8s.yaml")
        wrapped_model = UltralyticsModelWrapper(model=model)
        wrapped_model.to("meta")
        wrapped_model.forward_features(torch.rand(1, 3, 224, 224, device="meta"))


def test__sppf_skip_cv2_bn_act() -> None:
    sppf = SPPF(128, 5)
    new_sppf = ultralytics._sppf_skip_cv2_bn_act(sppf=sppf)
    assert new_sppf.cv1 is sppf.cv1
    assert new_sppf.cv2.conv is sppf.cv2.conv
    assert new_sppf.cv2.bn is not sppf.cv2.bn  # Updated
    assert new_sppf.cv2.act is not sppf.cv2.act  # Updated
    assert new_sppf.m is sppf.m
    assert not isinstance(sppf.cv2.bn, Identity)
    assert isinstance(new_sppf.cv2.bn, Identity)
    assert not isinstance(sppf.cv2.act, Identity)
    assert isinstance(new_sppf.cv2.act, Identity)


def test__c2f_skip_cv2_bn_act() -> None:
    c2f = C2f(128, 128)
    new_c2f = ultralytics._c2f_skip_cv2_bn_act(c2f=c2f)
    assert new_c2f.cv1 is c2f.cv1
    assert new_c2f.cv2.conv is c2f.cv2.conv
    assert new_c2f.cv2.bn is not c2f.cv2.bn  # Updated
    assert new_c2f.cv2.act is not c2f.cv2.act  # Updated
    assert new_c2f.m is c2f.m
    assert not isinstance(c2f.cv2.bn, Identity)
    assert isinstance(new_c2f.cv2.bn, Identity)
    assert not isinstance(c2f.cv2.act, Identity)
    assert isinstance(new_c2f.cv2.act, Identity)
