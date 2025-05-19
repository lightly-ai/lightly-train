#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import copy

import pytest
import torch

from lightly_train._models.dinov2_vit.dinov2_vit import DINOv2ViTModelWrapper
from lightly_train._models.dinov2_vit.dinov2_vit_package import DINOv2ViTPackage
from lightly_train._models.dinov2_vit.dinov2_vit_src.layers.drop_path import DropPath
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    vit_small as vit_small,
)


class TestDINOv2ViTModelWrapper:
    def test_init(self) -> None:
        model = vit_small()
        feature_extractor = DINOv2ViTModelWrapper(model=model)

        for name, param in feature_extractor.named_parameters():
            assert param.requires_grad, name

        for name, module in feature_extractor.named_modules():
            assert module.training, name

    def test_feature_dim(self) -> None:
        model = vit_small()
        feature_extractor = DINOv2ViTModelWrapper(model=model)

        assert feature_extractor.feature_dim() == 384

    def test_forward_features(self) -> None:
        model = vit_small()
        feature_extractor = DINOv2ViTModelWrapper(model=model)

        x = torch.rand(1, 3, 224, 224)
        features = feature_extractor.forward_features(x)["features"]
        cls_token = feature_extractor.forward_features(x)["cls_token"]
        assert features.shape == (1, 384, 14, 14)
        assert cls_token.shape == (1, 384)

    def test_forward_pool(self) -> None:
        model = vit_small()
        feature_extractor = DINOv2ViTModelWrapper(model=model)

        x = torch.rand(1, 384, 14, 14)
        pooled_features = feature_extractor.forward_pool({"features": x})[
            "pooled_features"
        ]
        assert pooled_features.shape == (1, 384, 1, 1)

    def test_get_model(self) -> None:
        model = vit_small()
        extractor = DINOv2ViTModelWrapper(model=model)
        assert extractor.get_model() is model

    @pytest.mark.parametrize(
        "model_name",
        ["vits14"],
    )
    def test_make_teacher(self, model_name: str) -> None:
        student = DINOv2ViTPackage.get_model(model_name)
        feature_extractor = DINOv2ViTModelWrapper(model=copy.deepcopy(student))
        feature_extractor.make_teacher()
        teacher = feature_extractor.get_model()

        #  Ensure models are the same expect for the drop paths
        assert len(list(student.parameters())) == len(list(teacher.parameters()))
        for (name_student, param_student), (name_teacher, param_student) in zip(
            student.named_parameters(), teacher.named_parameters()
        ):
            assert name_student == name_teacher
            assert param_student.dtype == param_student.dtype
            assert param_student.requires_grad == param_student.requires_grad
            assert torch.allclose(param_student, param_student, rtol=1e-3, atol=1e-4)

        for student_block, teacher_block in zip(student.blocks, teacher.blocks):
            assert isinstance(student_block.drop_path1, DropPath)
            assert isinstance(student_block.drop_path2, DropPath)
            assert student_block.sample_drop_ratio > 0.0
            assert isinstance(teacher_block.drop_path1, torch.nn.Identity)
            assert isinstance(teacher_block.drop_path2, torch.nn.Identity)
            assert teacher_block.sample_drop_ratio == 0.0
