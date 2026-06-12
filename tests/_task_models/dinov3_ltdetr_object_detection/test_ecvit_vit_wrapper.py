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

from lightly_train._models.ecvit.ecvit import ECViTWrapper
from lightly_train._task_models.dinov3_ltdetr_object_detection.ecvit_vit_wrapper import (
    ECViTBackboneWrapper,
)


class TestECViTBackboneWrapper:
    def test_patch_size_is_16(self) -> None:
        # ECViT uses a ConvPyramidPatchEmbed with a fixed patch size of 16; the
        # wrapper exposes that fixed value for the train/val transforms.
        ecvit = ECViTWrapper(name="ecvitt", depth=1, interaction_indexes=[0])
        wrapper = ECViTBackboneWrapper(model_wrapper=ecvit)
        assert wrapper.patch_size == 16

    def test_backbone_model_returns_wrapped_ecvit(self) -> None:
        ecvit = ECViTWrapper(name="ecvitt", depth=1, interaction_indexes=[0])
        wrapper = ECViTBackboneWrapper(model_wrapper=ecvit)
        assert wrapper.backbone_model is ecvit

    @pytest.mark.parametrize(
        "name",
        ["ecvitt", "ecvittplus", "ecvits", "ecvitsplus"],
    )
    def test_forward_matches_ecvit_wrapper_output(self, name: str) -> None:
        # The wrapper must be a pass-through: forward(x) == ECViTWrapper.forward(x).
        # We use a small model (depth=1, tiny dims) to keep the test fast and
        # independent of any pretrained weight download.
        ecvit = ECViTWrapper(
            name=name,
            depth=1,
            interaction_indexes=[0],
            embed_dim=16,
            num_heads=1,
            proj_dim=16,
        )
        ecvit.eval()
        wrapper = ECViTBackboneWrapper(model_wrapper=ecvit)
        wrapper.eval()

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            wrapped_out = wrapper(x)
            ecvit_out = ecvit(x)

        assert isinstance(wrapped_out, tuple)
        assert len(wrapped_out) == 3
        for w, e in zip(wrapped_out, ecvit_out):
            assert torch.equal(w, e)

    def test_wrapper_does_not_expose_mask_token(self) -> None:
        # Unlike the DINOv3 ViT-based DINOv3STAs, ECViT has no mask_token.
        # The task model constructor must not try to freeze one on the ECViT
        # branch (see DINOv3LTDETRObjectDetection.__init__).
        ecvit = ECViTWrapper(name="ecvitt", depth=1, interaction_indexes=[0])
        wrapper = ECViTBackboneWrapper(model_wrapper=ecvit)
        assert not hasattr(wrapper, "mask_token")
        assert not hasattr(wrapper.backbone_model, "mask_token")
