#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Callable

import pytest
from lightning_utilities.core.imports import RequirementCache

from lightly_train._methods.distillationv3.distillationv3 import (
    DistillationV3AdamWArgs,
    _is_probably_conv,
)
from lightly_train._models import package_helpers
from lightly_train._models.model_wrapper import ModelWrapper

from ...helpers import (
    DummyCustomModel,
    DummyCustomModelWithArchInfo,
    dummy_dinov2_vit_model,
    dummy_dinov3_convnext_model,
    dummy_dinov3_vit_model,
)

_TIMM_AVAILABLE = RequirementCache("timm")
_ULTRALYTICS_AVAILABLE = RequirementCache("ultralytics")


@pytest.mark.skipif(
    not _TIMM_AVAILABLE or not _ULTRALYTICS_AVAILABLE,
    reason="timm and ultralytics must be installed",
)
@pytest.mark.parametrize(
    "model, expected",
    [
        # Traditional CNNs: True
        (lambda: DummyCustomModel(), True),
        ("torchvision/resnet18", True),
        ("torchvision/resnet50", True),
        ("torchvision/shufflenet_v2_x0_5", True),
        ("torchvision/shufflenet_v2_x1_0", True),
        ("ultralytics/yolov8n.pt", True),
        ("timm/resnet18", True),
        ("timm/resnet50", True),
        # Transformers: False
        (dummy_dinov2_vit_model, False),
        (dummy_dinov3_vit_model, False),
        ("timm/vit_tiny_patch16_224", False),
        # ConvNeXt-style (LayerNorm, transformer training recipe): False
        (dummy_dinov3_convnext_model, False),
        ("torchvision/convnext_tiny", False),
        ("torchvision/convnext_small", False),
        ("timm/convnext_tiny", False),
    ],
)
def test_is_probably_conv(model: str | ModelWrapper, expected: bool) -> None:
    if callable(model):
        model = model()
    elif isinstance(model, str):
        model = package_helpers.get_wrapped_model(model=model, num_input_channels=3)
    assert _is_probably_conv(wrapped_model=model) is expected


@pytest.mark.parametrize(
    "make_model, expected_weight_decay",
    [
        # ArchitectureInfoGettable models: exact weight_decay from arch_info
        (
            lambda: DummyCustomModelWithArchInfo(
                {"model_type": "transformer", "norm_type": "layernorm"}
            ),
            0.04,
        ),
        (
            lambda: DummyCustomModelWithArchInfo(
                {"model_type": "hybrid", "norm_type": "layernorm"}
            ),
            0.04,
        ),
        (
            lambda: DummyCustomModelWithArchInfo(
                {"model_type": "convolutional", "norm_type": "batchnorm"}
            ),
            1e-6,
        ),
        (
            lambda: DummyCustomModelWithArchInfo(
                {"model_type": "convolutional", "norm_type": "layernorm"}
            ),
            0.04,
        ),
        # Non-ArchitectureInfoGettable model: heuristic fallback.
        # DummyCustomModel has a Conv2d and no attention/layernorm → is_conv=True → 1e-6.
        (lambda: DummyCustomModel(), 1e-6),
    ],
)
def test_distillationv3_adamw_args__resolve_auto(
    make_model: Callable[[], ModelWrapper], expected_weight_decay: float
) -> None:
    model = make_model()
    args = DistillationV3AdamWArgs()
    args.resolve_auto(wrapped_model=model)
    assert args.weight_decay == pytest.approx(expected_weight_decay)
