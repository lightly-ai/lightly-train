#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch

from lightly_train._task_models.ltdetr_object_detection.fastvit_wrapper import (
    FastViTBackboneWrapper,
)

from ...helpers import dummy_fastvit_model


def test_forward__returns_stride_8_16_32_features() -> None:
    wrapper = FastViTBackboneWrapper(model_wrapper=dummy_fastvit_model())

    features = wrapper(torch.randn(1, 3, 64, 64))

    assert [feature.shape[1] for feature in features] == [96, 192, 384]
    assert [feature.shape[-1] for feature in features] == [8, 4, 2]
