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

from lightly_train._task_models.picodet_object_detection.pico_head import PicoHead


@pytest.mark.parametrize("share_cls_reg", [True, False])
@pytest.mark.parametrize("checkpoint_classes", [2, 3, 5])
def test_picodet_gfl_cls_reuse_or_reinit_hook(
    share_cls_reg: bool, checkpoint_classes: int
) -> None:
    def create_head(num_classes: int) -> PicoHead:
        return PicoHead(
            in_channels=16,
            num_classes=num_classes,
            feat_channels=16,
            stacked_convs=1,
            strides=(8, 16),
            share_cls_reg=share_cls_reg,
        )

    source = create_head(num_classes=checkpoint_classes)
    target = create_head(num_classes=3)
    initial = {k: v.clone() for k, v in target.state_dict().items()}
    target.load_state_dict(source.state_dict(), strict=True)
    loaded = target.state_dict()
    checkpoint = source.state_dict()
    for key in loaded:
        if key.startswith("gfl_cls.") and checkpoint_classes != 3:
            torch.testing.assert_close(loaded[key], initial[key])
        else:
            torch.testing.assert_close(loaded[key], checkpoint[key])
