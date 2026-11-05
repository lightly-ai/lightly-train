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
from torch.nn import Linear, Module

from lightly_train import _torch_helpers
from lightly_train._task_models.eomt import hooks


class _Model(Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.proj = Linear(4, 4)
        self.class_head = Linear(4, num_classes + 1)
        _torch_helpers.register_load_state_dict_pre_hook(
            self, hooks.class_head_reuse_or_reinit_hook
        )


@pytest.mark.parametrize("checkpoint_classes", [2, 3, 5])
def test_class_head_reuse_or_reinit_hook(checkpoint_classes: int) -> None:
    source = _Model(num_classes=checkpoint_classes)
    target = _Model(num_classes=3)
    initial = target.class_head.weight.detach().clone()
    target.load_state_dict(source.state_dict(), strict=True)
    expected = initial if checkpoint_classes != 3 else source.class_head.weight
    torch.testing.assert_close(target.class_head.weight, expected)
    torch.testing.assert_close(target.proj.weight, source.proj.weight)
