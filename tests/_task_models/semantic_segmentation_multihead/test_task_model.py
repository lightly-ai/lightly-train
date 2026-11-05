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

from lightly_train._task_models.semantic_segmentation_multihead.task_model import (
    SemanticSegmentationMultihead,
)


@pytest.mark.parametrize("checkpoint_classes", [1, 2, 3])
def test_class_heads_reuse_or_reinit_hook(checkpoint_classes: int) -> None:
    def create_model(num_classes: int) -> SemanticSegmentationMultihead:
        return SemanticSegmentationMultihead(
            model="dinov2/_vittest14",
            classes={i: str(i) for i in range(num_classes)},
            head_names=["a", "b"],
            image_size=(14, 14),
            image_normalize=None,
            load_weights=False,
        )

    source = create_model(num_classes=checkpoint_classes)
    target = create_model(num_classes=2)
    initial = {k: v.clone() for k, v in target.state_dict().items()}
    target.load_state_dict(source.state_dict(), strict=True)
    checkpoint = source.state_dict()
    for key, value in target.state_dict().items():
        if key.startswith("class_heads.") and checkpoint_classes != 2:
            torch.testing.assert_close(value, initial[key])
        else:
            torch.testing.assert_close(value, checkpoint[key])
