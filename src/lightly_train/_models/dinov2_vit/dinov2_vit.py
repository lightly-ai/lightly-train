#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy

from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, Identity, Module

from lightly_train._models.model_wrapper import (
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
)
from lightly_train._modules.teachers.dinov2.layers.block import Block


class DINOv2ViTModelWrapper(Module, ModelWrapper):
    def __init__(self, model: Module) -> None:
        super().__init__()
        self._model = model
        self._feature_dim = int(self._model.embed_dim)
        self._pool = AdaptiveAvgPool2d((1, 1))
    
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        return {"features": self._model(x)}
    
    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def get_teacher(self) -> Module:
        teacher = copy.deepcopy(self._model)
        if teacher.chunked_blocks:
            for chunked_blocks in teacher.blocks:
                update_blocks_student_to_teacher(chunked_blocks)
        else:
            update_blocks_student_to_teacher(teacher.blocks)             
        return teacher

def update_blocks_student_to_teacher(blocks: list[Block]) -> None:
    for block in blocks:
        block.drop_path1 = Identity()
        block.drop_path2 = Identity()
        block.sample_drop_ratio = 0.0