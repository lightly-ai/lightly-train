#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, Identity, Module, ModuleList

from lightly_train._models.dinov3.dinov3_src.layers.block import (
    SelfAttentionBlock,
)
from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._models.model_wrapper import (
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    ModelWrapper,
)


class DINOv3ViTModelWrapper(Module, ModelWrapper):
    def __init__(self, model: DinoVisionTransformer) -> None:
        super().__init__()
        self._model = model
        self._feature_dim = int(self._model.embed_dim)
        self._pool = AdaptiveAvgPool2d((1, 1))

    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(
        self, x: Tensor, masks: Tensor | None = None
    ) -> ForwardFeaturesOutput:
        rt = self._model(x, masks, is_training=True)  # forcing to return all patches
        if rt["x_norm_patchtokens"].dim() == 3:
            x_norm_patchtokens = rt["x_norm_patchtokens"]
            b = x_norm_patchtokens.shape[0]
            d = x_norm_patchtokens.shape[2]
            h = x.shape[2] // self._model.patch_size
            w = x.shape[3] // self._model.patch_size

            features_reshaped = x_norm_patchtokens.permute(0, 2, 1).reshape(b, d, h, w)
        elif rt["x_norm_patchtokens"].dim() == 4:
            features_reshaped = rt["x_norm_patchtokens"]
        else:
            raise ValueError(
                f"Unexpected shape for x_norm_patchtokens: {rt['x_norm_patchtokens'].shape}"
            )
        return {"features": features_reshaped, "cls_token": rt["x_norm_clstoken"]}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        return {"pooled_features": self._pool(x["features"])}

    def get_model(self) -> DinoVisionTransformer:
        return self._model

    def make_teacher(self) -> None:
        if self._model.chunked_blocks:
            for chunked_blocks in self._model.blocks:
                update_blocks_student_to_teacher(chunked_blocks)
        else:
            update_blocks_student_to_teacher(self._model.blocks)


def update_blocks_student_to_teacher(blocks: ModuleList) -> None:
    for block in blocks:
        # For FSDP the blocks are grouped in chunks to reduce the peak
        # memory usage after un-sharding. The chunking makes it more
        # complicated to access the individual blocks as a consequence,
        # chunks are padded with Identity layers.
        assert isinstance(block, SelfAttentionBlock) or isinstance(block, Identity)
        if isinstance(block, Identity):
            continue
        block.drop_path1 = Identity()
        block.drop_path2 = Identity()
        block.sample_drop_ratio = 0.0
