#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Sequence

import torch
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torch import Tensor
from torch.nn import Identity, Module, ModuleList

from lightly_train._models.dinov2_vit.dinov2_vit_src.layers.block import Block
from lightly_train._models.dinov2_vit.dinov2_vit_src.models.vision_transformer import (
    DinoVisionTransformer,
)
from lightly_train._models.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    MultiScaleFeatureViT,
)


class DINOv2ViTModelWrapper(Module, MultiScaleFeatureViT, ArchitectureInfoGettable):
    _input_mean: Tensor
    _input_std: Tensor

    def __init__(self, model: DinoVisionTransformer) -> None:
        super().__init__()
        self._model = model
        self._feature_dim = int(self._model.embed_dim)
        self.register_buffer(
            "_input_mean",
            torch.tensor(IMAGENET_NORMALIZE["mean"]).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_input_std",
            torch.tensor(IMAGENET_NORMALIZE["std"]).view(1, -1, 1, 1),
            persistent=False,
        )

    def feature_dim(self) -> int:
        return self._feature_dim

    def patch_size(self) -> int:
        return int(self._model.patch_size)

    def forward_features(
        self, x: Tensor, masks: Tensor | None = None, n_blocks: int = 1
    ) -> ForwardFeaturesOutput:
        x = self._prepare_input(x)
        if n_blocks > 1:
            # ViT blocks all produce the same spatial resolution — no interpolation needed.
            x_list = list(
                self._model.get_intermediate_layers(
                    x, n=n_blocks, reshape=True, return_class_token=True
                )
            )
            features = torch.cat([feat for feat, _ in x_list], dim=1)  # (B, n*D, H, W)
            cls_token = torch.cat([cls for _, cls in x_list], dim=1)  # (B, n*D)
            return {"features": features, "cls_token": cls_token}

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
        # TIPSv2 repurposes its sole DINOv2 register token as a second class token.
        # The common LightlyTrain wrapper API exposes the primary class token only;
        # the second token remains internal and is intentionally not a multimodal API.
        return {"pooled_features": x["cls_token"][..., None, None]}

    def get_model(self) -> DinoVisionTransformer:
        return self._model

    def make_teacher(self) -> None:
        if self._model.chunked_blocks:
            for chunked_blocks in self._model.blocks:
                update_blocks_student_to_teacher(chunked_blocks)  # type: ignore[arg-type]
        else:
            update_blocks_student_to_teacher(self._model.blocks)

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "transformer", "norm_type": "layernorm"}

    def multiscale_feature_dims(self) -> list[int]:
        return [self._feature_dim] * self._model.n_blocks

    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Sequence[int]
    ) -> list[ForwardFeaturesOutput]:
        x = self._prepare_input(x)
        rt = self._model.get_intermediate_layers(
            x, n=list(layer_indices), reshape=True, return_class_token=True
        )
        return [{"features": feat, "cls_token": cls} for feat, cls in rt]

    def _prepare_input(self, x: Tensor) -> Tensor:
        if self._model.input_normalization == "none":
            # TiPSv2 is trained on images in [0, 1], while LightlyTrain's default
            # transforms apply ImageNet normalization.
            return x * self._input_std.to(dtype=x.dtype) + self._input_mean.to(
                dtype=x.dtype
            )
        return x


def update_blocks_student_to_teacher(blocks: ModuleList) -> None:
    for block in blocks:
        # For FSDP the blocks are grouped in chunks to reduce the peak
        # memory usage after un-sharding. The chunking makes it more
        # complicated to access the individual blocks as a consequence,
        # chunks are padded with Identity layers.
        assert isinstance(block, Block) or isinstance(block, Identity)
        if isinstance(block, Identity):
            continue
        block.drop_path1 = Identity()
        block.drop_path2 = Identity()
        block.sample_drop_ratio = 0.0
