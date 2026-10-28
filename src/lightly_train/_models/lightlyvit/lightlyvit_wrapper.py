#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
"""LightlyTrain model-wrapper integration for LightlyViT."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from lightly_train._models.lightlyvit.lightlyvit import (
    LightlyViT,
    LightlyViTBackboneOutput,
)
from lightly_train._models.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    MultiScaleFeatureViT,
    SupportsActivationCheckpointing,
)


class LightlyViTModelWrapper(
    nn.Module,
    MultiScaleFeatureViT,
    ArchitectureInfoGettable,
    SupportsActivationCheckpointing,
):
    """Expose LightlyViT through LightlyTrain's standard feature contracts."""

    def __init__(self, model: LightlyViT) -> None:
        super().__init__()
        self._model = model

    def set_activation_checkpointing(
        self, enabled: bool, every_n_blocks: int = 1
    ) -> None:
        self._model._activation_checkpointing = enabled
        self._model._activation_checkpointing_every_n_blocks = every_n_blocks

    def feature_dim(self) -> int:
        return self._model.embed_dim

    def patch_size(self) -> int:
        return self._model.patch_size

    def forward_internal(
        self, x: Tensor, masks: Tensor | None = None
    ) -> LightlyViTBackboneOutput:
        """Return the complete typed backbone output for internal consumers."""

        return self._model.forward_features(x, masks=masks)

    def forward_features(
        self, x: Tensor, masks: Tensor | None = None, n_blocks: int = 1
    ) -> ForwardFeaturesOutput:
        if n_blocks < 1 or n_blocks > self._model.num_blocks:
            raise ValueError(
                f"n_blocks must be in [1, {self._model.num_blocks}], got {n_blocks}."
            )
        if n_blocks == 1:
            output = self.forward_internal(x, masks=masks)
            return {"features": output.features, "cls_token": output.cls_token}
        outputs = self._model.get_intermediate_layers(
            x,
            n=n_blocks,
            reshape=True,
            return_class_token=True,
            masks=masks,
        )
        features = [output[0] for output in outputs]
        cls_tokens = [output[1] for output in outputs]
        return {
            "features": torch.cat(features, dim=1),
            "cls_token": torch.cat(cls_tokens, dim=1),
        }

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        cls_token = x.get("cls_token")
        if cls_token is None:
            raise ValueError("LightlyViT CLS pooling requires a 'cls_token' feature.")
        return {"pooled_features": cls_token[..., None, None]}

    def get_model(self) -> LightlyViT:
        return self._model

    def make_teacher(self) -> None:
        """Disable stochastic depth for deterministic teacher features."""

        for block in self._model.blocks:
            block.drop_path1 = nn.Identity()
            block.drop_path2 = nn.Identity()

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "hybrid", "norm_type": "layernorm"}

    def multiscale_feature_dims(self) -> list[int]:
        return [self._model.embed_dim] * self._model.num_blocks

    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Sequence[int]
    ) -> list[ForwardFeaturesOutput]:
        outputs = self._model.forward_intermediate_features(x, layer_indices)
        return [
            {"features": output.features, "cls_token": output.cls_token}
            for output in outputs
        ]
