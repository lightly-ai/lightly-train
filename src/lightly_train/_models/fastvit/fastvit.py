#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from lightly_train._models.model_wrapper import (
    ArchitectureInfo,
    ArchitectureInfoGettable,
    ForwardFeaturesOutput,
    ForwardPoolOutput,
    MultiScaleFeatureCNN,
)

logger = logging.getLogger(__name__)

# Network indices that correspond to the end of each stage in FastViT's
# self.network ModuleList.  The network alternates [stage_blocks, downsample, ...]
# so stage 0 ends at index 0, stage 1 at 2, stage 2 at 4, stage 3 at 6.
_FASTVIT_STAGE_INDICES = [0, 2, 4, 6]


class FastViTModelWrapper(Module, MultiScaleFeatureCNN, ArchitectureInfoGettable):
    def __init__(self, model: Module) -> None:
        for attr in ("forward_embeddings", "network"):
            if not hasattr(model, attr):
                raise ValueError(f"Model must have a '{attr}' attribute")
        super().__init__()
        self._model = model
        self._cached_feature_dims: list[int] | None = None
        self._cached_feature_strides: list[int] | None = None
        self._cached_pooled_dim: int | None = None

    def feature_dim(self) -> int:
        if self._cached_pooled_dim is not None:
            return self._cached_pooled_dim
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                x = torch.randn(1, 3, 64, 64)
                out = self.forward_features(x)
                pooled = self.forward_pool(out)
                self._cached_pooled_dim = pooled["pooled_features"].flatten(1).shape[1]
        finally:
            if was_training:
                self._model.train()
        return self._cached_pooled_dim

    def forward_features(self, x: Tensor) -> ForwardFeaturesOutput:
        x = self._model.forward_embeddings(x)  # type: ignore[operator]
        for module in self._model.network:  # type: ignore[operator, union-attr]
            x = module(x)
        if hasattr(self._model, "conv_exp"):
            x = self._model.conv_exp(x)  # type: ignore[operator]
        else:
            # FastViT's fork_feat mode exposes normalized stage outputs. Apply the
            # final-stage norm here too so forward_features agrees with its pyramid.
            norm_name = f"norm{_FASTVIT_STAGE_INDICES[-1]}"
            if hasattr(self._model, norm_name):
                x = getattr(self._model, norm_name)(x)
        return {"features": x}

    def forward_pool(self, x: ForwardFeaturesOutput) -> ForwardPoolOutput:
        features = F.adaptive_avg_pool2d(x["features"], output_size=1)
        return {"pooled_features": features}

    def get_model(self) -> Module:
        return self._model

    def architecture_info(self) -> ArchitectureInfo:
        return {"model_type": "hybrid", "norm_type": "batchnorm"}  # type: ignore[return-value]

    def multiscale_feature_dims(self) -> list[int]:
        if self._cached_feature_dims is not None:
            return self._cached_feature_dims
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                x = torch.randn(1, 3, 64, 64)
                x = self._model.forward_embeddings(x)  # type: ignore[operator]
                dims: list[int] = []
                net_idx = 0
                for net_end_idx in _FASTVIT_STAGE_INDICES:
                    while net_idx <= net_end_idx:
                        x = self._model.network[net_idx](x)  # type: ignore[operator, index]
                        net_idx += 1
                    dims.append(x.shape[1])
        finally:
            if was_training:
                self._model.train()
        self._cached_feature_dims = dims
        return dims

    def multiscale_feature_strides(self) -> list[int]:
        if self._cached_feature_strides is not None:
            return self._cached_feature_strides
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                h_in, w_in = 64, 64
                x = torch.randn(1, 3, h_in, w_in)
                x = self._model.forward_embeddings(x)  # type: ignore[operator]
                strides: list[int] = []
                net_idx = 0
                for net_end_idx in _FASTVIT_STAGE_INDICES:
                    while net_idx <= net_end_idx:
                        x = self._model.network[net_idx](x)  # type: ignore[operator, index]
                        net_idx += 1
                    out_h = x.shape[-2]
                    strides.append(h_in // out_h)
        finally:
            if was_training:
                self._model.train()
        self._cached_feature_strides = strides
        return strides

    def forward_multiscale_features(
        self, x: Tensor, layer_indices: Sequence[int]
    ) -> list[ForwardFeaturesOutput]:
        requested = set(layer_indices)
        assert requested.issubset(set(range(len(_FASTVIT_STAGE_INDICES)))), (
            f"layer_indices must be in [0, {len(_FASTVIT_STAGE_INDICES) - 1}], "
            f"got {list(requested)}"
        )

        x = self._model.forward_embeddings(x)  # type: ignore[operator]
        results: dict[int, Tensor] = {}
        net_idx = 0
        for stage_idx, net_end_idx in enumerate(_FASTVIT_STAGE_INDICES):
            while net_idx <= net_end_idx:
                x = self._model.network[net_idx](x)  # type: ignore[operator, index]
                net_idx += 1
            if stage_idx in requested:
                # Apply per-stage norm if available (when fork_feat is enabled).
                norm_name = f"norm{_FASTVIT_STAGE_INDICES[stage_idx]}"
                if hasattr(self._model, norm_name):
                    norm = getattr(self._model, norm_name)
                    x_stage = norm(x)
                else:
                    x_stage = x
                results[stage_idx] = x_stage

        return [{"features": results[idx]} for idx in layer_indices]
