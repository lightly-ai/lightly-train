#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from torch import Tensor
from torch.nn import Conv2d, Identity, Module

from lightly_train._models.model_wrapper import ModelWrapper


# EmbeddingModel is not combined into a single class with ModelWrapper to keep
# implementing new extractors as simple as possible.
#
# Note that in the future we might want to support feature extractors that also generate
# features from intermediate layers, for this we'll have to add support for multiple
# embedding heads with different dimensions.
class EmbeddingModel(Module):
    def __init__(
        self,
        wrapped_model: ModelWrapper,
        embed_dim: None | int = None,
    ):
        """A model that extracts features from input data and maps them to an embedding
        space.

        Args:
            model_wrapper:
                A feature extractor that implements the `ModelWrapper` interface.
            embed_dim:
                The dimensionality of the embedding space. If None, the output of the
                feature extractor is used as the embedding.
            pool:
                Whether to apply the pooling layer of the feature extractor. If False,
                the features are embedded and returned without pooling.
        """
        super().__init__()
        self.wrapped_model = wrapped_model
        self.embed_head = (
            Identity()
            if embed_dim is None
            else Conv2d(
                in_channels=self.wrapped_model.feature_dim(),
                out_channels=embed_dim,
                kernel_size=1,
            )
        )

    @property
    def embed_dim(self) -> int:
        if isinstance(self.embed_head, Identity):
            return self.wrapped_model.feature_dim()
        else:
            out_channels: int = self.embed_head.out_channels
            return out_channels

    def forward(
        self,
        x: Tensor,
        pool: bool = True,
    ) -> Tensor:
        """Extract features from input image and map them to an embedding space.

        Args:
            x: Input images with shape (B, C, H_in, W_in).

        Returns:
            Embeddings with shape (B, embed_dim, H_out, W_out). H_out and W_out depend
            on the pooling layer of the feature extractor and are 1 in most cases.
        """
        features_out = self.wrapped_model.forward_features(x)
        x = features_out["features"]
        if pool:
            x = self.wrapped_model.forward_pool(features_out)["pooled_features"]
        x = self.embed_head(x)
        return x
