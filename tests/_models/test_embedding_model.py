#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

from lightly_train._models.embedding_model import EmbeddingModel
from tests import helpers


class TestEmbeddingModel:
    def test___init___with_embed_dim(self) -> None:
        feature_extractor = helpers.get_feature_extractor()
        embed_dim = 64
        model = EmbeddingModel(feature_extractor=feature_extractor, embed_dim=embed_dim)
        assert isinstance(model.embed_head, nn.Conv2d)
        assert model.embed_dim == embed_dim

    def test___init___without_embed_dim(self) -> None:
        feature_extractor = helpers.get_feature_extractor()
        model = EmbeddingModel(feature_extractor=feature_extractor, embed_dim=None)
        assert isinstance(model.embed_head, nn.Identity)
        assert model.embed_dim == feature_extractor.feature_dim()

    def test_forward__with_pooling(self) -> None:
        feature_extractor = helpers.get_feature_extractor()
        model = EmbeddingModel(feature_extractor=feature_extractor, embed_dim=64)
        x = torch.rand(2, 3, 32, 32)
        output = model(x, pool=True)
        assert output.shape == (2, 64, 1, 1)

    def test_forward__without_pooling(self) -> None:
        feature_extractor = helpers.get_feature_extractor()
        model = EmbeddingModel(feature_extractor=feature_extractor, embed_dim=64)
        x = torch.rand(2, 3, 32, 32)
        output = model(x, pool=False)
        assert output.shape == (2, 64, 31, 31)

    def test_forward__identity_head(self) -> None:
        feature_extractor = helpers.get_feature_extractor()
        model = EmbeddingModel(feature_extractor=feature_extractor, embed_dim=None)
        x = torch.rand(2, 3, 32, 32)
        output = model(x, pool=True)
        assert output.shape == (2, 2, 1, 1)
