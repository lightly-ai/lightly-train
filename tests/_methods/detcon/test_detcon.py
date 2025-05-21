#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from torch import Size

from lightly_train._methods.detcon import detcon
from lightly_train._methods.detcon.detcon import (
    DetConB,
    DetConBArgs,
    DetConBSGDArgs,
    DetConS,
    DetConSArgs,
    DetConSSGDArgs,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train.types import Batch

from ...helpers import DummyCustomModel


def test__subsample_mask_indices() -> None:
    """Test whether the subsample chooses the correct indices, and whether the the
    probability of choosing the class equals the probability of the class appearing in
    the mask.
    """
    masks = torch.tensor([[[0, 1], [2, 3]], [[1, 1], [2, 3]]])  # masks for batch_size=2
    indices = []
    for i in range(10000):
        index = detcon._subsample_mask_indices(masks, 4, 2)
        indices.append(index)
    indices0 = (
        torch.stack([index[0] for index in indices]).view(-1).float()
    )  # indices for first sample
    indices1 = (
        torch.stack([index[1] for index in indices]).view(-1).float()
    )  # indices for second sample
    uniques0, counts0 = torch.unique(indices0, return_counts=True)
    uniques1, counts1 = torch.unique(indices1, return_counts=True)

    # only the classes appearing in `masks` from above
    assert (uniques0 == torch.tensor([0, 1, 2, 3])).all()
    assert (uniques1 == torch.tensor([1, 2, 3])).all()

    # normalize to probabilities
    counts0 = counts0 / counts0.sum()
    counts1 = counts1 / counts1.sum()
    assert torch.isclose(
        counts0, torch.tensor([0.25, 0.25, 0.25, 0.25]), atol=0.05
    ).all()
    assert torch.isclose(counts1, torch.tensor([0.33, 0.33, 0.33]), atol=0.05).all()


def test__subsample_mask_indices__weighted() -> None:
    """Test whether the subsample chooses the correct indices, and whether the the
    probability of choosing the class equals the probability of the class appearing in
    the mask.
    """
    masks = torch.tensor([[[0, 1], [2, 3]], [[1, 1], [2, 3]]])  # masks for batch_size=2
    indices = []
    for i in range(10000):
        index = detcon._subsample_mask_indices(
            masks, 4, 2, class_weighted_sampling=True
        )
        indices.append(index)
    indices0 = (
        torch.stack([index[0] for index in indices]).view(-1).float()
    )  # indices for first sample
    indices1 = (
        torch.stack([index[1] for index in indices]).view(-1).float()
    )  # indices for second sample
    uniques0, counts0 = torch.unique(indices0, return_counts=True)
    uniques1, counts1 = torch.unique(indices1, return_counts=True)

    # only the classes appearing in `masks` from above
    assert (uniques0 == torch.tensor([0, 1, 2, 3])).all()
    assert (uniques1 == torch.tensor([1, 2, 3])).all()

    # normalize to probabilities
    counts0 = counts0 / counts0.sum()
    counts1 = counts1 / counts1.sum()
    assert torch.isclose(
        counts0, torch.tensor([0.25, 0.25, 0.25, 0.25]), atol=0.05
    ).all()
    assert torch.isclose(counts1, torch.tensor([0.5, 0.25, 0.25]), atol=0.05).all()


def test__subsample_pooled_features() -> None:
    """Test whether the subsample chooses the correct indices, by comparing it to a
    manual calculation.
    """
    b, n_total_cls, c = 4, 100, 2
    pooled_feature = torch.randn(
        (b, n_total_cls, c)
    )  # output of embedding model + masked pooling
    indices = torch.tensor(
        [[3, 1], [2, 1], [50, 61], [99, 98]]
    )  # define 2 random classes to keep per sample

    # manually stack pooled features from only those classes
    exp_sample0 = torch.stack([pooled_feature[0, 3], pooled_feature[0, 1]])
    exp_sample1 = torch.stack([pooled_feature[1, 2], pooled_feature[1, 1]])
    exp_sample2 = torch.stack([pooled_feature[2, 50], pooled_feature[2, 61]])
    exp_sample3 = torch.stack([pooled_feature[3, 99], pooled_feature[3, 98]])
    exp_samples = torch.stack([exp_sample0, exp_sample1, exp_sample2, exp_sample3])

    samples = detcon._subsample_pooled_features(pooled_feature, indices)
    assert (samples == exp_samples).all()


class TestDetConS:
    def test_training_step_impl(self) -> None:
        emb_model = EmbeddingModel(wrapped_model=DummyCustomModel())
        b = 16

        view0 = torch.rand(b, 3, 8, 8)
        view1 = torch.rand(b, 3, 8, 8)
        mask0 = torch.randint(0, 256, (b, 224, 224))
        mask1 = torch.randint(0, 256, (b, 224, 224))
        batch: Batch = {
            "views": [view0, view1],
            "masks": [mask0, mask1],
            "filename": [f"img_{i}" for i in range(b)],
        }

        # run DetConS
        detcons_args = DetConSArgs()
        detcons = DetConS(
            method_args=detcons_args,
            optimizer_args=DetConSSGDArgs(),
            embedding_model=emb_model,
            global_batch_size=b,
        )

        out = detcons.training_step_impl(batch, 0)
        assert out.loss.shape == Size([])


class TestDetConB:
    def test_training_step_impl(self) -> None:
        emb_model = EmbeddingModel(wrapped_model=DummyCustomModel())
        b = 16

        view0 = torch.rand(b, 3, 8, 8)
        view1 = torch.rand(b, 3, 8, 8)
        mask0 = torch.randint(0, 256, (b, 224, 224))
        mask1 = torch.randint(0, 256, (b, 224, 224))
        batch: Batch = {
            "views": [view0, view1],
            "masks": [mask0, mask1],
            "filename": [f"img_{i}" for i in range(b)],
        }

        # run DetConB
        detconb_args = DetConBArgs()
        detconb = DetConB(
            method_args=detconb_args,
            optimizer_args=DetConBSGDArgs(),
            embedding_model=emb_model,
            global_batch_size=b,
        )
        out = detconb.training_step_impl(batch, 0)
        assert out.loss.shape == Size([])
