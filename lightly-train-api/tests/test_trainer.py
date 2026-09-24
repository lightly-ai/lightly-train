#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch

from lightly_train_api import trainer


def test_dump_weights() -> None:
    weights = {"weight": torch.randn(3, 4), "bias": torch.randn(3)}

    loaded = trainer.load_weights(trainer.dump_weights(weights))

    assert set(loaded) == set(weights)
    assert all(torch.equal(loaded[key], weights[key]) for key in weights)


def test_dump_weights__detection_head() -> None:
    weights = {
        "decoder.enc_score_head.weight": torch.randn(2, 8),
        "decoder.enc_score_head.bias": torch.randn(2),
        "decoder.denoising_class_embed.weight": torch.randn(3, 8),
    }

    loaded = trainer.load_weights(trainer.dump_weights(weights))

    assert set(loaded) == set(weights)
    assert all(torch.equal(loaded[key], weights[key]) for key in weights)
