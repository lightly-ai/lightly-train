#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Callable

import torch

from lightly_train_api import encoder

ImageBytes = Callable[[tuple[int, int, int]], bytes]


def test_encode(image_bytes: ImageBytes) -> None:
    images = [encoder.decode_image(image_bytes((255, 0, 0)))] * 2
    features = encoder.encode(images)
    assert features.shape == (2, encoder.feature_dim())
    assert torch.allclose(features[0], features[1])


def test_encode__deterministic(image_bytes: ImageBytes) -> None:
    image = encoder.decode_image(image_bytes((0, 128, 255)))
    assert torch.allclose(encoder.encode([image]), encoder.encode([image]))


def test_feature_to_blob() -> None:
    feature = torch.randn(encoder.feature_dim())
    assert torch.equal(
        encoder.blob_to_feature(encoder.feature_to_blob(feature)), feature
    )
