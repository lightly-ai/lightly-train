#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from torch import Tensor

from lightly_train._transforms.batch_transform import BatchReplayCompose, BatchTransform
from lightly_train._transforms.scale_jitter import ScaleJitter


class TestBatchTransform:
    def test__call__(self) -> None:
        transform = BatchTransform(Compose([ToTensorV2()]))
        batch = [
            {"image": np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)},
            {"image": np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)},
        ]
        transformed = transform(batch)
        assert all(isinstance(item["image"], Tensor) for item in transformed)


class TestBatchReplayCompose:
    def test__call__(self) -> None:
        transform = BatchReplayCompose(
            transforms=[ScaleJitter(sizes=[(5, 5), (7, 7), (9, 9)])],
        )

        batch = [
            {
                "image": np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8),
                "mask": np.random.randint(0, 255, size=(8, 8), dtype=np.uint8),
            },
            {
                "image": np.random.randint(0, 255, size=(20, 20, 3), dtype=np.uint8),
                "mask": np.random.randint(0, 255, size=(20, 20), dtype=np.uint8),
            },
        ]

        # Test that transform generates same image sizes for all items in the same batch
        for _ in range(5):
            transformed = transform(batch)
            image_shape = transformed[0]["image"].shape
            mask_shape = transformed[0]["mask"].shape
            assert all(item["image"].shape == image_shape for item in transformed)
            assert all(item["mask"].shape == mask_shape for item in transformed)
            assert all("replay" not in item for item in transformed)

        # Test that transform generates different image sizes for different batches
        transformed = transform(batch)
        image_shape = transformed[0]["image"].shape
        for _ in range(100):
            transformed = transform(batch)
            if image_shape != transformed[0]["image"].shape:
                return
        assert False, f"Transformed images always have shape {image_shape}"
