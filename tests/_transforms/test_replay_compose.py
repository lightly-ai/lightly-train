#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np

from lightly_train._transforms.replay_compose import ReplayCompose
from lightly_train._transforms.scale_jitter import ScaleJitter


class TestReplayCompose:
    def test_transform_batch(self) -> None:
        transform = ReplayCompose(
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
            transformed = transform.transform_batch(batch)
            image_shape = transformed[0]["image"].shape
            mask_shape = transformed[0]["mask"].shape
            assert all(item["image"].shape == image_shape for item in transformed)
            assert all(item["mask"].shape == mask_shape for item in transformed)

        # Test that transform generates different image sizes for different batches
        transformed = transform.transform_batch(batch)
        image_shape = transformed[0]["image"].shape
        for _ in range(100):
            transformed = transform.transform_batch(batch)
            if image_shape != transformed[0]["image"].shape:
                return
        assert False, f"Transformed images has always shape {image_shape}"
