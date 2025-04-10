#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

import lightly_train._plot as _plot
from lightly_train.types import Batch


def test_plot_example_augmentations() -> None:
    view_sizes = [(3, 128, 32), (3, 64, 64), (3, 64, 256)]
    n_images = 4
    expected_grid_height = 128 + 10
    expected_grid_width = 256 + 10
    expected_image_height = expected_grid_height * n_images + 30
    expected_image_width = expected_grid_width * len(view_sizes) + 100

    batch_per_view = [torch.rand(n_images, *size) for size in view_sizes]
    multi_view_batch: Batch = {
        "views": batch_per_view,
        "filename": [f"img_{i}" for i in range(n_images)],
    }

    pil_image = _plot.plot_example_augmentations(
        train_batch=multi_view_batch, max_examples=5
    )
    assert pil_image.size == (expected_image_width, expected_image_height)
