#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from torch import Tensor

from lightly_train._methods.distillationv2.distillationv2_transform import (
    DistillationV2Transform,
)
from lightly_train.types import TransformInput


class TestDistillationV2Transform:
    def test_transform_shapes(self) -> None:
        img_np = np.random.uniform(0, 255, size=(1234, 1234, 3))
        input: TransformInput = {
            "image": img_np.astype(np.uint8),
        }

        transform_args = DistillationV2Transform.transform_args_cls()()
        transform = DistillationV2Transform(transform_args)

        transformed = transform(input)
        assert len(transformed) == 1
        image = transformed[0]["image"]
        assert isinstance(image, Tensor)
        assert image.shape == (3, 224, 224)
