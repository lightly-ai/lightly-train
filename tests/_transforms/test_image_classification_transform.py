#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._configs import validate
from lightly_train._task_models.image_classification.transforms import (
    ImageClassificationTrainTransformArgs,
)
from lightly_train._transforms.image_classification_transform import (
    ImageClassificationTransform,
)


class TestImageClassificationTransform:
    def test_ratio_is_passed_through(self) -> None:
        # Fine-tuning shares the crop builder with pretraining, so the aspect ratio
        # range reaches the transform here as well.
        args = validate.pydantic_model_validate(
            ImageClassificationTrainTransformArgs,
            {"random_crop": {"min_ratio": 0.5, "max_ratio": 2.0}},
        )
        args.resolve_auto(model_init_args={})
        args.resolve_incompatible()
        transform = ImageClassificationTransform(transform_args=args)
        crops = [
            t
            for t in transform.transform.transforms
            if type(t).__name__ == "RandomResizedCrop"
        ]
        assert len(crops) == 1
        assert crops[0].ratio == (0.5, 2.0)
