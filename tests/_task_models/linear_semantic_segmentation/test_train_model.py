#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch
from lightning_fabric import Fabric
from pytest_mock import MockerFixture

from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDataArgs,
    SingleChannelClassInfo,
    SplitArgs,
)
from lightly_train._metrics.semantic_segmentation.task_metric import (
    SemanticSegmentationTaskMetricArgs,
)
from lightly_train._task_models.linear_semantic_segmentation.train_model import (
    LinearSemanticSegmentationTrain,
    LinearSemanticSegmentationTrainArgs,
)
from lightly_train._task_models.linear_semantic_segmentation.transforms import (
    LinearSemanticSegmentationTrainTransformArgs,
    LinearSemanticSegmentationValTransformArgs,
)
from lightly_train._transforms.transform import NormalizeArgs
from lightly_train.types import MaskSemanticSegmentationBatch


def _make_train_model() -> LinearSemanticSegmentationTrain:
    data_args = MaskSemanticSegmentationDataArgs(
        train=SplitArgs(images="train/images", masks="train/masks"),
        val=SplitArgs(images="val/images", masks="val/masks"),
        classes={
            0: SingleChannelClassInfo(name="background", labels={0}),
            1: SingleChannelClassInfo(name="car", labels={1}),
        },
    )
    return LinearSemanticSegmentationTrain(
        model_name="dinov3/_vittest16-linear",
        model_args=LinearSemanticSegmentationTrainArgs(drop_path_rate=0.0),
        data_args=data_args,
        train_transform_args=LinearSemanticSegmentationTrainTransformArgs(
            ignore_index=data_args.ignore_index
        ),
        val_transform_args=LinearSemanticSegmentationValTransformArgs(
            image_size=(16, 16),
            normalize=NormalizeArgs(),
            ignore_index=data_args.ignore_index,
        ),
        load_weights=False,
        metric_args=SemanticSegmentationTaskMetricArgs(),
        gradient_accumulation_steps=1,
    )


class TestLinearSemanticSegmentationTrain:
    def test_validation_step__bounds_forward_batch_size(
        self, mocker: MockerFixture
    ) -> None:
        train_model = _make_train_model()
        train_model.eval()

        # Short side 16 (= crop size) with aspect ratios 2, 3, and 1, so tiling
        # yields 2 + 3 + 1 = 6 crops for a batch of 3 images.
        images = [
            torch.rand(3, 16, 32),
            torch.rand(3, 16, 48),
            torch.rand(3, 16, 16),
        ]
        masks = [torch.randint(0, 2, size=image.shape[-2:]) for image in images]
        batch: MaskSemanticSegmentationBatch = {
            "image_path": ["image_0.png", "image_1.png", "image_2.png"],
            "image": images,
            "mask": masks,
            "binary_masks": [],
        }

        spy = mocker.spy(train_model.model, "forward_train")
        with torch.no_grad():
            result = train_model.validation_step(
                fabric=Fabric(accelerator="cpu"), batch=batch, step=0
            )

        # The 6 crops must be forwarded in chunks of at most the batch size (3).
        assert spy.call_count == 2
        for call in spy.call_args_list:
            assert call.args[0].shape[0] <= len(images)

        assert result.loss.shape == ()
        assert torch.isfinite(result.loss)
