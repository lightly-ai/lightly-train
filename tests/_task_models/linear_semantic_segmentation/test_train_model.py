#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
import torch
from lightning_fabric import Fabric
from lightning_utilities.core.imports import RequirementCache
from pytest_mock import MockerFixture

if RequirementCache("torchmetrics<1.5"):
    # Skip test if torchmetrics version is too old. This can happen if SuperGradients
    # is installed which requires torchmetrics==0.8
    pytest.skip("Old torchmetrics version", allow_module_level=True)


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

        # Images are tiled, forwarded, and un-tiled one at a time so that at most a
        # single image's full-resolution logits are held on the device at once
        # (independent of the validation batch size). Each image's crops are then
        # forwarded in chunks of at most the batch size (3), so the three images
        # (2, 3, and 1 crops) each need exactly one forward call.
        assert spy.call_count == 3
        assert [call.args[0].shape[0] for call in spy.call_args_list] == [2, 3, 1]
        for call in spy.call_args_list:
            assert call.args[0].shape[0] <= len(images)

        assert result.loss.shape == ()
        assert torch.isfinite(result.loss)

    def test_validation_step__chunks_single_wide_image(
        self, mocker: MockerFixture
    ) -> None:
        train_model = _make_train_model()
        train_model.eval()

        # A single very wide image (short side 16 = crop size, aspect ratio 5)
        # tiles into 5 crops. Even with a batch of one image, forwarding all crops
        # at once is what runs out of memory, so the crops must still be forwarded
        # in chunks of at most the batch size (1).
        images = [torch.rand(3, 16, 80)]
        masks = [torch.randint(0, 2, size=image.shape[-2:]) for image in images]
        batch: MaskSemanticSegmentationBatch = {
            "image_path": ["image_0.png"],
            "image": images,
            "mask": masks,
            "binary_masks": [],
        }

        spy = mocker.spy(train_model.model, "forward_train")
        with torch.no_grad():
            result = train_model.validation_step(
                fabric=Fabric(accelerator="cpu"), batch=batch, step=0
            )

        # 5 crops forwarded one at a time (chunk size == batch size == 1).
        assert spy.call_count == 5
        for call in spy.call_args_list:
            assert call.args[0].shape[0] <= len(images)

        assert result.loss.shape == ()
        assert torch.isfinite(result.loss)
