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
from torchvision import tv_tensors
from torchvision.transforms import v2

from lightly_train._transforms.torchvision_dispatcher import (
    SeededRandomChoice,
    TorchVisionScaleJitter,
)


@pytest.fixture
def dummy_tv_image() -> tv_tensors.Image:
    return tv_tensors.Image(torch.rand(3, 100, 100))


@pytest.fixture
def dummy_tv_obb() -> tv_tensors.BoundingBoxes:
    return tv_tensors.BoundingBoxes(  # type: ignore[call-arg]
        torch.tensor([[50.0, 50.0, 20.0, 10.0, 45.0]]),
        format=tv_tensors.BoundingBoxFormat.CXCYWHR,
        canvas_size=(100, 100),
    )


@pytest.fixture
def example_seeded_random_choice_resizes() -> SeededRandomChoice:
    transform = SeededRandomChoice(
        transforms=[
            v2.Resize(size=(50, 50)),
            v2.Resize(size=(30, 30)),
        ],
        seed=42,
    )
    return transform


class TestSeededRandomChoice:
    def test_same_seed(
        self,
        dummy_tv_obb: tv_tensors.BoundingBoxes,
        dummy_tv_image: tv_tensors.Image,
        example_seeded_random_choice_resizes: SeededRandomChoice,
    ) -> None:
        output1 = example_seeded_random_choice_resizes(dummy_tv_image, dummy_tv_obb)
        output2 = example_seeded_random_choice_resizes(dummy_tv_image, dummy_tv_obb)

        assert isinstance(output1, tuple)
        assert isinstance(output2, tuple)
        assert len(output1) == 2
        assert len(output2) == 2
        assert torch.equal(output1[0], output2[0])
        assert torch.equal(output1[1], output2[1])

    def test_different_seed(
        self,
        dummy_tv_obb: tv_tensors.BoundingBoxes,
        dummy_tv_image: tv_tensors.Image,
        example_seeded_random_choice_resizes: SeededRandomChoice,
    ) -> None:
        for i in range(100):
            output1 = example_seeded_random_choice_resizes(dummy_tv_image, dummy_tv_obb)
            example_seeded_random_choice_resizes.step()
            output2 = example_seeded_random_choice_resizes(dummy_tv_image, dummy_tv_obb)

            if not torch.equal(output1[0], output2[0]):
                break
        else:
            raise AssertionError(
                "Transform did not change after 100 steps, expected a change."
            )


class TestTorchVisionScaleJitter:
    def test_same_seed(
        self,
        dummy_tv_obb: tv_tensors.BoundingBoxes,
        dummy_tv_image: tv_tensors.Image,
    ) -> None:
        scale_jitter = TorchVisionScaleJitter(
            target_size=(100, 100),
            scale_range=(0.5, 1.5),
            num_scales=5,
            seed=42,
        )

        with scale_jitter.same_seed():
            output1 = scale_jitter(dummy_tv_image, dummy_tv_obb)
            output2 = scale_jitter(dummy_tv_image, dummy_tv_obb)

        assert isinstance(output1, tuple)
        assert isinstance(output2, tuple)
        assert torch.equal(output1[0], output2[0])

    def test_different_seed(
        self,
        dummy_tv_obb: tv_tensors.BoundingBoxes,
        dummy_tv_image: tv_tensors.Image,
    ) -> None:
        scale_jitter = TorchVisionScaleJitter(
            target_size=(100, 100),
            scale_range=(0.5, 1.5),
            num_scales=5,
            seed=42,
        )

        outputs = []
        for i in range(10):
            with scale_jitter.same_seed():
                output = scale_jitter(dummy_tv_image, dummy_tv_obb)
                outputs.append(output[0].shape)

        unique_shapes = set(outputs)
        assert len(unique_shapes) > 1, "Expected different sizes across iterations"
