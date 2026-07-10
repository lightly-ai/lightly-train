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

from lightly_train._pre_post_processing.object_detection import (
    ObjectDetectionOutput,
    ObjectDetectionPostprocessor,
    ObjectDetectionPreprocessor,
)


class TestObjectDetectionPreprocessor:
    def test_preprocess_image__resizes_and_returns_orig_size(self) -> None:
        pre = ObjectDetectionPreprocessor(
            image_size=(256, 256),
            image_normalize=None,
            expected_input_channels=3,
        )
        image = torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8)
        x, meta = pre.preprocess_image(
            image, device=torch.device("cpu"), dtype=torch.float32
        )
        assert x.shape == (3, 256, 256)
        assert x.dtype == torch.float32
        # to_dtype(..., scale=True) maps uint8 -> [0, 1].
        assert x.min() >= 0.0 and x.max() <= 1.0
        assert meta == {"orig_h": 480, "orig_w": 640}

    def test_preprocess_image__expands_grayscale(self) -> None:
        pre = ObjectDetectionPreprocessor(
            image_size=(64, 64),
            image_normalize=None,
            expected_input_channels=3,
        )
        image = torch.rand(1, 32, 32)
        x, _ = pre.preprocess_image(
            image, device=torch.device("cpu"), dtype=torch.float32
        )
        assert x.shape == (3, 64, 64)

    def test_preprocess_image__raises_on_channel_mismatch(self) -> None:
        pre = ObjectDetectionPreprocessor(
            image_size=(64, 64),
            image_normalize=None,
            expected_input_channels=3,
        )
        image = torch.rand(2, 32, 32)
        with pytest.raises(ValueError, match="channels"):
            pre.preprocess_image(image, device=torch.device("cpu"), dtype=torch.float32)

    def test_preprocess_batch__applies_normalization(self) -> None:
        pre = ObjectDetectionPreprocessor(
            image_size=(8, 8),
            image_normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
            expected_input_channels=3,
        )
        batch = torch.zeros(2, 3, 8, 8)
        out = pre.preprocess_batch(batch)
        # (0 - 0.5) / 0.5 == -1.
        torch.testing.assert_close(out, torch.full_like(batch, -1.0))

    def test_preprocess_batch__noop_without_normalize(self) -> None:
        pre = ObjectDetectionPreprocessor(
            image_size=(8, 8),
            image_normalize=None,
            expected_input_channels=3,
        )
        batch = torch.rand(2, 3, 8, 8)
        torch.testing.assert_close(pre.preprocess_batch(batch), batch)


def _make_postprocessor() -> ObjectDetectionPostprocessor:
    return ObjectDetectionPostprocessor(
        num_classes=2,
        num_top_queries=5,
        # Map internal ids {0, 1} to user ids {10, 20}.
        internal_class_to_class=torch.tensor([10, 20], dtype=torch.long),
    )


class TestObjectDetectionPostprocessor:
    def test_postprocess__remaps_classes_and_keeps_all(self) -> None:
        post = _make_postprocessor()
        raw = ObjectDetectionOutput(
            logits=torch.randn(1, 5, 2),
            boxes=torch.rand(1, 5, 4),
        )
        # threshold=-1 keeps every prediction.
        out = post.postprocess(
            raw, metadata=[{"orig_w": 100, "orig_h": 200}], threshold=-1.0
        )
        assert len(out) == 1
        result = out[0]
        assert set(result.keys()) == {"labels", "bboxes", "scores"}
        assert result["labels"].numel() == 5
        assert result["bboxes"].shape == (5, 4)
        # Labels are remapped into the user class-id space.
        assert set(int(label) for label in result["labels"]).issubset({10, 20})

    def test_postprocess__threshold_filters_everything(self) -> None:
        post = _make_postprocessor()
        raw = ObjectDetectionOutput(
            logits=torch.randn(1, 5, 2),
            boxes=torch.rand(1, 5, 4),
        )
        # Sigmoid scores are always < 1, so threshold=1 removes everything.
        out = post.postprocess(
            raw, metadata=[{"orig_w": 100, "orig_h": 200}], threshold=1.0
        )
        assert out[0]["labels"].numel() == 0
        assert out[0]["bboxes"].shape == (0, 4)

    def test_decode__applies_class_remap(self) -> None:
        post = _make_postprocessor()
        raw = ObjectDetectionOutput(
            logits=torch.randn(1, 5, 2),
            boxes=torch.rand(1, 5, 4),
        )
        orig_sizes = torch.tensor([[100, 200]], dtype=torch.int64)
        labels, boxes, scores = post.decode(raw, orig_sizes)
        assert labels.shape == (1, 5)
        assert boxes.shape == (1, 5, 4)
        assert scores.shape == (1, 5)
        assert set(int(label) for label in labels.flatten()).issubset({10, 20})
