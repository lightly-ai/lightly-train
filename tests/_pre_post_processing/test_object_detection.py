#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import numpy as np
import pytest
import torch
from lightlytrain_deploy_py.pre_post_processing import (
    ObjectDetectionMetadata as DeployObjectDetectionMetadata,
)
from lightlytrain_deploy_py.pre_post_processing import (
    ObjectDetectionPostprocessor as DeployObjectDetectionPostprocessor,
)
from lightlytrain_deploy_py.pre_post_processing import (
    ObjectDetectionPreprocessor as DeployObjectDetectionPreprocessor,
)
from PIL import Image

from lightly_train._pre_post_processing.object_detection import (
    ObjectDetectionMetadata as TrainObjectDetectionMetadata,
)
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

    def test_preprocess_image__matches_deploy_preprocessor_for_rgb_pil(self) -> None:
        pre = ObjectDetectionPreprocessor(
            image_size=(8, 10),
            image_normalize=None,
            expected_input_channels=3,
        )
        deploy_pre = DeployObjectDetectionPreprocessor(
            image_size=(8, 10),
            image_normalize=None,
            expected_input_channels=3,
        )
        image_np = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        image = Image.fromarray(image_np)

        x, meta = pre.preprocess_image(
            image, device=torch.device("cpu"), dtype=torch.float32
        )
        deploy_x, deploy_meta = deploy_pre.preprocess_image(image)

        assert meta == deploy_meta
        np.testing.assert_allclose(
            x.numpy(),
            deploy_x,
            atol=3 / 255,
            rtol=0,
        )

    def test_preprocess_image__matches_deploy_preprocessor_for_grayscale_pil(
        self,
    ) -> None:
        pre = ObjectDetectionPreprocessor(
            image_size=(6, 6),
            image_normalize=None,
            expected_input_channels=3,
        )
        deploy_pre = DeployObjectDetectionPreprocessor(
            image_size=(6, 6),
            image_normalize=None,
            expected_input_channels=3,
        )
        image_np = np.arange(9 * 7, dtype=np.uint8).reshape(9, 7)
        image = Image.fromarray(image_np)

        x, meta = pre.preprocess_image(
            image, device=torch.device("cpu"), dtype=torch.float32
        )
        deploy_x, deploy_meta = deploy_pre.preprocess_image(image)

        assert meta == deploy_meta
        np.testing.assert_allclose(
            x.numpy(),
            deploy_x,
            atol=3 / 255,
            rtol=0,
        )

    def test_preprocess_batch__matches_deploy_preprocessor(self) -> None:
        pre = ObjectDetectionPreprocessor(
            image_size=(2, 2),
            image_normalize={
                "mean": (0.2, 0.4, 0.6),
                "std": (0.5, 0.25, 0.125),
            },
            expected_input_channels=3,
        )
        deploy_pre = DeployObjectDetectionPreprocessor(
            image_size=(2, 2),
            image_normalize={
                "mean": (0.2, 0.4, 0.6),
                "std": (0.5, 0.25, 0.125),
            },
            expected_input_channels=3,
        )
        batch_np = np.linspace(0.0, 1.0, num=2 * 3 * 2 * 2, dtype=np.float32).reshape(
            2, 3, 2, 2
        )
        batch = torch.from_numpy(batch_np)

        out = pre.preprocess_batch(batch)
        deploy_out = deploy_pre.preprocess_batch(batch_np)

        np.testing.assert_allclose(out.numpy(), deploy_out, atol=1e-6, rtol=1e-6)

    def test_preprocess_sahi_image__returns_batch_and_metadata(self) -> None:
        pre = ObjectDetectionPreprocessor(
            image_size=(4, 6),
            image_normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
            expected_input_channels=3,
        )
        image = torch.zeros(3, 8, 10, dtype=torch.uint8)
        batch, meta = pre.preprocess_sahi_image(
            image, device=torch.device("cpu"), dtype=torch.float32, overlap=0.5
        )

        assert batch.shape == (10, 3, 4, 6)
        assert batch.dtype == torch.float32
        torch.testing.assert_close(batch, torch.zeros_like(batch))
        assert meta["orig_h"] == 8
        assert meta["orig_w"] == 10
        assert meta["tiles_coordinates"].shape == (9, 2)

    def test_preprocess_sahi_image__uses_shared_channel_validation(self) -> None:
        pre = ObjectDetectionPreprocessor(
            image_size=(4, 4),
            image_normalize=None,
            expected_input_channels=3,
        )
        image = torch.rand(1, 8, 8)
        batch, _ = pre.preprocess_sahi_image(
            image, device=torch.device("cpu"), dtype=torch.float32, overlap=0.0
        )
        assert batch.shape[1] == 3

        with pytest.raises(ValueError, match="channels"):
            pre.preprocess_sahi_image(
                torch.rand(2, 8, 8),
                device=torch.device("cpu"),
                dtype=torch.float32,
                overlap=0.0,
            )

    def test_preprocess_sahi_batch__applies_batch_preprocessing(self) -> None:
        pre = ObjectDetectionPreprocessor(
            image_size=(4, 6),
            image_normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
            expected_input_channels=3,
        )
        batch = torch.zeros(12, 3, 4, 6)
        out = pre.preprocess_sahi_batch(batch)

        assert out.shape == (12, 3, 4, 6)
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, torch.full_like(batch, -1.0))


def _make_postprocessor() -> ObjectDetectionPostprocessor:
    return ObjectDetectionPostprocessor(
        num_classes=2,
        num_top_queries=5,
        # Map internal ids {0, 1} to user ids {10, 20}.
        internal_class_to_class=torch.tensor([10, 20], dtype=torch.long),
    )


def _make_parity_raw_output() -> ObjectDetectionOutput:
    return ObjectDetectionOutput(
        logits=torch.tensor(
            [
                [
                    [-2.0, 1.5, 0.1],
                    [3.0, -0.5, 2.0],
                    [0.7, 4.0, -1.0],
                    [2.5, 0.3, 1.2],
                ],
                [
                    [1.1, -1.5, 2.7],
                    [-0.2, 3.2, 0.8],
                    [2.2, 1.7, -2.0],
                    [0.4, -0.7, 3.5],
                ],
            ],
            dtype=torch.float32,
        ),
        boxes=torch.tensor(
            [
                [
                    [0.50, 0.50, 0.20, 0.40],
                    [0.25, 0.60, 0.30, 0.20],
                    [0.70, 0.35, 0.10, 0.30],
                    [0.40, 0.80, 0.25, 0.10],
                ],
                [
                    [0.20, 0.40, 0.30, 0.20],
                    [0.80, 0.70, 0.20, 0.30],
                    [0.55, 0.45, 0.40, 0.15],
                    [0.30, 0.75, 0.10, 0.20],
                ],
            ],
            dtype=torch.float32,
        ),
    )


def _make_parity_postprocessor() -> ObjectDetectionPostprocessor:
    return ObjectDetectionPostprocessor(
        num_classes=3,
        num_top_queries=5,
        internal_class_to_class=torch.tensor([10, 20, 30], dtype=torch.long),
    )


def _make_deploy_parity_postprocessor() -> DeployObjectDetectionPostprocessor:
    return DeployObjectDetectionPostprocessor(
        num_classes=3,
        num_top_queries=5,
        internal_class_to_class=np.array([10, 20, 30], dtype=np.int64),
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

    def test_decode__matches_deploy_postprocessor(self) -> None:
        post = _make_parity_postprocessor()
        deploy_post = _make_deploy_parity_postprocessor()
        raw = _make_parity_raw_output()
        orig_sizes = torch.tensor([[100, 200], [50, 80]], dtype=torch.int64)

        labels, boxes, scores = post.decode(raw, orig_sizes)
        deploy_labels, deploy_boxes, deploy_scores = deploy_post.decode(
            raw.logits.numpy(),
            raw.boxes.numpy(),
            orig_sizes.numpy(),
        )

        np.testing.assert_array_equal(labels.numpy(), deploy_labels)
        np.testing.assert_allclose(boxes.numpy(), deploy_boxes, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(scores.numpy(), deploy_scores, atol=1e-6, rtol=1e-6)

    def test_postprocess__matches_deploy_postprocessor(self) -> None:
        post = _make_parity_postprocessor()
        deploy_post = _make_deploy_parity_postprocessor()
        raw = _make_parity_raw_output()
        metadata: list[TrainObjectDetectionMetadata] = [
            {"orig_w": 100, "orig_h": 200},
            {"orig_w": 50, "orig_h": 80},
        ]
        deploy_metadata: list[DeployObjectDetectionMetadata] = [
            {"orig_w": 100, "orig_h": 200},
            {"orig_w": 50, "orig_h": 80},
        ]

        out = post.postprocess(raw, metadata=metadata, threshold=0.8)
        deploy_out = deploy_post.postprocess(
            raw.logits.numpy(),
            raw.boxes.numpy(),
            metadata=deploy_metadata,
            threshold=0.8,
        )

        assert len(out) == len(deploy_out)
        for result, deploy_result in zip(out, deploy_out):
            assert set(result.keys()) == set(deploy_result.keys())
            np.testing.assert_array_equal(
                result["labels"].numpy(), deploy_result["labels"]
            )
            np.testing.assert_allclose(
                result["bboxes"].numpy(),
                deploy_result["bboxes"],
                atol=1e-6,
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                result["scores"].numpy(),
                deploy_result["scores"],
                atol=1e-6,
                rtol=1e-6,
            )

    def test_postprocess_sahi__decodes_offsets_filters_and_merges(self) -> None:
        post = ObjectDetectionPostprocessor(
            num_classes=1,
            num_top_queries=1,
            internal_class_to_class=torch.tensor([7], dtype=torch.long),
        )
        raw = ObjectDetectionOutput(
            logits=torch.tensor([[[10.0]], [[9.0]], [[-10.0]]]),
            boxes=torch.tensor(
                [
                    [[0.5, 0.5, 0.2, 0.2]],
                    [[0.5, 0.5, 0.2, 0.2]],
                    [[0.5, 0.5, 0.2, 0.2]],
                ]
            ),
        )
        out = post.postprocess_sahi(
            raw,
            {
                "orig_h": 50,
                "orig_w": 100,
                "tiles_coordinates": torch.tensor([[5, 7], [30, 20]]),
            },
            threshold=0.5,
            nms_iou_threshold=0.3,
            global_local_iou_threshold=0.1,
            tile_size=(10, 20),
        )

        torch.testing.assert_close(out["labels"], torch.tensor([7, 7]))
        torch.testing.assert_close(
            out["bboxes"],
            torch.tensor([[40.0, 20.0, 60.0, 30.0], [13.0, 11.0, 17.0, 13.0]]),
        )
        assert out["scores"].shape == (2,)
