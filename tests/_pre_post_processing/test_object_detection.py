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
    ObjectDetectionMetadata,
    ObjectDetectionOutput,
    ObjectDetectionPostprocessor,
    ObjectDetectionPrediction,
    ObjectDetectionPreprocessor,
    ObjectDetectionSahiConfig,
)


class TestObjectDetectionPreprocessor:
    def test_preprocess_image__resizes_scales_and_returns_metadata(self) -> None:
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(32, 48), image_normalize=None, expected_input_channels=3
        )
        image = torch.randint(0, 256, (3, 60, 80), dtype=torch.uint8)

        output, metadata = preprocessor.preprocess_image(
            image, device=torch.device("cpu"), dtype=torch.float32
        )

        assert output.shape == (3, 32, 48)
        assert output.dtype == torch.float32
        assert output.min() >= 0 and output.max() <= 1
        assert metadata == ObjectDetectionMetadata(orig_h=60, orig_w=80)

    def test_preprocess_image__validates_channels(self) -> None:
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(16, 16), image_normalize=None, expected_input_channels=3
        )
        grayscale, _ = preprocessor.preprocess_image(
            torch.rand(1, 8, 8),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert grayscale.shape == (3, 16, 16)
        with pytest.raises(ValueError, match="channels"):
            preprocessor.preprocess_image(
                torch.rand(2, 8, 8),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_preprocess_batch__normalizes(self) -> None:
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(4, 4),
            image_normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
            expected_input_channels=3,
        )
        output = preprocessor.preprocess_batch(torch.zeros(2, 3, 4, 4))
        torch.testing.assert_close(output, torch.full_like(output, -1))

    def test_preprocess_sahi_image__returns_global_and_tiles(self) -> None:
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(4, 6),
            image_normalize=None,
            expected_input_channels=3,
        )
        batch, metadata = preprocessor.preprocess_image(
            torch.zeros(3, 8, 10, dtype=torch.uint8),
            device=torch.device("cpu"),
            dtype=torch.float32,
            sahi_config=ObjectDetectionSahiConfig(
                overlap=0.5,
                nms_iou_threshold=0.3,
                global_local_iou_threshold=0.1,
            ),
        )
        assert batch.shape == (10, 3, 4, 6)
        assert metadata.orig_h == 8
        assert metadata.orig_w == 10
        assert metadata.tile_coordinates is not None
        assert metadata.tile_coordinates.shape == (9, 2)


def _prediction() -> ObjectDetectionPrediction:
    return ObjectDetectionPrediction(
        labels=torch.tensor([17, 3, 17]),
        bboxes=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        scores=torch.tensor([0.9, 0.4, 0.8]),
    )


class TestObjectDetectionPrediction:
    def test_getitem__filters_by_score(self) -> None:
        prediction = _prediction()

        kept = prediction[prediction.scores > 0.5]

        torch.testing.assert_close(kept.labels, torch.tensor([17, 17]))
        torch.testing.assert_close(kept.scores, torch.tensor([0.9, 0.8]))
        assert kept.bboxes.shape == (2, 4)

    def test_getitem__filters_by_label(self) -> None:
        prediction = _prediction()

        cats = prediction[prediction.labels == 17]

        torch.testing.assert_close(cats.labels, torch.tensor([17, 17]))
        torch.testing.assert_close(cats.scores, torch.tensor([0.9, 0.8]))

    def test_getitem__returns_new_prediction(self) -> None:
        prediction = _prediction()

        kept = prediction[prediction.scores > 0.0]

        assert kept is not prediction
        kept.bboxes[0, 0] = 999.0
        assert prediction.bboxes[0, 0] == 0.0

    def test_getitem__preserves_mapping_semantics(self) -> None:
        prediction = _prediction()

        assert prediction["bboxes"] is prediction.bboxes
        assert len(prediction) == 3
        assert list(prediction) == ["labels", "bboxes", "scores"]
        assert dict(prediction)["scores"] is prediction.scores

    def test_num_detections__counts_detections_not_fields(self) -> None:
        prediction = _prediction()

        assert prediction.num_detections == 3
        assert len(prediction) == 3
        assert prediction[prediction.scores > 0.5].num_detections == 2

    def test_to_torchmetrics__after_filtering(self) -> None:
        prediction = _prediction()[_prediction().scores > 0.5]

        converted = prediction.to_torchmetrics()

        assert set(converted) == {"boxes", "scores", "labels"}
        assert converted["boxes"] is prediction.bboxes
        torch.testing.assert_close(converted["labels"], torch.tensor([17, 17]))


def _postprocessor() -> ObjectDetectionPostprocessor:
    return ObjectDetectionPostprocessor(
        num_classes=2,
        num_top_queries=3,
        internal_class_to_class=torch.tensor([10, 20]),
        image_size=(20, 30),
    )


class TestObjectDetectionPostprocessor:
    def test_postprocess__selects_rescales_and_remaps(self) -> None:
        logits = torch.tensor([[[8.0, -8.0], [1.0, 7.0], [6.0, 0.0]]])
        boxes = torch.tensor(
            [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.2, 0.2], [0.8, 0.5, 0.1, 0.2]]]
        )
        output = _postprocessor().postprocess(
            ObjectDetectionOutput(logits=logits, boxes=boxes),
            [ObjectDetectionMetadata(orig_w=100, orig_h=200)],
            threshold=0.0,
        )[0]
        torch.testing.assert_close(output.labels, torch.tensor([10, 20, 10]))
        torch.testing.assert_close(
            output.bboxes[0], torch.tensor([40.0, 60.0, 60.0, 140.0])
        )
        assert output.scores.shape == (3,)

    def test_postprocess__filters_by_threshold(self) -> None:
        logits = torch.full((1, 3, 2), -10.0)
        boxes = torch.rand(1, 3, 4)
        output = _postprocessor().postprocess(
            ObjectDetectionOutput(logits=logits, boxes=boxes),
            [ObjectDetectionMetadata(orig_w=30, orig_h=20)],
            threshold=0.5,
        )
        assert output[0].labels.shape == (0,)
        assert output[0].bboxes.shape == (0, 4)

    def test_postprocess_sahi__offsets_tiles(self) -> None:
        postprocessor = ObjectDetectionPostprocessor(
            num_classes=1,
            num_top_queries=1,
            internal_class_to_class=torch.tensor([7]),
            image_size=(10, 20),
        )
        output = postprocessor.postprocess(
            ObjectDetectionOutput(
                logits=torch.tensor([[[10.0]], [[9.0]], [[-10.0]]]),
                boxes=torch.tensor(
                    [
                        [[0.5, 0.5, 0.2, 0.2]],
                        [[0.5, 0.5, 0.2, 0.2]],
                        [[0.5, 0.5, 0.2, 0.2]],
                    ]
                ),
            ),
            [
                ObjectDetectionMetadata(
                    orig_w=100,
                    orig_h=50,
                    tile_coordinates=torch.tensor([[5, 7], [30, 20]]),
                )
            ],
            threshold=0.5,
            sahi_config=ObjectDetectionSahiConfig(
                overlap=0.2,
                nms_iou_threshold=0.3,
                global_local_iou_threshold=0.1,
            ),
        )[0]
        torch.testing.assert_close(output.labels, torch.tensor([7, 7]))
        torch.testing.assert_close(
            output.bboxes,
            torch.tensor([[40.0, 20.0, 60.0, 30.0], [13.0, 11.0, 17.0, 13.0]]),
        )

    def test_postprocess__rejects_sahi_metadata_for_multiple_images(self) -> None:
        raw = ObjectDetectionOutput(
            logits=torch.zeros(2, 3, 2), boxes=torch.zeros(2, 3, 4)
        )
        with pytest.raises(ValueError, match="metadata for one image"):
            _postprocessor().postprocess(
                raw,
                [
                    ObjectDetectionMetadata(orig_w=10, orig_h=10),
                    ObjectDetectionMetadata(orig_w=10, orig_h=10),
                ],
                threshold=0.5,
                sahi_config=ObjectDetectionSahiConfig(
                    overlap=0.2,
                    nms_iou_threshold=0.3,
                    global_local_iou_threshold=0.1,
                ),
            )

    def test_postprocess__prediction_supports_mapping_protocol(self) -> None:
        postprocessor = ObjectDetectionPostprocessor(
            num_classes=2,
            num_top_queries=1,
            internal_class_to_class=torch.tensor([10, 20]),
            image_size=(20, 30),
        )
        logits = torch.tensor([[[8.0, -8.0]]])
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4]]])
        prediction = postprocessor.postprocess(
            ObjectDetectionOutput(logits=logits, boxes=boxes),
            [ObjectDetectionMetadata(orig_w=100, orig_h=200)],
            threshold=0.5,
        )[0]

        assert set(prediction.keys()) == {"labels", "bboxes", "scores"}
        assert "bboxes" in prediction
        assert prediction["bboxes"] is prediction.bboxes
        as_dict = dict(prediction)
        assert as_dict["labels"] is prediction.labels
        assert as_dict["scores"] is prediction.scores
