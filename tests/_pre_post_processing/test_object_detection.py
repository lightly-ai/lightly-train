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
    filter_predictions_by_score,
    rescale_predictions_to_original_size,
    targets_to_torchmetrics,
    combine_object_detection_tiles,
    yolo_to_xyxy,
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


def test_combine_object_detection_tiles() -> None:
    labels_global = torch.tensor([1])
    boxes_global = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores_global = torch.tensor([0.8])
    labels_tiles = torch.tensor([2, 3])
    boxes_tiles = torch.tensor(
        [
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],
        ]
    )
    scores_tiles = torch.tensor([0.7, 0.9])

    labels_out, boxes_out, scores_out = combine_object_detection_tiles(
        pred_global={
            "labels": labels_global,
            "bboxes": boxes_global,
            "scores": scores_global,
        },
        pred_tiles={
            "labels": labels_tiles,
            "bboxes": boxes_tiles,
            "scores": scores_tiles,
        },
        nms_iou_threshold=0.5,
        global_local_iou_threshold=0.1,
    )

    torch.testing.assert_close(labels_out, torch.tensor([1, 3, 2]))
    torch.testing.assert_close(
        boxes_out,
        torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [40.0, 40.0, 50.0, 50.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ),
    )
    torch.testing.assert_close(scores_out, torch.tensor([0.8, 0.9, 0.7]))


def test_combine_object_detection_tiles__suppresses_tile_nms() -> None:
    labels_tiles = torch.tensor([1, 1, 2])
    boxes_tiles = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    scores_tiles = torch.tensor([0.8, 0.9, 0.7])

    labels_out, boxes_out, scores_out = combine_object_detection_tiles(
        pred_global={
            "labels": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty(0, 4),
            "scores": torch.empty(0),
        },
        pred_tiles={
            "labels": labels_tiles,
            "bboxes": boxes_tiles,
            "scores": scores_tiles,
        },
        nms_iou_threshold=0.5,
        global_local_iou_threshold=0.1,
    )

    torch.testing.assert_close(labels_out, torch.tensor([1, 2]))
    torch.testing.assert_close(boxes_out, boxes_tiles[[1, 2]])
    torch.testing.assert_close(scores_out, torch.tensor([0.9, 0.7]))


def test_combine_object_detection_tiles__keeps_overlapping_different_labels() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
    labels_out, boxes_out, scores_out = combine_object_detection_tiles(
        pred_global={
            "labels": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty(0, 4),
            "scores": torch.empty(0),
        },
        pred_tiles={
            "labels": torch.tensor([1, 2]),
            "bboxes": boxes,
            "scores": torch.tensor([0.9, 0.8]),
        },
        nms_iou_threshold=0.5,
        global_local_iou_threshold=0.1,
    )

    torch.testing.assert_close(labels_out, torch.tensor([1, 2]))
    torch.testing.assert_close(boxes_out, boxes)
    torch.testing.assert_close(scores_out, torch.tensor([0.9, 0.8]))


def test_combine_object_detection_tiles__suppresses_same_label_global_overlap() -> None:
    labels_global = torch.tensor([1])
    boxes_global = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores_global = torch.tensor([0.8])
    labels_tiles = torch.tensor([1, 2, 1])
    boxes_tiles = torch.tensor(
        [
            [1.0, 1.0, 9.0, 9.0],
            [1.0, 1.0, 9.0, 9.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )
    scores_tiles = torch.tensor([0.9, 0.7, 0.6])

    labels_out, boxes_out, scores_out = combine_object_detection_tiles(
        pred_global={
            "labels": labels_global,
            "bboxes": boxes_global,
            "scores": scores_global,
        },
        pred_tiles={
            "labels": labels_tiles,
            "bboxes": boxes_tiles,
            "scores": scores_tiles,
        },
        nms_iou_threshold=1.0,
        global_local_iou_threshold=0.5,
    )

    torch.testing.assert_close(labels_out, torch.tensor([1, 2, 1]))
    torch.testing.assert_close(
        boxes_out,
        torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 9.0, 9.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        ),
    )
    torch.testing.assert_close(scores_out, torch.tensor([0.8, 0.7, 0.6]))


def test_combine_object_detection_tiles__suppresses_lower_iou_same_label_global() -> (
    None
):
    # The tile box overlaps a different-label global box most strongly, but also
    # overlaps a same-label global box above the threshold. It must be suppressed
    # based on the same-label overlap, not the single strongest (different-label)
    # match.
    labels_global = torch.tensor([2, 1])
    boxes_global = torch.tensor(
        [
            [1.0, 1.0, 9.0, 9.0],  # different label, IoU == 1.0 with the tile box
            [0.0, 0.0, 10.0, 10.0],  # same label, IoU == 0.64 with the tile box
        ]
    )
    scores_global = torch.tensor([0.9, 0.8])
    labels_tiles = torch.tensor([1])
    boxes_tiles = torch.tensor([[1.0, 1.0, 9.0, 9.0]])
    scores_tiles = torch.tensor([0.7])

    labels_out, boxes_out, scores_out = combine_object_detection_tiles(
        pred_global={
            "labels": labels_global,
            "bboxes": boxes_global,
            "scores": scores_global,
        },
        pred_tiles={
            "labels": labels_tiles,
            "bboxes": boxes_tiles,
            "scores": scores_tiles,
        },
        nms_iou_threshold=0.5,
        global_local_iou_threshold=0.5,
    )

    torch.testing.assert_close(labels_out, labels_global)
    torch.testing.assert_close(boxes_out, boxes_global)
    torch.testing.assert_close(scores_out, scores_global)


def test_combine_object_detection_tiles__handles_empty_predictions() -> None:
    labels_out, boxes_out, scores_out = combine_object_detection_tiles(
        pred_global={
            "labels": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty(0, 4),
            "scores": torch.empty(0),
        },
        pred_tiles={
            "labels": torch.empty(0, dtype=torch.long),
            "bboxes": torch.empty(0, 4),
            "scores": torch.empty(0),
        },
        nms_iou_threshold=0.5,
        global_local_iou_threshold=0.1,
    )

    assert labels_out.shape == (0,)
    assert boxes_out.shape == (0, 4)
    assert scores_out.shape == (0,)
    assert labels_out.dtype == torch.long
    assert boxes_out.dtype == torch.float32
    assert scores_out.dtype == torch.float32


def test_yolo_to_xyxy_accepts_1d_box() -> None:
    boxes = [torch.tensor([0.5, 0.5, 0.2, 0.4], dtype=torch.float32)]
    converted = yolo_to_xyxy(boxes)

    assert len(converted) == 1
    assert converted[0].shape == (1, 4)
    expected = torch.tensor([[0.4, 0.3, 0.6, 0.7]], dtype=torch.float32)
    torch.testing.assert_close(converted[0], expected)


def test_yolo_to_xyxy_accepts_empty_boxes() -> None:
    boxes = [torch.zeros((0,), dtype=torch.float32)]
    converted = yolo_to_xyxy(boxes)

    assert len(converted) == 1
    assert converted[0].shape == (0, 4)


def test_yolo_to_xyxy_accepts_two_boxes() -> None:
    boxes = [
        torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.4],
                [0.25, 0.75, 0.1, 0.2],
            ],
            dtype=torch.float32,
        )
    ]
    converted = yolo_to_xyxy(boxes)

    assert len(converted) == 1
    assert converted[0].shape == (2, 4)
    expected = torch.tensor(
        [
            [0.4, 0.3, 0.6, 0.7],
            [0.2, 0.65, 0.3, 0.85],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(converted[0], expected)


def test_filter_predictions_by_score() -> None:
    predictions = [
        ObjectDetectionPrediction(
            labels=torch.tensor([1, 2, 3]),
            bboxes=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            scores=torch.tensor([0.1, 0.5, 0.9]),
        )
    ]

    filtered = filter_predictions_by_score(predictions, threshold=0.5)

    assert len(filtered) == 1
    # The threshold is exclusive: a score exactly at it is dropped.
    assert filtered[0].num_detections == 1
    torch.testing.assert_close(filtered[0].labels, torch.tensor([3]))


def test_rescale_predictions_to_original_size() -> None:
    predictions = [
        ObjectDetectionPrediction(
            labels=torch.tensor([1]),
            bboxes=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            scores=torch.tensor([0.9]),
        )
    ]
    metadata = [ObjectDetectionMetadata(orig_h=320, orig_w=1280)]

    rescaled = rescale_predictions_to_original_size(
        predictions=predictions, metadata=metadata, model_size=(640, 640)
    )

    # x scales by 1280/640 = 2, y scales by 320/640 = 0.5.
    torch.testing.assert_close(
        rescaled[0].bboxes, torch.tensor([[20.0, 10.0, 60.0, 20.0]])
    )
    # Labels and scores are carried through untouched.
    torch.testing.assert_close(rescaled[0].labels, predictions[0].labels)
    torch.testing.assert_close(rescaled[0].scores, predictions[0].scores)


def test_rescale_predictions_to_original_size__does_not_mutate_input() -> None:
    bboxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    predictions = [
        ObjectDetectionPrediction(
            labels=torch.tensor([1]), bboxes=bboxes, scores=torch.tensor([0.9])
        )
    ]

    rescale_predictions_to_original_size(
        predictions=predictions,
        metadata=[ObjectDetectionMetadata(orig_h=320, orig_w=1280)],
        model_size=(640, 640),
    )

    torch.testing.assert_close(bboxes, torch.tensor([[10.0, 20.0, 30.0, 40.0]]))


def test_targets_to_torchmetrics() -> None:
    # One box centered in the image, half as wide and a quarter as tall.
    bboxes = [torch.tensor([[0.5, 0.5, 0.5, 0.25]])]
    classes = [torch.tensor([7])]

    targets = targets_to_torchmetrics(
        bboxes=bboxes, classes=classes, original_sizes=[(200, 400)]
    )

    assert len(targets) == 1
    torch.testing.assert_close(
        targets[0]["boxes"], torch.tensor([[50.0, 150.0, 150.0, 250.0]])
    )
    torch.testing.assert_close(targets[0]["labels"], torch.tensor([7]))


def test_targets_to_torchmetrics__matches_prediction_coordinates() -> None:
    # A ground truth box and a prediction covering the same region must land on the
    # same numbers, otherwise the metric compares boxes in different coordinates.
    original_size = (640, 480)
    targets = targets_to_torchmetrics(
        bboxes=[torch.tensor([[0.5, 0.5, 1.0, 1.0]])],
        classes=[torch.tensor([0])],
        original_sizes=[original_size],
    )
    full_image = ObjectDetectionPrediction(
        labels=torch.tensor([0]),
        bboxes=torch.tensor([[0.0, 0.0, 640.0, 480.0]]),
        scores=torch.tensor([1.0]),
    )

    torch.testing.assert_close(
        targets[0]["boxes"], full_image.to_torchmetrics()["boxes"]
    )


def test_targets_to_torchmetrics__handles_empty_boxes() -> None:
    targets = targets_to_torchmetrics(
        bboxes=[torch.zeros((0, 4))],
        classes=[torch.zeros((0,), dtype=torch.long)],
        original_sizes=[(640, 480)],
    )

    assert targets[0]["boxes"].shape == (0, 4)
    assert targets[0]["labels"].shape == (0,)
