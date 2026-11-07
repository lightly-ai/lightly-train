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
    combine_object_detection_tiles,
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
        assert metadata == {"orig_h": 60, "orig_w": 80}

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
            image_size=(4, 6), image_normalize=None, expected_input_channels=3
        )
        batch, metadata = preprocessor.preprocess_sahi_image(
            torch.zeros(3, 8, 10, dtype=torch.uint8),
            device=torch.device("cpu"),
            dtype=torch.float32,
            overlap=0.5,
        )
        assert batch.shape == (10, 3, 4, 6)
        assert metadata["orig_h"] == 8
        assert metadata["orig_w"] == 10
        assert metadata["tiles_coordinates"].shape == (9, 2)


def _postprocessor() -> ObjectDetectionPostprocessor:
    return ObjectDetectionPostprocessor(
        num_classes=2,
        num_top_queries=3,
        internal_class_to_class=torch.tensor([10, 20]),
    )


class TestObjectDetectionPostprocessor:
    def test_decode__selects_rescales_and_remaps(self) -> None:
        logits = torch.tensor([[[8.0, -8.0], [1.0, 7.0], [6.0, 0.0]]])
        boxes = torch.tensor(
            [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.2, 0.2], [0.8, 0.5, 0.1, 0.2]]]
        )
        labels, decoded_boxes, scores = _postprocessor().decode(
            ObjectDetectionOutput(logits=logits, boxes=boxes),
            torch.tensor([[100, 200]]),
        )
        torch.testing.assert_close(labels, torch.tensor([[10, 20, 10]]))
        torch.testing.assert_close(
            decoded_boxes[0, 0], torch.tensor([40.0, 60.0, 60.0, 140.0])
        )
        assert scores.shape == (1, 3)

    def test_decode_internal__keeps_internal_class_ids(self) -> None:
        logits = torch.tensor([[[8.0, -8.0], [1.0, 7.0], [6.0, 0.0]]])
        boxes = torch.tensor(
            [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.2, 0.2], [0.8, 0.5, 0.1, 0.2]]]
        )
        labels, _, _ = _postprocessor().decode_internal(
            ObjectDetectionOutput(logits=logits, boxes=boxes),
            torch.tensor([[100, 200]]),
        )
        torch.testing.assert_close(labels, torch.tensor([[0, 1, 0]]))

    def test_postprocess__filters_by_threshold(self) -> None:
        logits = torch.full((1, 3, 2), -10.0)
        boxes = torch.rand(1, 3, 4)
        output = _postprocessor().postprocess(
            ObjectDetectionOutput(logits=logits, boxes=boxes),
            [{"orig_h": 20, "orig_w": 30}],
            threshold=0.5,
        )
        assert output[0]["labels"].shape == (0,)
        assert output[0]["bboxes"].shape == (0, 4)

    def test_postprocess_sahi__offsets_tiles(self) -> None:
        postprocessor = ObjectDetectionPostprocessor(
            num_classes=1,
            num_top_queries=1,
            internal_class_to_class=torch.tensor([7]),
        )
        output = postprocessor.postprocess_sahi(
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
        torch.testing.assert_close(output["labels"], torch.tensor([7, 7]))
        torch.testing.assert_close(
            output["bboxes"],
            torch.tensor([[40.0, 20.0, 60.0, 30.0], [13.0, 11.0, 17.0, 13.0]]),
        )


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


def test_combine_object_detection_tiles__keeps_tile_overlap_with_different_labels() -> (
    None
):
    labels_tiles = torch.tensor([1, 2])
    boxes_tiles = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
    scores_tiles = torch.tensor([0.8, 0.9])

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

    torch.testing.assert_close(labels_out, torch.tensor([2, 1]))
    torch.testing.assert_close(boxes_out, boxes_tiles[[1, 0]])
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
