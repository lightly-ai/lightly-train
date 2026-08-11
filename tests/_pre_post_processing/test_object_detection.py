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
    ObjectDetectionBatchOutput,
    ObjectDetectionBatchPrediction,
    ObjectDetectionMetadata,
    ObjectDetectionPostprocessor,
    ObjectDetectionPrediction,
    ObjectDetectionPreprocessor,
    ObjectDetectionSAHIConfig,
    ObjectDetectionTiling,
    targets_to_torchmetrics,
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

        assert output.shape == (1, 3, 32, 48)
        assert output.dtype == torch.float32
        assert output.min() >= 0 and output.max() <= 1
        assert metadata == ObjectDetectionMetadata(orig_h=60, orig_w=80)
        assert metadata.num_rows == output.shape[0]

    def test_preprocess_image__validates_channels(self) -> None:
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(16, 16), image_normalize=None, expected_input_channels=3
        )
        grayscale, _ = preprocessor.preprocess_image(
            torch.rand(1, 8, 8),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert grayscale.shape == (1, 3, 16, 16)
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
            sahi_config=ObjectDetectionSAHIConfig(
                overlap=0.5,
                nms_iou_threshold=0.3,
                global_local_iou_threshold=0.1,
            ),
        )
        assert batch.shape == (10, 3, 4, 6)
        assert metadata.orig_h == 8
        assert metadata.orig_w == 10
        assert metadata.tiling is not None
        assert metadata.tiling.coordinates.shape == (9, 2)
        assert metadata.tiling.num_tiles == 9
        assert metadata.num_rows == batch.shape[0]
        # The merge settings are recorded on the metadata, so the postprocessor does
        # not need the config again.
        assert metadata.tiling.tile_size == (4, 6)
        assert metadata.tiling.nms_iou_threshold == 0.3
        assert metadata.tiling.global_local_iou_threshold == 0.1
        # Row 0 is the global row and scales back to the original image, tile rows
        # stay in tile coordinates.
        assert metadata.row_sizes == [(10, 8)] + [(6, 4)] * 9

    def test_preprocess__stacks_images_and_returns_metadata_per_image(self) -> None:
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(4, 4),
            image_normalize={"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
            expected_input_channels=3,
        )
        batch, metadata = preprocessor.preprocess(
            [torch.zeros(3, 8, 6, dtype=torch.uint8), torch.zeros(3, 5, 5)],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert batch.shape == (2, 3, 4, 4)
        # preprocess_batch normalized the (already scaled to [0, 1]) images.
        torch.testing.assert_close(batch, torch.full_like(batch, -1))
        assert metadata == [
            ObjectDetectionMetadata(orig_h=8, orig_w=6),
            ObjectDetectionMetadata(orig_h=5, orig_w=5),
        ]

    def test_preprocess__concatenates_tiles_of_all_images(self) -> None:
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(4, 6), image_normalize=None, expected_input_channels=3
        )
        batch, metadata = preprocessor.preprocess(
            [torch.zeros(3, 8, 10), torch.zeros(3, 4, 6)],
            device=torch.device("cpu"),
            dtype=torch.float32,
            sahi_config=ObjectDetectionSAHIConfig(
                overlap=0.5, nms_iou_threshold=0.3, global_local_iou_threshold=0.1
            ),
        )

        # One metadata per image, but one batch row per global image and per tile.
        assert len(metadata) == 2
        assert batch.shape[0] == sum(item.num_rows for item in metadata)

    def test_preprocess__rejects_empty_input(self) -> None:
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(4, 4), image_normalize=None, expected_input_channels=3
        )
        with pytest.raises(ValueError, match="at least one image"):
            preprocessor.preprocess([], device=torch.device("cpu"), dtype=torch.float32)


def _prediction() -> ObjectDetectionPrediction:
    return ObjectDetectionPrediction(
        labels=torch.tensor([17, 3, 17]),
        bboxes=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        scores=torch.tensor([0.9, 0.4, 0.8]),
    )


def _pred(
    labels: list[int], bboxes: list[list[float]], scores: list[float]
) -> ObjectDetectionPrediction:
    return ObjectDetectionPrediction(
        labels=torch.tensor(labels, dtype=torch.int64),
        bboxes=torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4),
        scores=torch.tensor(scores, dtype=torch.float32),
    )


def _batch_prediction(
    *, num_rows: int, num_queries: int = 2
) -> ObjectDetectionBatchPrediction:
    """A dense batch prediction whose values identify their row and query."""
    return ObjectDetectionBatchPrediction(
        labels=torch.arange(num_rows * num_queries).reshape(num_rows, num_queries),
        bboxes=torch.arange(num_rows * num_queries * 4, dtype=torch.float32).reshape(
            num_rows, num_queries, 4
        ),
        scores=torch.linspace(0.1, 0.9, num_rows * num_queries).reshape(
            num_rows, num_queries
        ),
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

    def test_offset__shifts_every_corner(self) -> None:
        prediction = _pred([1], [[0, 0, 10, 10]], [0.9])

        shifted = prediction.offset_boxes(torch.tensor([5, 7]))

        torch.testing.assert_close(
            shifted.bboxes, torch.tensor([[5.0, 7.0, 15.0, 17.0]])
        )
        # A copy, not a view: the original is untouched.
        torch.testing.assert_close(
            prediction.bboxes, torch.tensor([[0.0, 0.0, 10, 10]])
        )

    def test_offset__accepts_one_offset_per_detection(self) -> None:
        prediction = _pred([1, 2], [[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8])

        shifted = prediction.offset_boxes(torch.tensor([[5, 7], [1, 2]]))

        torch.testing.assert_close(
            shifted.bboxes,
            torch.tensor([[5.0, 7.0, 15.0, 17.0], [1.0, 2.0, 11.0, 12.0]]),
        )

    def test_nms__suppresses_overlapping_same_label(self) -> None:
        prediction = _pred(
            [1, 1, 2],
            [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]],
            [0.8, 0.9, 0.7],
        )

        kept = prediction.apply_nms(0.5)

        # Highest score first, as batched_nms orders its output by score.
        torch.testing.assert_close(kept.labels, torch.tensor([1, 2]))
        torch.testing.assert_close(kept.bboxes, prediction.bboxes[[1, 2]])
        torch.testing.assert_close(kept.scores, torch.tensor([0.9, 0.7]))

    def test_nms__keeps_overlapping_different_labels(self) -> None:
        # Suppression is class-aware so a high-confidence detection cannot hide a
        # detection of another class.
        prediction = _pred([1, 2], [[0, 0, 10, 10], [1, 1, 11, 11]], [0.9, 0.8])

        kept = prediction.apply_nms(0.5)

        torch.testing.assert_close(kept.labels, torch.tensor([1, 2]))
        torch.testing.assert_close(kept.bboxes, prediction.bboxes)
        torch.testing.assert_close(kept.scores, torch.tensor([0.9, 0.8]))

    def test_nms__handles_empty(self) -> None:
        kept = _pred([], [], []).apply_nms(0.5)

        assert kept.num_detections == 0
        assert kept.bboxes.shape == (0, 4)
        assert kept.labels.dtype == torch.int64

    def test_drop_overlapping__drops_same_label_matches_only(self) -> None:
        other = _pred([1], [[0, 0, 10, 10]], [0.8])
        prediction = _pred(
            [1, 2, 1],
            [[1, 1, 9, 9], [1, 1, 9, 9], [20, 20, 30, 30]],
            [0.9, 0.7, 0.6],
        )

        kept = prediction.drop_overlapping_predictions(other, 0.5)

        # Index 0 overlaps a same-label box and goes; index 1 has a different label
        # and index 2 does not overlap, so both stay.
        torch.testing.assert_close(kept.labels, torch.tensor([2, 1]))
        torch.testing.assert_close(kept.bboxes, prediction.bboxes[[1, 2]])
        torch.testing.assert_close(kept.scores, torch.tensor([0.7, 0.6]))

    def test_drop_overlapping__compares_labels_before_reducing(self) -> None:
        # The detection overlaps a different-label box most strongly, but also
        # overlaps a same-label box above the threshold. It must be dropped based on
        # the same-label overlap, not on the single strongest (different-label) match.
        other = _pred(
            [2, 1],
            [
                [1, 1, 9, 9],  # different label, IoU == 1.0
                [0, 0, 10, 10],  # same label, IoU == 0.64
            ],
            [0.9, 0.8],
        )
        prediction = _pred([1], [[1, 1, 9, 9]], [0.7])

        kept = prediction.drop_overlapping_predictions(other, 0.5)

        assert kept.num_detections == 0

    def test_drop_overlapping__handles_empty(self) -> None:
        empty = _pred([], [], [])
        full = _pred([1], [[0, 0, 10, 10]], [0.9])

        assert empty.drop_overlapping_predictions(full, 0.1).num_detections == 0
        assert full.drop_overlapping_predictions(empty, 0.1).num_detections == 1

    def test_map_labels__looks_labels_up(self) -> None:
        prediction = _pred([0, 2, 0], [[0, 0, 1, 1]] * 3, [0.9, 0.8, 0.7])

        remapped = prediction.map_labels(torch.tensor([10, 20, 30]))

        torch.testing.assert_close(remapped.labels, torch.tensor([10, 30, 10]))
        assert remapped.bboxes is prediction.bboxes

    def test_concat__keeps_order_and_dtypes(self) -> None:
        first = _pred([1], [[0, 0, 10, 10]], [0.8])
        second = _pred([2, 3], [[20, 20, 30, 30], [40, 40, 50, 50]], [0.7, 0.9])

        combined = ObjectDetectionPrediction.concat([first, second])

        torch.testing.assert_close(combined.labels, torch.tensor([1, 2, 3]))
        torch.testing.assert_close(combined.scores, torch.tensor([0.8, 0.7, 0.9]))
        assert combined.bboxes.shape == (3, 4)

    def test_concat__handles_empty(self) -> None:
        empty = _pred([], [], [])

        combined = ObjectDetectionPrediction.concat([empty, empty])

        assert combined.labels.shape == (0,)
        assert combined.bboxes.shape == (0, 4)
        assert combined.scores.shape == (0,)
        assert combined.labels.dtype == torch.int64
        assert combined.bboxes.dtype == torch.float32
        assert combined.scores.dtype == torch.float32


def _postprocessor() -> ObjectDetectionPostprocessor:
    return ObjectDetectionPostprocessor(
        num_top_queries=3, internal_class_to_class=torch.tensor([10, 20])
    )


def _tiling(
    coordinates: torch.Tensor,
    *,
    tile_size: tuple[int, int] = (20, 30),
    nms_iou_threshold: float = 0.3,
    global_local_iou_threshold: float = 0.1,
) -> ObjectDetectionTiling:
    return ObjectDetectionTiling(
        coordinates=coordinates,
        tile_size=tile_size,
        nms_iou_threshold=nms_iou_threshold,
        global_local_iou_threshold=global_local_iou_threshold,
    )


class TestObjectDetectionPostprocessor:
    def test_postprocess__selects_rescales_and_remaps(self) -> None:
        logits = torch.tensor([[[8.0, -8.0], [1.0, 7.0], [6.0, 0.0]]])
        boxes = torch.tensor(
            [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.2, 0.2], [0.8, 0.5, 0.1, 0.2]]]
        )
        output = _postprocessor().postprocess(
            ObjectDetectionBatchOutput(logits=logits, boxes=boxes),
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
            ObjectDetectionBatchOutput(logits=logits, boxes=boxes),
            [ObjectDetectionMetadata(orig_w=30, orig_h=20)],
            threshold=0.5,
        )
        assert output[0].labels.shape == (0,)
        assert output[0].bboxes.shape == (0, 4)

    def test_postprocess_sahi__offsets_tiles(self) -> None:
        postprocessor = ObjectDetectionPostprocessor(
            num_top_queries=1, internal_class_to_class=torch.tensor([7])
        )
        output = postprocessor.postprocess(
            ObjectDetectionBatchOutput(
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
                    tiling=_tiling(
                        torch.tensor([[5, 7], [30, 20]]), tile_size=(10, 20)
                    ),
                )
            ],
            threshold=0.5,
        )[0]
        torch.testing.assert_close(output.labels, torch.tensor([7, 7]))
        torch.testing.assert_close(
            output.bboxes,
            torch.tensor([[40.0, 20.0, 60.0, 30.0], [13.0, 11.0, 17.0, 13.0]]),
        )

    def test_postprocess_sahi__handles_multiple_images(self) -> None:
        """SAHI predictions for a batch match postprocessing each image on its own."""
        postprocessor = _postprocessor()
        torch.manual_seed(0)
        # Two images with a different number of tiles: 2 and 3.
        raw = ObjectDetectionBatchOutput(
            logits=torch.randn(7, 4, 2), boxes=torch.rand(7, 4, 4)
        )
        metadata = [
            ObjectDetectionMetadata(
                orig_w=100, orig_h=50, tiling=_tiling(torch.tensor([[0, 0], [30, 20]]))
            ),
            ObjectDetectionMetadata(
                orig_w=64,
                orig_h=64,
                tiling=_tiling(torch.tensor([[0, 0], [10, 0], [0, 10]])),
            ),
        ]

        predictions = postprocessor.postprocess(raw, metadata, threshold=0.3)

        assert len(predictions) == 2
        start = 0
        for item, prediction in zip(metadata, predictions):
            end = start + item.num_rows
            expected = postprocessor.postprocess(
                ObjectDetectionBatchOutput(
                    logits=raw.logits[start:end], boxes=raw.boxes[start:end]
                ),
                [item],
                threshold=0.3,
            )[0]
            torch.testing.assert_close(prediction.labels, expected.labels)
            torch.testing.assert_close(prediction.bboxes, expected.bboxes)
            torch.testing.assert_close(prediction.scores, expected.scores)
            start = end

    def test_postprocess_batch__keeps_every_row_dense(self) -> None:
        # Scores are far below any sane threshold, yet the dense stage must keep
        # num_top_queries detections for every row.
        logits = torch.full((2, 3, 2), -9.0)
        boxes = torch.rand(1, 3, 4).expand(2, -1, -1)
        metadata = [
            ObjectDetectionMetadata(orig_w=100, orig_h=200),
            ObjectDetectionMetadata(orig_w=10, orig_h=20),
        ]

        batch_prediction = _postprocessor().postprocess_batch(
            ObjectDetectionBatchOutput(logits=logits, boxes=boxes), metadata
        )

        assert batch_prediction.scores.max() < 0.001
        assert batch_prediction.labels.shape == (2, 3)
        assert batch_prediction.bboxes.shape == (2, 3, 4)
        assert batch_prediction.scores.shape == (2, 3)
        # Both rows hold the same normalized boxes but are rescaled to their own
        # original image size, which here differ by a factor of ten.
        torch.testing.assert_close(
            batch_prediction.bboxes[0], batch_prediction.bboxes[1] * 10
        )

    def test_postprocess_batch__labels_stay_in_internal_id_space(self) -> None:
        # The dense stage must not remap to user-facing class ids: that happens once,
        # in postprocess_image, on the final filtered/merged prediction.
        logits = torch.tensor(
            [[[8.0, -8.0], [1.0, 7.0], [6.0, 0.0]]]
        )  # -> internal labels [0, 1, 0]
        boxes = torch.tensor(
            [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.2, 0.2], [0.8, 0.5, 0.1, 0.2]]]
        )
        batch_prediction = _postprocessor().postprocess_batch(
            ObjectDetectionBatchOutput(logits=logits, boxes=boxes),
            [ObjectDetectionMetadata(orig_w=100, orig_h=200)],
        )

        torch.testing.assert_close(batch_prediction.labels, torch.tensor([[0, 1, 0]]))

    def test_postprocess_batch__then_postprocess_image_matches_postprocess(
        self,
    ) -> None:
        postprocessor = _postprocessor()
        torch.manual_seed(0)
        raw = ObjectDetectionBatchOutput(
            logits=torch.randn(2, 4, 2), boxes=torch.rand(2, 4, 4)
        )
        metadata = [
            ObjectDetectionMetadata(orig_w=100, orig_h=50),
            ObjectDetectionMetadata(orig_w=17, orig_h=33),
        ]

        batch_prediction = postprocessor.postprocess_batch(raw, metadata)
        num_rows = [item.num_rows for item in metadata]
        stagewise = [
            postprocessor.postprocess_image(item_prediction, item, threshold=0.4)
            for item_prediction, item in zip(batch_prediction.split(num_rows), metadata)
        ]
        combined = postprocessor.postprocess(raw, metadata, threshold=0.4)

        for expected, actual in zip(combined, stagewise):
            torch.testing.assert_close(actual.labels, expected.labels)
            torch.testing.assert_close(actual.bboxes, expected.bboxes)
            torch.testing.assert_close(actual.scores, expected.scores)

    def test_postprocess__tiled_metadata_needs_no_extra_arguments(self) -> None:
        # The metadata alone carries the merge settings, so the same call that handles
        # untiled images handles tiled ones. This is what lets TaskModel.postprocess,
        # which has no SAHI parameter, work with tiled metadata.
        postprocessor = _postprocessor()
        raw = ObjectDetectionBatchOutput(
            logits=torch.zeros(2, 3, 2), boxes=torch.rand(2, 3, 4)
        )
        metadata = ObjectDetectionMetadata(
            orig_w=10, orig_h=10, tiling=_tiling(torch.tensor([[0, 0]]))
        )

        prediction = postprocessor.postprocess(raw, [metadata], threshold=0.1)[0]

        assert prediction.num_detections > 0
        assert set(prediction.labels.tolist()) <= {10, 20}

    def test_postprocess__prediction_supports_mapping_protocol(self) -> None:
        postprocessor = ObjectDetectionPostprocessor(
            num_top_queries=1, internal_class_to_class=torch.tensor([10, 20])
        )
        logits = torch.tensor([[[8.0, -8.0]]])
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.4]]])
        prediction = postprocessor.postprocess(
            ObjectDetectionBatchOutput(logits=logits, boxes=boxes),
            [ObjectDetectionMetadata(orig_w=100, orig_h=200)],
            threshold=0.5,
        )[0]

        assert set(prediction.keys()) == {"labels", "bboxes", "scores"}
        assert "bboxes" in prediction
        assert prediction["bboxes"] is prediction.bboxes
        as_dict = dict(prediction)
        assert as_dict["labels"] is prediction.labels
        assert as_dict["scores"] is prediction.scores


class TestObjectDetectionBatchPrediction:
    def test_split__groups_consecutive_rows(self) -> None:
        batch_prediction = _batch_prediction(num_rows=4)

        groups = batch_prediction.split([1, 3])

        assert [group.num_rows for group in groups] == [1, 3]
        torch.testing.assert_close(groups[1].labels, batch_prediction.labels[1:])

    def test_getitem__selects_model_input_rows(self) -> None:
        batch_prediction = _batch_prediction(num_rows=3)

        tiles = batch_prediction[1:]

        assert tiles.num_rows == 2
        assert tiles.bboxes.shape == (2, 2, 4)

    def test_row__returns_a_flat_prediction(self) -> None:
        batch_prediction = _batch_prediction(num_rows=3)

        row = batch_prediction.row(0)

        assert isinstance(row, ObjectDetectionPrediction)
        assert row.num_detections == 2
        torch.testing.assert_close(row.bboxes, batch_prediction.bboxes[0])

    def test_offset_rows__shifts_each_row_by_its_own_offset(self) -> None:
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.zeros(2, 1, dtype=torch.int64),
            bboxes=torch.zeros(2, 1, 4),
            scores=torch.zeros(2, 1),
        )

        shifted = batch_prediction.offset_rows(torch.tensor([[5, 7], [1, 2]]))

        torch.testing.assert_close(
            shifted.bboxes,
            torch.tensor([[[5.0, 7.0, 5.0, 7.0]], [[1.0, 2.0, 1.0, 2.0]]]),
        )

    def test_flatten__collapses_the_row_dimension(self) -> None:
        batch_prediction = _batch_prediction(num_rows=3)

        flat = batch_prediction.flatten()

        assert flat.num_detections == 6
        assert flat.bboxes.shape == (6, 4)

    def test_merge_tiles__offsets_filters_and_merges(self) -> None:
        # Row 0 is the global view, rows 1: are tiles at (20, 20) and (40, 40). Each
        # row holds one detection above and one below the threshold.
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[1, 0], [2, 1], [3, 1]]),
            bboxes=torch.tensor(
                [
                    [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 1.0, 1.0]],
                    [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 1.0, 1.0]],
                    [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 1.0, 1.0]],
                ]
            ),
            scores=torch.tensor([[0.8, 0.05], [0.7, 0.02], [0.9, 0.01]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[20, 20], [40, 40]])), threshold=0.1
        )

        # Global first, then the surviving tile boxes in NMS (score) order, all in
        # original-image coordinates.
        torch.testing.assert_close(merged.labels, torch.tensor([1, 3, 2]))
        torch.testing.assert_close(
            merged.bboxes,
            torch.tensor(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [40.0, 40.0, 50.0, 50.0],
                    [20.0, 20.0, 30.0, 30.0],
                ]
            ),
        )
        torch.testing.assert_close(merged.scores, torch.tensor([0.8, 0.9, 0.7]))

    def test_merge_tiles__never_suppresses_global_boxes(self) -> None:
        # The tile box duplicates the global box exactly. The tile copy is dropped and
        # the global one is kept, even though it has the lower score.
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[1], [1]]),
            bboxes=torch.tensor([[[0.0, 0.0, 10.0, 10.0]], [[0.0, 0.0, 10.0, 10.0]]]),
            scores=torch.tensor([[0.5], [0.9]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[0, 0]])), threshold=0.1
        )

        torch.testing.assert_close(merged.labels, torch.tensor([1]))
        torch.testing.assert_close(merged.scores, torch.tensor([0.5]))

    def test_merge_tiles__handles_everything_filtered_out(self) -> None:
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[1], [1]]),
            bboxes=torch.zeros(2, 1, 4),
            scores=torch.tensor([[0.01], [0.02]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[0, 0]])), threshold=0.1
        )

        assert merged.num_detections == 0
        assert merged.bboxes.shape == (0, 4)


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
