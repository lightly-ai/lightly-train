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
                # Explicit, so the tile arithmetic stays small enough to state exactly.
                # The default is half the model input, which is covered separately.
                tile_size=(4, 6),
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
        # Row 0 covers the whole image, each tile row covers tile_size pixels of it.
        assert metadata.row_sizes == [(10, 8)] + [(6, 4)] * 9

    def test_preprocess_sahi_image__magnifies_tiles_smaller_than_the_model_input(
        self,
    ) -> None:
        # The point of SAHI: a tile covers a *smaller* region of the image than the model
        # input, so resizing it up magnifies the content and small objects become
        # detectable. Cutting tiles at the model input size instead makes them
        # native-resolution crops that magnify nothing.
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(8, 8), image_normalize=None, expected_input_channels=3
        )

        batch, metadata = preprocessor.preprocess_image(
            torch.zeros(3, 8, 8, dtype=torch.uint8),
            device=torch.device("cpu"),
            dtype=torch.float32,
            sahi_config=ObjectDetectionSAHIConfig(
                overlap=0.0,
                nms_iou_threshold=0.3,
                global_local_iou_threshold=0.1,
                tile_size=(4, 4),
            ),
        )

        assert metadata.tiling is not None
        # A 4x4 grid of 4x4 windows over the 8x8 image, hence 4 tiles plus the global row.
        assert metadata.tiling.num_tiles == 4
        assert batch.shape == (5, 3, 8, 8)
        # Every row is at the model input size, but a tile row *covers* only 4x4 pixels
        # of the original image, and that is what its boxes are decoded against.
        assert metadata.tiling.tile_size == (4, 4)
        assert metadata.row_sizes == [(8, 8)] + [(4, 4)] * 4

    def test_preprocess_sahi_image__tile_size_defaults_to_half_the_model_input(
        self,
    ) -> None:
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(8, 12), image_normalize=None, expected_input_channels=3
        )

        _, metadata = preprocessor.preprocess_image(
            torch.zeros(3, 16, 24, dtype=torch.uint8),
            device=torch.device("cpu"),
            dtype=torch.float32,
            sahi_config=ObjectDetectionSAHIConfig(
                overlap=0.0, nms_iou_threshold=0.3, global_local_iou_threshold=0.1
            ),
        )

        assert metadata.tiling is not None
        assert metadata.tiling.tile_size == (4, 6)

    def test_preprocess_sahi_image__keeps_coordinates_in_original_pixels(self) -> None:
        # The regression test for the coordinate frame bug: an image smaller than a tile
        # in one dimension used to be upscaled inside tile_image, which returned
        # coordinates in that upscaled frame with no record of the scale, so every tile
        # detection came back stretched and could land outside the image.
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(4, 6), image_normalize=None, expected_input_channels=3
        )

        batch, metadata = preprocessor.preprocess_image(
            torch.full((3, 2, 10), 255, dtype=torch.uint8),
            device=torch.device("cpu"),
            dtype=torch.float32,
            sahi_config=ObjectDetectionSAHIConfig(
                overlap=0.0,
                nms_iou_threshold=0.3,
                global_local_iou_threshold=0.1,
                tile_size=(4, 6),
            ),
        )

        assert metadata.tiling is not None
        assert int(metadata.tiling.coordinates[:, 0].max()) < metadata.orig_w
        assert int(metadata.tiling.coordinates[:, 1].max()) < metadata.orig_h
        # The tile holds the original pixels plus a zero band, not an upscaled image.
        torch.testing.assert_close(batch[1, :, :2, :], torch.ones(3, 2, 6))
        assert torch.all(batch[1, :, 2:, :] == 0)

    def test_preprocess_sahi_image__skips_tiling_when_image_fits_in_one_tile(
        self,
    ) -> None:
        # The only tile would be the zero-padded original at native resolution, which
        # magnifies nothing the global row does not already show. Note that tiling is
        # None even though a sahi_config was given.
        preprocessor = ObjectDetectionPreprocessor(
            image_size=(4, 6), image_normalize=None, expected_input_channels=3
        )

        for size in [(4, 6), (2, 3), (1, 1)]:
            batch, metadata = preprocessor.preprocess_image(
                torch.zeros(3, *size),
                device=torch.device("cpu"),
                dtype=torch.float32,
                sahi_config=ObjectDetectionSAHIConfig(
                    overlap=0.0,
                    nms_iou_threshold=0.3,
                    global_local_iou_threshold=0.1,
                    tile_size=(4, 6),
                ),
            )

            assert metadata.tiling is None, size
            assert metadata.num_rows == 1, size
            assert batch.shape == (1, 3, 4, 6), size
            assert metadata.row_sizes == [(size[1], size[0])], size

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
                overlap=0.5,
                nms_iou_threshold=0.3,
                global_local_iou_threshold=0.1,
                tile_size=(4, 6),
            ),
        )

        # One metadata per image, but one batch row per global image and per tile. The
        # second image is exactly one tile, so it is not tiled and contributes one row --
        # mixed row counts within a batch are exactly why metadata travels with it.
        assert len(metadata) == 2
        assert [item.num_rows for item in metadata] == [10, 1]
        assert metadata[0].tiling is not None
        assert metadata[1].tiling is None
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

    def test_drop_contained__drops_boxes_inside_a_region(self) -> None:
        prediction = _pred(
            [1, 2], [[2, 2, 8, 8], [2, 2, 8, 8]], [0.9, 0.8]
        ).drop_contained_predictions(torch.tensor([[0.0, 0.0, 10.0, 10.0]]))

        assert prediction.num_detections == 0

    def test_drop_contained__keeps_boxes_crossing_a_region_edge(self) -> None:
        # Only full containment counts: a box poking out of every region is kept, since
        # no single tile saw the whole object.
        prediction = _pred(
            [1, 2, 3],
            [[2, 2, 12, 8], [-1, 2, 8, 8], [2, 2, 8, 8]],
            [0.9, 0.8, 0.7],
        )

        kept = prediction.drop_contained_predictions(
            torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        )

        torch.testing.assert_close(kept.labels, torch.tensor([1, 2]))

    def test_drop_contained__ignores_labels_and_scores(self) -> None:
        # Unlike drop_overlapping_predictions this is purely geometric, so two boxes of
        # different classes inside the same region both go.
        prediction = _pred([1, 7], [[2, 2, 4, 4], [5, 5, 8, 8]], [0.9, 0.1])

        kept = prediction.drop_contained_predictions(
            torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        )

        assert kept.num_detections == 0

    def test_drop_contained__handles_infinite_regions(self) -> None:
        # Tile regions touching the image border extend to infinity, so a box leaving the
        # image only through that border still counts as contained.
        prediction = _pred([1, 2], [[-5, 2, 8, 8], [-5, 2, 20, 8]], [0.9, 0.8])

        kept = prediction.drop_contained_predictions(
            torch.tensor([[-torch.inf, 0.0, 10.0, 10.0]])
        )

        torch.testing.assert_close(kept.labels, torch.tensor([2]))

    def test_drop_contained__handles_empty(self) -> None:
        empty = _pred([], [], [])
        full = _pred([1], [[2, 2, 8, 8]], [0.9])
        region = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

        assert empty.drop_contained_predictions(region).num_detections == 0
        # No regions at all: nothing can be contained, so everything is kept.
        assert full.drop_contained_predictions(torch.zeros(0, 4)).num_detections == 1

    def test_clip_to_image__clips_every_corner(self) -> None:
        prediction = _pred([1], [[-5, -5, 15, 25]], [0.9])

        clipped = prediction.clip_to_image(height=20, width=10)

        torch.testing.assert_close(
            clipped.bboxes, torch.tensor([[0.0, 0.0, 10.0, 20.0]])
        )
        # A copy, not a view.
        torch.testing.assert_close(
            prediction.bboxes, torch.tensor([[-5.0, -5.0, 15.0, 25.0]])
        )

    def test_clip_to_image__drops_boxes_that_clip_to_empty(self) -> None:
        # Box 0 is entirely right of the image and box 1 entirely below it: both clip to
        # zero area, which is what a detection inside a tile's zero padding looks like.
        # A zero-area box has IoU 0 with everything, so it would survive NMS.
        prediction = _pred(
            [1, 2, 3],
            [[12, 0, 20, 5], [0, 25, 5, 30], [2, 2, 8, 8]],
            [0.9, 0.8, 0.7],
        )

        clipped = prediction.clip_to_image(height=20, width=10)

        torch.testing.assert_close(clipped.labels, torch.tensor([3]))
        torch.testing.assert_close(clipped.bboxes, torch.tensor([[2.0, 2.0, 8.0, 8.0]]))
        torch.testing.assert_close(clipped.scores, torch.tensor([0.7]))

    def test_clip_to_image__is_a_no_op_for_in_bounds_boxes(self) -> None:
        prediction = _pred([1, 2], [[0, 0, 10, 20], [3, 4, 5, 6]], [0.9, 0.8])

        clipped = prediction.clip_to_image(height=20, width=10)

        torch.testing.assert_close(clipped.bboxes, prediction.bboxes)
        torch.testing.assert_close(clipped.labels, prediction.labels)

    def test_clip_to_image__handles_empty(self) -> None:
        clipped = _pred([], [], []).clip_to_image(height=4, width=4)

        assert clipped.num_detections == 0
        assert clipped.bboxes.shape == (0, 4)
        assert clipped.labels.dtype == torch.int64

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


class TestObjectDetectionTiling:
    def test_tile_boxes__returns_one_region_per_tile(self) -> None:
        # tile_size is (height, width) while coordinates are (x, y), so the region for
        # coordinate (x, y) is [x, y, x + width, y + height]. The interior seam between
        # the two tiles is left alone; only the outer edges are pushed out.
        tiling = _tiling(torch.tensor([[20, 20], [40, 20]]), tile_size=(20, 30))

        torch.testing.assert_close(
            tiling.tile_boxes,
            torch.tensor(
                [
                    [-torch.inf, -torch.inf, 50.0, torch.inf],
                    [40.0, -torch.inf, torch.inf, torch.inf],
                ]
            ),
        )

    def test_tile_boxes__pushes_only_border_edges_outwards(self) -> None:
        # A 2x2 grid. Every tile has two interior edges, which must stay finite, and two
        # border edges, which go to infinity.
        tiling = _tiling(
            torch.tensor([[0, 0], [20, 0], [0, 10], [20, 10]]), tile_size=(20, 30)
        )

        torch.testing.assert_close(
            tiling.tile_boxes,
            torch.tensor(
                [
                    [-torch.inf, -torch.inf, 30.0, 20.0],
                    [20.0, -torch.inf, torch.inf, 20.0],
                    [-torch.inf, 10.0, 30.0, torch.inf],
                    [20.0, 10.0, torch.inf, torch.inf],
                ]
            ),
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
                        # The global box spans the whole image, so no tile contains it and
                        # it survives the size split.
                        [[0.5, 0.5, 1.0, 1.0]],
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
            torch.tensor([[0.0, 0.0, 100.0, 50.0], [13.0, 11.0, 17.0, 13.0]]),
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
        # row holds one detection above and one below the threshold. The global box spans
        # the whole image, so no tile could contain it and it survives the size split.
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[1, 0], [2, 1], [3, 1]]),
            bboxes=torch.tensor(
                [
                    [[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 1.0, 1.0]],
                    [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 1.0, 1.0]],
                    [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 1.0, 1.0]],
                ]
            ),
            scores=torch.tensor([[0.8, 0.05], [0.7, 0.02], [0.9, 0.01]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[20, 20], [40, 40]])),
            threshold=0.1,
            orig_h=100,
            orig_w=100,
        )

        # Descending score order, all in original-image coordinates.
        torch.testing.assert_close(merged.labels, torch.tensor([3, 1, 2]))
        torch.testing.assert_close(
            merged.bboxes,
            torch.tensor(
                [
                    [40.0, 40.0, 50.0, 50.0],
                    [0.0, 0.0, 100.0, 100.0],
                    [20.0, 20.0, 30.0, 30.0],
                ]
            ),
        )
        torch.testing.assert_close(merged.scores, torch.tensor([0.9, 0.8, 0.7]))

    def test_merge_tiles__drops_global_boxes_that_fit_inside_a_tile(self) -> None:
        # The regression test for the merge defect. The global view sees this object at a
        # reduced resolution and returns a coarse box with a *higher* score; the tile sees
        # it magnified and localizes it accurately. The tiles cover the object completely,
        # so the global box is dropped and can no longer evict the accurate tile box.
        # Tiles at (20, 20) and (40, 20) cover x in [20, 50] and [40, 70], and their
        # vertical edges are on the image border so they extend to +-inf in y.
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[1], [1], [1]]),
            bboxes=torch.tensor(
                [
                    [[28.0, 28.0, 44.0, 38.0]],  # global: coarse, fits inside tile 0
                    [[10.0, 10.0, 20.0, 18.0]],  # tile at (20, 20): accurate
                    [[0.0, 0.0, 1.0, 1.0]],  # tile at (40, 20): below threshold
                ]
            ),
            scores=torch.tensor([[0.9], [0.6], [0.02]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[20, 20], [40, 20]])),
            threshold=0.1,
            orig_h=100,
            orig_w=100,
        )

        torch.testing.assert_close(merged.labels, torch.tensor([1]))
        torch.testing.assert_close(
            merged.bboxes, torch.tensor([[30.0, 30.0, 40.0, 38.0]])
        )
        torch.testing.assert_close(merged.scores, torch.tensor([0.6]))

    def test_merge_tiles__keeps_global_boxes_no_tile_can_contain(self) -> None:
        # An object wider than a tile is only ever seen in parts by the tiles, so the
        # global box is the only correct detection: it survives, and the tile fragments of
        # it are dropped. This is what a symmetric merge would get wrong.
        # Tiles at (20, 20) and (40, 20) each span 30 pixels of an object 40 wide, so
        # neither contains it and each only ever sees a part.
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[1], [1], [1]]),
            bboxes=torch.tensor(
                [
                    [[25.0, 25.0, 65.0, 35.0]],  # global: spans both tiles
                    [[5.0, 5.0, 30.0, 15.0]],  # tile at (20, 20): left part
                    [[0.0, 5.0, 25.0, 15.0]],  # tile at (40, 20): right part
                ]
            ),
            scores=torch.tensor([[0.7], [0.9], [0.8]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[20, 20], [40, 20]])),
            threshold=0.1,
            orig_h=100,
            orig_w=100,
        )

        torch.testing.assert_close(merged.labels, torch.tensor([1]))
        torch.testing.assert_close(
            merged.bboxes, torch.tensor([[25.0, 25.0, 65.0, 35.0]])
        )
        torch.testing.assert_close(merged.scores, torch.tensor([0.7]))

    def test_merge_tiles__fragment_suppression_is_class_aware(self) -> None:
        # An over-sized global box must not delete a tile detection of another class
        # inside it -- a person standing in front of a bus is not a fragment of the bus.
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[1], [2], [1]]),
            bboxes=torch.tensor(
                [
                    [[20.0, 20.0, 70.0, 40.0]],  # global: spans both tiles, label 1
                    [[0.0, 0.0, 30.0, 20.0]],  # tile: same region, label 2
                    [[0.0, 0.0, 1.0, 1.0]],
                ]
            ),
            scores=torch.tensor([[0.7], [0.9], [0.02]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[20, 20], [40, 40]])),
            threshold=0.1,
            orig_h=100,
            orig_w=100,
        )

        torch.testing.assert_close(merged.labels, torch.tensor([2, 1]))
        torch.testing.assert_close(merged.scores, torch.tensor([0.9, 0.7]))

    def test_merge_tiles__deduplicates_boxes_from_overlapping_tiles(self) -> None:
        # Two overlapping tiles see the same object. Class-aware NMS keeps the
        # higher-scoring copy.
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[1], [1], [1]]),
            bboxes=torch.tensor(
                [
                    [[0.0, 0.0, 1.0, 1.0]],  # global: below threshold
                    [[10.0, 0.0, 20.0, 10.0]],  # tile at (20, 20) -> (30, 20, 40, 30)
                    [[0.0, 0.0, 10.0, 10.0]],  # tile at (30, 20) -> (30, 20, 40, 30)
                ]
            ),
            scores=torch.tensor([[0.02], [0.6], [0.9]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[20, 20], [30, 20]])),
            threshold=0.1,
            orig_h=100,
            orig_w=100,
        )

        torch.testing.assert_close(
            merged.bboxes, torch.tensor([[30.0, 20.0, 40.0, 30.0]])
        )
        torch.testing.assert_close(merged.scores, torch.tensor([0.9]))

    def test_merge_tiles__clips_boxes_to_the_original_image(self) -> None:
        # Tiles are zero padded where the image is smaller than a tile, so a detection can
        # reach into the padding (row 1) or lie entirely inside it (row 2). The first is
        # trimmed to the image, the second is dropped as empty.
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[0], [1], [2]]),
            bboxes=torch.tensor(
                [
                    [[0.0, 0.0, 40.0, 10.0]],  # global: spans the image, kept
                    [[1.0, 1.0, 15.0, 25.0]],  # tile at (0, 0): reaches past y=10
                    [[0.0, 15.0, 20.0, 25.0]],  # tile at (20, 0): entirely below y=10
                ]
            ),
            scores=torch.tensor([[0.9], [0.8], [0.7]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[0, 0], [20, 0]])),
            threshold=0.1,
            orig_h=10,
            orig_w=40,
        )

        torch.testing.assert_close(merged.labels, torch.tensor([0, 1]))
        torch.testing.assert_close(
            merged.bboxes,
            torch.tensor([[0.0, 0.0, 40.0, 10.0], [1.0, 1.0, 15.0, 10.0]]),
        )
        torch.testing.assert_close(merged.scores, torch.tensor([0.9, 0.8]))

    def test_merge_tiles__clips_before_nms(self) -> None:
        # The tile's second detection lies entirely in the bottom padding and outscores
        # the first, which is real. Un-clipped their IoU is above the NMS threshold, so
        # suppressing on raw boxes would keep the padding box and throw the real detection
        # away. Clipping first drops the padding box as empty instead.
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[1, 1], [1, 1], [1, 1]]),
            bboxes=torch.tensor(
                [
                    [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                    [[0.0, 0.0, 20.0, 20.0], [0.0, 12.0, 20.0, 20.0]],
                    [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                ]
            ),
            scores=torch.tensor([[0.02, 0.02], [0.8, 0.9], [0.02, 0.02]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[0, 0], [20, 0]])),
            threshold=0.1,
            orig_h=10,
            orig_w=40,
        )

        torch.testing.assert_close(merged.labels, torch.tensor([1]))
        torch.testing.assert_close(
            merged.bboxes, torch.tensor([[0.0, 0.0, 20.0, 10.0]])
        )
        torch.testing.assert_close(merged.scores, torch.tensor([0.8]))

    def test_merge_tiles__keeps_the_global_view_with_a_single_tile(self) -> None:
        # One tile spans the whole image and has no resolution advantage, so there is
        # nothing to arbitrate and the global view is not gated.
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[1], [2]]),
            bboxes=torch.tensor([[[0.0, 0.0, 10.0, 10.0]], [[0.0, 0.0, 10.0, 10.0]]]),
            scores=torch.tensor([[0.5], [0.9]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[0, 0]])), threshold=0.1, orig_h=20, orig_w=30
        )

        torch.testing.assert_close(merged.labels, torch.tensor([2, 1]))
        torch.testing.assert_close(merged.scores, torch.tensor([0.9, 0.5]))

    def test_merge_tiles__handles_everything_filtered_out(self) -> None:
        batch_prediction = ObjectDetectionBatchPrediction(
            labels=torch.tensor([[1], [1]]),
            bboxes=torch.zeros(2, 1, 4),
            scores=torch.tensor([[0.01], [0.02]]),
        )

        merged = batch_prediction.merge_tiles(
            _tiling(torch.tensor([[0, 0]])), threshold=0.1, orig_h=20, orig_w=30
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


def test_preprocess_and_postprocess_sahi__merged_boxes_stay_inside_the_image() -> None:
    # The end-to-end shape of the coordinate bug: an image smaller than a tile in one
    # dimension, so every tile carries a zero padded band. The raw box deliberately
    # overshoots the tile on the left and the right. Every merged box must still be
    # inside the 10x2 image, and must have positive area.
    preprocessor = ObjectDetectionPreprocessor(
        image_size=(4, 6), image_normalize=None, expected_input_channels=3
    )
    postprocessor = ObjectDetectionPostprocessor(
        num_top_queries=1, internal_class_to_class=torch.tensor([7])
    )
    batch, metadata = preprocessor.preprocess_image(
        torch.zeros(3, 2, 10),
        device=torch.device("cpu"),
        dtype=torch.float32,
        sahi_config=ObjectDetectionSAHIConfig(
            overlap=0.0,
            nms_iou_threshold=0.5,
            global_local_iou_threshold=0.1,
            tile_size=(4, 6),
        ),
    )
    # One global row plus two tiles, at x = 0 and x = 4.
    assert batch.shape[0] == 3
    assert metadata.tiling is not None

    # Row 0 (global) is below the threshold, so the tile boxes are not compared against
    # it. cxcywh (0.5, 0.3, 1.2, 0.6) is xyxy (-0.1, 0.0, 1.1, 0.6): wider than the tile
    # and starting left of it.
    raw = ObjectDetectionBatchOutput(
        logits=torch.tensor([[[-10.0]], [[10.0]], [[9.0]]]),
        boxes=torch.tensor([[[0.5, 0.3, 1.2, 0.6]]]).repeat(3, 1, 1),
    )

    prediction = postprocessor.postprocess(raw, [metadata], threshold=0.5)[0]

    torch.testing.assert_close(prediction.labels, torch.tensor([7, 7]))
    torch.testing.assert_close(
        prediction.bboxes,
        torch.tensor([[0.0, 0.0, 6.6, 2.0], [3.4, 0.0, 10.0, 2.0]]),
    )
    assert prediction.bboxes[:, :2].min() >= 0.0
    assert prediction.bboxes[:, 2].max() <= metadata.orig_w
    assert prediction.bboxes[:, 3].max() <= metadata.orig_h
    assert torch.all(prediction.bboxes[:, 2:] > prediction.bboxes[:, :2])
