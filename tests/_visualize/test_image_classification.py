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
from torch import Tensor

from lightly_train._visualize import image_classification
from lightly_train.types import ImageClassificationBatch

_WHITE_COLOR: float = 1.0
_WHITE_PIXEL: tuple[int, int, int] = (255, 255, 255)


def _make_batch(
    *,
    batch_size: int = 1,
    height: int = 32,
    width: int = 32,
    classes: list[Tensor] | None = None,
) -> ImageClassificationBatch:
    image = torch.rand(batch_size, 3, height, width)
    if classes is None:
        classes = [torch.zeros(0, dtype=torch.long) for _ in range(batch_size)]
    return ImageClassificationBatch(
        image_path=[f"img_{i}.jpg" for i in range(batch_size)],
        image=image,
        classes=classes,
    )


def _make_batch_from_image(
    *,
    image: Tensor,
    classes: list[Tensor] | None = None,
) -> ImageClassificationBatch:
    batch_size = image.shape[0]
    if classes is None:
        classes = [torch.zeros(0, dtype=torch.long) for _ in range(batch_size)]
    return ImageClassificationBatch(
        image_path=[f"img_{i}.jpg" for i in range(batch_size)],
        image=image,
        classes=classes,
    )


class TestPlotImageClassificationLabels:
    def test_plot_image_classification_labels_grid_caps_at_max_images(self) -> None:
        batch = _make_batch(batch_size=4, height=16, width=16)
        result = image_classification.plot_image_classification_labels(
            batch=batch, included_classes={}, max_images=2
        )
        assert result.size == (32, 16)

    def test_plot_image_classification_labels_label_drawn(self) -> None:
        # The corner label draws a semi-transparent black overlay at (0, 0);
        # on a white background the top-left pixel becomes noticeably darker.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 128, 128), _WHITE_COLOR),
            classes=[torch.tensor([1], dtype=torch.long)],
        )
        result = image_classification.plot_image_classification_labels(
            batch=batch, included_classes={1: "dog"}, max_images=1
        )
        assert result.getpixel((0, 0)) != _WHITE_PIXEL
        # Far corner is untouched by the label overlay.
        assert result.getpixel((127, 127)) == _WHITE_PIXEL

    def test_plot_image_classification_labels_unknown_class_draws_label(self) -> None:
        # A class ID absent from included_classes falls back to "Class {id}" text
        # but still draws the corner label overlay.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 128, 128), _WHITE_COLOR),
            classes=[torch.tensor([99], dtype=torch.long)],
        )
        result = image_classification.plot_image_classification_labels(
            batch=batch, included_classes={}, max_images=1
        )
        assert result.getpixel((0, 0)) != _WHITE_PIXEL
        assert result.getpixel((127, 127)) == _WHITE_PIXEL

    def test_plot_image_classification_labels_empty_classes_produces_clean_image(
        self,
    ) -> None:
        # No class labels → no overlay drawn; the image passes through unchanged.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 32, 32), _WHITE_COLOR),
            classes=[torch.zeros(0, dtype=torch.long)],
        )
        result = image_classification.plot_image_classification_labels(
            batch=batch, included_classes={}, max_images=1
        )
        assert result.getpixel((0, 0)) == _WHITE_PIXEL
        assert result.getpixel((31, 31)) == _WHITE_PIXEL

    @pytest.mark.parametrize(
        "image_value, mean, std, expected_pixel",
        [
            # image=0 -> denormalized = mean. Per-channel means verify channel order.
            # (0.2, 0.4, 0.6) * 255 = (51, 102, 153).
            (0.0, (0.2, 0.4, 0.6), (0.5, 0.5, 0.5), (51, 102, 153)),
            # Non-zero image: pixel = image * std + mean.
            # 0.4 * 0.5 + 0.5 = 0.7 -> 178 (PIL truncates 0.7 * 255 = 178.5).
            (0.4, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (178, 178, 178)),
            # Per-channel std with zero mean: 0.5 * (0.2, 0.4, 0.6) = (0.1, 0.2, 0.3)
            # -> (25, 51, 76).
            (0.5, (0.0, 0.0, 0.0), (0.2, 0.4, 0.6), (25, 51, 76)),
            # Values > 1 are clamped to 1 -> 255.
            (5.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (255, 255, 255)),
            # Values < 0 are clamped to 0.
            (-5.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0, 0, 0)),
        ],
    )
    def test_plot_image_classification_labels_mean_std_denormalizes_image(
        self,
        image_value: float,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        expected_pixel: tuple[int, int, int],
    ) -> None:
        # Empty classes so no label overlay modifies any pixel.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 32, 32), image_value),
            classes=[torch.zeros(0, dtype=torch.long)],
        )
        result = image_classification.plot_image_classification_labels(
            batch=batch,
            included_classes={},
            max_images=1,
            mean=mean,
            std=std,
        )
        assert result.getpixel((0, 0)) == expected_pixel
        assert result.getpixel((31, 31)) == expected_pixel

    def test_plot_image_classification_labels_no_mean_std_skips_denormalization(
        self,
    ) -> None:
        # Without mean/std the image tensor passes through unchanged.
        # Uniform 0.4 -> 102, no labels drawn.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 32, 32), 0.4),
            classes=[torch.zeros(0, dtype=torch.long)],
        )
        result = image_classification.plot_image_classification_labels(
            batch=batch, included_classes={}, max_images=1
        )
        assert result.getpixel((0, 0)) == (102, 102, 102)
        assert result.getpixel((31, 31)) == (102, 102, 102)

    def test_plot_image_classification_labels_multiple_classes_stack_vertically(
        self,
    ) -> None:
        # Two labels are drawn stacked vertically. The second label extends the
        # darkened overlay area further down compared to a single label.
        image = torch.full((1, 3, 256, 256), _WHITE_COLOR)
        result_one = image_classification.plot_image_classification_labels(
            batch=_make_batch_from_image(
                image=image,
                classes=[torch.tensor([0], dtype=torch.long)],
            ),
            included_classes={0: "cat", 1: "dog"},
            max_images=1,
        )
        result_two = image_classification.plot_image_classification_labels(
            batch=_make_batch_from_image(
                image=image,
                classes=[torch.tensor([0, 1], dtype=torch.long)],
            ),
            included_classes={0: "cat", 1: "dog"},
            max_images=1,
        )
        # Scan downward to find the first row below the single-label overlay.
        _, height = result_one.size
        first_white_y = next(
            y for y in range(height) if result_one.getpixel((0, y)) == _WHITE_PIXEL
        )
        # That row is untouched in result_one (first label ends above it).
        assert result_one.getpixel((0, first_white_y)) == _WHITE_PIXEL
        # With two stacked labels, the second overlay extends into that row.
        assert result_two.getpixel((0, first_white_y)) != _WHITE_PIXEL

    def test_plot_image_classification_labels_mixed_empty_nonempty_annotations(
        self,
    ) -> None:
        # Image 0 has a class label; image 1 has none.
        # Grid is 2×1 (256 wide, 128 tall): image 0 at x=0..127, image 1 at x=128..255.
        batch = _make_batch_from_image(
            image=torch.full((2, 3, 128, 128), _WHITE_COLOR),
            classes=[
                torch.tensor([0], dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            ],
        )
        result = image_classification.plot_image_classification_labels(
            batch=batch, included_classes={0: "cat"}, max_images=2
        )
        # Image 0's top-left is darkened by the label overlay.
        assert result.getpixel((0, 0)) != _WHITE_PIXEL
        # Image 1 has no labels, so its top-left stays white.
        assert result.getpixel((128, 0)) == _WHITE_PIXEL


class TestPlotImageClassificationPredictions:
    def test_plot_image_classification_predictions_grid_caps_at_max_images(
        self,
    ) -> None:
        batch = _make_batch(batch_size=4, height=16, width=16)
        logits = torch.zeros(4, 2)
        result = image_classification.plot_image_classification_predictions(
            batch=batch,
            logits=logits,
            included_classes={},
            max_images=2,
            top_k=1,
        )
        assert result.size == (32, 16)

    def test_plot_image_classification_predictions_label_drawn(self) -> None:
        # Top-1 prediction draws a corner label overlay, darkening the top-left pixel.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 128, 128), _WHITE_COLOR),
            classes=[torch.tensor([0], dtype=torch.long)],
        )
        logits = torch.tensor([[1.0, 5.0]])
        result = image_classification.plot_image_classification_predictions(
            batch=batch,
            logits=logits,
            included_classes={0: "cat", 1: "dog"},
            max_images=1,
            top_k=1,
        )
        assert result.getpixel((0, 0)) != _WHITE_PIXEL
        assert result.getpixel((127, 127)) == _WHITE_PIXEL

    def test_plot_image_classification_predictions_unknown_class_draws_label(
        self,
    ) -> None:
        # Class IDs absent from included_classes get a "Class {id}" fallback label.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 128, 128), _WHITE_COLOR),
            classes=[torch.tensor([0], dtype=torch.long)],
        )
        logits = torch.tensor([[5.0, 1.0]])
        result = image_classification.plot_image_classification_predictions(
            batch=batch,
            logits=logits,
            included_classes={},
            max_images=1,
            top_k=1,
        )
        assert result.getpixel((0, 0)) != _WHITE_PIXEL
        assert result.getpixel((127, 127)) == _WHITE_PIXEL

    @pytest.mark.parametrize(
        "image_value, mean, std, expected_pixel",
        [
            # image=0 -> denormalized = mean. Per-channel means verify channel order.
            # (0.2, 0.4, 0.6) * 255 = (51, 102, 153).
            (0.0, (0.2, 0.4, 0.6), (0.5, 0.5, 0.5), (51, 102, 153)),
            # Non-zero image: pixel = image * std + mean.
            # 0.4 * 0.5 + 0.5 = 0.7 -> 178 (PIL truncates 0.7 * 255 = 178.5).
            (0.4, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (178, 178, 178)),
            # Per-channel std with zero mean: 0.5 * (0.2, 0.4, 0.6) = (0.1, 0.2, 0.3)
            # -> (25, 51, 76).
            (0.5, (0.0, 0.0, 0.0), (0.2, 0.4, 0.6), (25, 51, 76)),
            # Values > 1 are clamped to 1 -> 255.
            (5.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (255, 255, 255)),
            # Values < 0 are clamped to 0.
            (-5.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0, 0, 0)),
        ],
    )
    def test_plot_image_classification_predictions_mean_std_denormalizes_image(
        self,
        image_value: float,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        expected_pixel: tuple[int, int, int],
    ) -> None:
        # Use a 128×128 image and check the far corner (127, 127), which sits well
        # below the corner-label overlay area and reflects the denormalized image.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 128, 128), image_value),
            classes=[torch.zeros(0, dtype=torch.long)],
        )
        logits = torch.zeros(1, 2)
        result = image_classification.plot_image_classification_predictions(
            batch=batch,
            logits=logits,
            included_classes={},
            max_images=1,
            top_k=1,
            mean=mean,
            std=std,
        )
        assert result.getpixel((127, 127)) == expected_pixel

    def test_plot_image_classification_predictions_no_mean_std_skips_denormalization(
        self,
    ) -> None:
        # Without mean/std the image tensor passes through unchanged.
        # Check the far corner to avoid the corner-label overlay area.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 128, 128), 0.4),
            classes=[torch.zeros(0, dtype=torch.long)],
        )
        logits = torch.zeros(1, 2)
        result = image_classification.plot_image_classification_predictions(
            batch=batch,
            logits=logits,
            included_classes={},
            max_images=1,
            top_k=1,
        )
        assert result.getpixel((127, 127)) == (102, 102, 102)

    def test_plot_image_classification_predictions_effective_k_multi_label(
        self,
    ) -> None:
        # With top_k=1 but two ground-truth labels, effective_k=max(1,2)=2 predictions
        # are drawn. This results in a larger overlay area than top_k=1 alone.
        image = torch.full((1, 3, 256, 256), _WHITE_COLOR)
        logits = torch.tensor([[10.0, 5.0, 1.0]])
        result_one_gt = image_classification.plot_image_classification_predictions(
            batch=_make_batch_from_image(
                image=image,
                classes=[torch.tensor([0], dtype=torch.long)],
            ),
            logits=logits,
            included_classes={0: "cat", 1: "dog", 2: "bird"},
            max_images=1,
            top_k=1,
        )
        result_two_gt = image_classification.plot_image_classification_predictions(
            batch=_make_batch_from_image(
                image=image,
                classes=[torch.tensor([0, 1], dtype=torch.long)],
            ),
            logits=logits,
            included_classes={0: "cat", 1: "dog", 2: "bird"},
            max_images=1,
            top_k=1,
        )
        # Scan downward to find the first row below the single-prediction overlay.
        # This is robust to changes in font size or padding.
        _, height = result_one_gt.size
        first_white_y = next(
            y for y in range(height) if result_one_gt.getpixel((0, y)) == _WHITE_PIXEL
        )
        # That row is untouched in result_one_gt (one prediction ends above it).
        assert result_one_gt.getpixel((0, first_white_y)) == _WHITE_PIXEL
        # With effective_k=2, the second prediction overlay extends into that row.
        assert result_two_gt.getpixel((0, first_white_y)) != _WHITE_PIXEL

    def test_plot_image_classification_predictions_mixed_empty_nonempty_annotations(
        self,
    ) -> None:
        # Image 0 has a ground-truth label; image 1 has none. Both still receive
        # top_k=1 predictions (effective_k=max(1,0)=1 for image 1).
        # Grid is 2×1 (256 wide, 128 tall): image 0 at x=0..127, image 1 at x=128..255.
        batch = _make_batch_from_image(
            image=torch.full((2, 3, 128, 128), _WHITE_COLOR),
            classes=[
                torch.tensor([0], dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
            ],
        )
        logits = torch.zeros(2, 2)
        result = image_classification.plot_image_classification_predictions(
            batch=batch,
            logits=logits,
            included_classes={0: "cat"},
            max_images=2,
            top_k=1,
        )
        assert result.size == (256, 128)
        # Both images get a top-1 prediction drawn.
        assert result.getpixel((0, 0)) != _WHITE_PIXEL
        assert result.getpixel((128, 0)) != _WHITE_PIXEL

    def test_plot_image_classification_predictions_multilabel_uses_sigmoid(
        self,
    ) -> None:
        # For multilabel, scores are sigmoid(logits) and do NOT sum to 1. Verify
        # that the two displayed scores are both sigmoid values, not softmax values.
        batch = _make_batch_from_image(
            image=torch.full((1, 3, 32, 32), _WHITE_COLOR),
            classes=[torch.tensor([0, 1], dtype=torch.long)],
        )
        # logits [2.0, 1.0]: sigmoid -> [0.88, 0.73], softmax -> [0.73, 0.27].
        # With softmax the scores sum to 1; with sigmoid they exceed 1 in total.
        logits = torch.tensor([[2.0, 1.0]])

        # The function should not raise and should use sigmoid-based ordering/scores.
        result = image_classification.plot_image_classification_predictions(
            batch=batch,
            logits=logits,
            included_classes={0: "cat", 1: "dog"},
            max_images=1,
            top_k=2,
            classification_task="multilabel",
        )
        # With sigmoid both scores are > 0.5 so the sum exceeds 1.0, which would be
        # impossible under softmax.  Verify the corner overlay is drawn (non-white).
        assert result.getpixel((0, 0)) != _WHITE_PIXEL
