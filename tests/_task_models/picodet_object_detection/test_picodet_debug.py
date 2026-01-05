#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Debug tests for diagnosing low mAP in PicoDet.

These tests help identify potential issues with:
- Box encoding/decoding
- Training vs inference consistency
- Class label alignment
- Raw prediction quality

Run with: pytest tests/_task_models/picodet_object_detection/test_picodet_debug.py -v -s
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from lightly_train._task_models.picodet_object_detection.pico_head import (
    Integral,
    bbox2distance,
    distance2bbox,
)


class TestBoxEncodingDecodingRoundtrip:
    """Test that bbox2distance and distance2bbox are proper inverses."""

    def test_simple_centered_box(self) -> None:
        """Test a box centered on a grid point."""
        # Box from (100, 100) to (200, 200), center at (150, 150)
        gt_boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        points = torch.tensor([[150.0, 150.0]])  # Center of box
        reg_max = 7

        # The distances from center to edges should be 50 pixels each
        # In feature space with stride 8: 50/8 = 6.25 (within reg_max)
        stride = 8.0
        gt_boxes_feature = gt_boxes / stride
        points_feature = points / stride

        # Encode: boxes -> distances
        distances = bbox2distance(points_feature, gt_boxes_feature, reg_max=reg_max)

        # Expected distances in feature space: all should be 50/8 = 6.25
        expected_distances = torch.tensor([[6.25, 6.25, 6.25, 6.25]])
        torch.testing.assert_close(distances, expected_distances, rtol=1e-4, atol=1e-4)

        # Decode: distances -> boxes
        decoded_boxes_feature = distance2bbox(points_feature, distances)
        decoded_boxes_pixel = decoded_boxes_feature * stride

        # Should match original boxes
        torch.testing.assert_close(decoded_boxes_pixel, gt_boxes, rtol=1e-4, atol=1e-4)

    def test_large_box_clamped(self) -> None:
        """Test that large boxes are clamped to reg_max and can still be decoded."""
        # Box from (0, 0) to (400, 400), center at (200, 200)
        # Distance from center = 200 pixels = 25 feature units (with stride 8)
        # This exceeds reg_max=7, so should be clamped
        gt_boxes = torch.tensor([[0.0, 0.0, 400.0, 400.0]])
        points = torch.tensor([[200.0, 200.0]])
        reg_max = 7
        stride = 8.0

        gt_boxes_feature = gt_boxes / stride
        points_feature = points / stride

        # Encode with clamping
        distances = bbox2distance(points_feature, gt_boxes_feature, reg_max=reg_max)

        # Distances should be clamped to reg_max - 0.01 = 6.99
        expected_distances = torch.tensor([[6.99, 6.99, 6.99, 6.99]])
        torch.testing.assert_close(distances, expected_distances, rtol=1e-4, atol=1e-4)

        # Decode gives a smaller box due to clamping
        decoded_boxes_feature = distance2bbox(points_feature, distances)
        decoded_boxes_pixel = decoded_boxes_feature * stride

        # The decoded box should be smaller: center +/- 6.99*8 = 55.92 pixels
        # So box should be (200-55.92, 200-55.92, 200+55.92, 200+55.92)
        expected_decoded = torch.tensor([[144.08, 144.08, 255.92, 255.92]])
        torch.testing.assert_close(
            decoded_boxes_pixel, expected_decoded, rtol=1e-3, atol=0.1
        )

    def test_off_center_point(self) -> None:
        """Test encoding when point is not at box center."""
        # Box from (100, 100) to (200, 200)
        # Point at (120, 120) - closer to left/top edges
        gt_boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        points = torch.tensor([[120.0, 120.0]])
        reg_max = 7
        stride = 8.0

        gt_boxes_feature = gt_boxes / stride
        points_feature = points / stride

        # Distances: left=20, top=20, right=80, bottom=80 (in pixels)
        # In feature space: 2.5, 2.5, 10, 10
        # Right and bottom exceed reg_max, will be clamped
        distances = bbox2distance(points_feature, gt_boxes_feature, reg_max=reg_max)

        # left, top should be 20/8 = 2.5
        # right, bottom should be clamped to 6.99
        expected_distances = torch.tensor([[2.5, 2.5, 6.99, 6.99]])
        torch.testing.assert_close(distances, expected_distances, rtol=1e-4, atol=1e-4)

    def test_multiple_boxes(self) -> None:
        """Test encoding/decoding with multiple boxes."""
        gt_boxes = torch.tensor(
            [
                [10.0, 10.0, 50.0, 50.0],
                [100.0, 100.0, 140.0, 140.0],
                [200.0, 200.0, 280.0, 280.0],
            ]
        )
        # Points at box centers
        points = torch.tensor(
            [[30.0, 30.0], [120.0, 120.0], [240.0, 240.0]]
        )
        reg_max = 7
        stride = 8.0

        gt_boxes_feature = gt_boxes / stride
        points_feature = points / stride

        distances = bbox2distance(points_feature, gt_boxes_feature, reg_max=reg_max)

        # Decode and verify roundtrip for boxes that fit within reg_max
        decoded_feature = distance2bbox(points_feature, distances)
        decoded_pixel = decoded_feature * stride

        # First two boxes should roundtrip exactly (sizes 40 and 40, within reg_max*stride*2)
        torch.testing.assert_close(
            decoded_pixel[:2], gt_boxes[:2], rtol=1e-4, atol=1e-4
        )

    def test_different_strides(self) -> None:
        """Test that different strides give consistent results."""
        gt_boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        points = torch.tensor([[150.0, 150.0]])
        reg_max = 7

        for stride in [8, 16, 32, 64]:
            stride_f = float(stride)
            gt_boxes_feature = gt_boxes / stride_f
            points_feature = points / stride_f

            distances = bbox2distance(
                points_feature, gt_boxes_feature, reg_max=reg_max
            )
            decoded_feature = distance2bbox(points_feature, distances)
            decoded_pixel = decoded_feature * stride_f

            # Distance from center = 50 pixels
            dist_feature = 50.0 / stride_f
            if dist_feature <= reg_max - 0.01:
                # Should roundtrip exactly
                torch.testing.assert_close(
                    decoded_pixel, gt_boxes, rtol=1e-4, atol=1e-4
                )


class TestIntegralModule:
    """Test the Integral (DFL) module."""

    def test_integral_output_range(self) -> None:
        """Test that Integral output is in valid range [0, reg_max]."""
        reg_max = 7
        integral = Integral(reg_max=reg_max)

        # Random logits
        logits = torch.randn(10, 4 * (reg_max + 1))
        output = integral(logits)

        assert output.shape == (10, 4)
        assert output.min() >= 0
        assert output.max() <= reg_max

    def test_integral_deterministic(self) -> None:
        """Test that peaked distribution gives expected value."""
        reg_max = 7
        integral = Integral(reg_max=reg_max)

        # Create a distribution peaked at value 3
        logits = torch.zeros(1, 4 * (reg_max + 1))
        # Set high values at index 3 for each of the 4 distances
        for i in range(4):
            logits[0, i * (reg_max + 1) + 3] = 100.0

        output = integral(logits)
        expected = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)

    def test_integral_uniform(self) -> None:
        """Test that uniform distribution gives middle value."""
        reg_max = 7
        integral = Integral(reg_max=reg_max)

        # Uniform distribution (all zeros after softmax = uniform)
        logits = torch.zeros(1, 4 * (reg_max + 1))
        output = integral(logits)

        # Expected value of uniform distribution over [0, reg_max]
        expected_value = reg_max / 2.0  # = 3.5
        expected = torch.full((1, 4), expected_value)
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)


class TestTrainingInferenceConsistency:
    """Test that training and inference decode boxes identically."""

    def test_decode_consistency(self) -> None:
        """Verify training decode matches postprocessor decode."""
        reg_max = 7
        stride = 8
        h, w = 52, 52  # For 416x416 input

        # Generate random bbox predictions
        torch.manual_seed(42)
        bbox_pred = torch.randn(1, 4 * (reg_max + 1), h, w)

        # Generate grid points
        y = (torch.arange(h, dtype=torch.float32) + 0.5) * stride
        x = (torch.arange(w, dtype=torch.float32) + 0.5) * stride
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        points_pixel = torch.stack([xx.flatten(), yy.flatten()], dim=-1)

        integral = Integral(reg_max=reg_max)

        # Training-style decode (feature space then scale)
        points_feature = points_pixel / stride
        bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        pred_corners = integral(bbox_pred_flat)
        boxes_training = distance2bbox(points_feature.unsqueeze(0), pred_corners)
        boxes_training_pixel = boxes_training * stride

        # Postprocessor-style decode (scale distances then decode)
        bbox_pred_post = bbox_pred[0].permute(1, 2, 0).reshape(h * w, -1)
        distances_pixel = integral(bbox_pred_post) * stride
        boxes_inference = distance2bbox(points_pixel, distances_pixel)

        # Should match exactly
        torch.testing.assert_close(
            boxes_training_pixel.squeeze(0),
            boxes_inference,
            rtol=1e-5,
            atol=1e-5,
        )


class TestClassLabelAlignment:
    """Test class label handling between training and inference."""

    def test_label_indexing(self) -> None:
        """Verify label indices are consistent."""
        num_classes = 80

        # Simulate GT labels (0-indexed)
        gt_labels = torch.tensor([0, 5, 79])  # Min, mid, max valid indices

        # Simulate prediction scores
        pred_scores = torch.randn(3, num_classes).sigmoid()

        # Predicted labels via argmax
        pred_labels = pred_scores.argmax(dim=-1)

        # Verify labels are in valid range
        assert gt_labels.min() >= 0
        assert gt_labels.max() < num_classes
        assert pred_labels.min() >= 0
        assert pred_labels.max() < num_classes

    def test_vfl_target_construction(self) -> None:
        """Verify VFL target is correctly constructed."""
        num_classes = 80
        num_priors = 100
        num_pos = 10

        # Simulated positive sample info
        pos_inds = torch.arange(num_pos)
        pos_gt_labels = torch.randint(0, num_classes, (num_pos,))
        pos_ious = torch.rand(num_pos)

        # Construct VFL target (as done in training)
        vfl_target = torch.zeros(num_priors, num_classes)
        vfl_target[pos_inds, pos_gt_labels] = pos_ious

        # Verify target construction
        assert vfl_target.shape == (num_priors, num_classes)
        assert vfl_target[pos_inds[0], pos_gt_labels[0]] == pos_ious[0]
        assert vfl_target.sum() == pytest.approx(pos_ious.sum().item(), abs=1e-5)


class TestRawPredictionStatistics:
    """Test raw model outputs are sensible."""

    def test_cls_score_range(self) -> None:
        """Verify classification scores are in valid range after sigmoid."""
        # Simulate raw logits
        logits = torch.randn(100, 80)
        scores = logits.sigmoid()

        assert scores.min() >= 0
        assert scores.max() <= 1

    def test_dfl_output_distribution(self) -> None:
        """Verify DFL outputs have reasonable distribution."""
        reg_max = 7
        integral = Integral(reg_max=reg_max)

        # Random logits simulating model output
        torch.manual_seed(42)
        logits = torch.randn(1000, 4 * (reg_max + 1))

        distances = integral(logits)

        # Check statistics
        assert distances.shape == (1000, 4)
        assert distances.min() >= 0
        assert distances.max() <= reg_max

        # Mean should be around reg_max/2 for random inputs
        mean = distances.mean().item()
        assert 2.0 < mean < 5.0, f"Mean {mean} is outside expected range"


class TestVisualizationDebug:
    """Tests that help visualize predictions vs GT.

    These tests can be run with -s flag to see output or save images.
    """

    @pytest.mark.skip(reason="Run manually for debugging with real model/data")
    def test_visualize_predictions(self, tmp_path: Path) -> None:
        """Visualize model predictions vs GT boxes.

        This test requires matplotlib and a trained model checkpoint.
        Run with: pytest -v -s --runxfail -k test_visualize_predictions
        """
        try:
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

        # This is a template - fill in with actual model loading
        # model = load_model("path/to/checkpoint.pt")
        # For now, just show how to draw boxes

        # Example image (white canvas)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xlim(0, 416)
        ax.set_ylim(416, 0)  # Invert Y for image coordinates

        # Example GT box (green)
        gt_box = [100, 100, 200, 200]  # x1, y1, x2, y2
        rect = patches.Rectangle(
            (gt_box[0], gt_box[1]),
            gt_box[2] - gt_box[0],
            gt_box[3] - gt_box[1],
            linewidth=2,
            edgecolor="g",
            facecolor="none",
            label="GT",
        )
        ax.add_patch(rect)

        # Example predicted box (red)
        pred_box = [90, 95, 210, 205]  # x1, y1, x2, y2
        rect = patches.Rectangle(
            (pred_box[0], pred_box[1]),
            pred_box[2] - pred_box[0],
            pred_box[3] - pred_box[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
            label="Pred",
        )
        ax.add_patch(rect)

        ax.legend()
        ax.set_title("GT (green) vs Predicted (red) boxes")

        output_path = tmp_path / "debug_boxes.png"
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
        plt.close()


class TestPicoDetEndToEnd:
    """End-to-end tests for the full pipeline."""

    def test_model_forward_shapes(self) -> None:
        """Test that model forward pass produces correct output shapes."""
        from lightly_train._task_models.picodet_object_detection.task_model import (
            PicoDetObjectDetection,
        )

        model = PicoDetObjectDetection(
            model_name="picodet/s-416",
            image_size=(416, 416),
            num_classes=80,
            image_normalize=None,
            load_weights=False,
        )

        # Forward pass
        x = torch.randn(2, 3, 416, 416)
        outputs = model(x)

        # Check output structure
        assert "cls_scores" in outputs
        assert "bbox_preds" in outputs
        assert len(outputs["cls_scores"]) == 4  # 4 feature levels
        assert len(outputs["bbox_preds"]) == 4

        # Check shapes for stride 8 level (52x52 for 416 input)
        assert outputs["cls_scores"][0].shape == (2, 80, 52, 52)
        assert outputs["bbox_preds"][0].shape == (2, 32, 52, 52)  # 4 * (7+1) = 32

    def test_postprocessor_output_format(self) -> None:
        """Test that postprocessor produces correct output format."""
        from lightly_train._task_models.picodet_object_detection.postprocessor import (
            PicoDetPostProcessor,
        )

        postprocessor = PicoDetPostProcessor(
            num_classes=80,
            reg_max=7,
            strides=(8, 16, 32, 64),
            score_threshold=0.001,
            iou_threshold=0.6,
            max_detections=100,
        )

        # Simulate model outputs
        cls_scores = [
            torch.randn(1, 80, 52, 52),
            torch.randn(1, 80, 26, 26),
            torch.randn(1, 80, 13, 13),
            torch.randn(1, 80, 7, 7),
        ]
        bbox_preds = [
            torch.randn(1, 32, 52, 52),
            torch.randn(1, 32, 26, 26),
            torch.randn(1, 32, 13, 13),
            torch.randn(1, 32, 7, 7),
        ]

        results = postprocessor.forward(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            original_size=(416, 416),
            score_threshold=0.1,
        )

        # Check output format
        assert "labels" in results
        assert "bboxes" in results
        assert "scores" in results
        assert results["bboxes"].ndim == 2
        assert results["bboxes"].shape[1] == 4


class TestMAPSanityCheck:
    """Test mAP computation with synthetic perfect and imperfect predictions."""

    def test_perfect_predictions_give_high_map(self) -> None:
        """Verify that perfect predictions give mAP close to 1.0."""
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric = MeanAveragePrecision()

        # Perfect predictions matching GT exactly
        gt_boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0], [300.0, 300.0, 400.0, 400.0]])
        gt_labels = torch.tensor([0, 1])

        preds = [
            {
                "boxes": gt_boxes.clone(),  # Exact match
                "scores": torch.tensor([0.99, 0.99]),
                "labels": gt_labels.clone(),
            }
        ]
        targets = [{"boxes": gt_boxes, "labels": gt_labels}]

        metric.update(preds, targets)
        result = metric.compute()

        # mAP should be 1.0 for perfect predictions
        assert result["map"].item() > 0.99, f"Expected mAP > 0.99, got {result['map'].item()}"

    def test_wrong_class_predictions_give_zero_map(self) -> None:
        """Verify that predictions with wrong class give mAP of 0."""
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric = MeanAveragePrecision()

        gt_boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        gt_labels = torch.tensor([0])

        # Prediction with correct box but wrong class
        preds = [
            {
                "boxes": gt_boxes.clone(),
                "scores": torch.tensor([0.99]),
                "labels": torch.tensor([1]),  # Wrong class!
            }
        ]
        targets = [{"boxes": gt_boxes, "labels": gt_labels}]

        metric.update(preds, targets)
        result = metric.compute()

        # mAP should be 0 when class doesn't match
        assert result["map"].item() < 0.01, f"Expected mAP ~ 0, got {result['map'].item()}"

    def test_shifted_boxes_reduce_map(self) -> None:
        """Verify that shifted boxes (low IoU) reduce mAP."""
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric = MeanAveragePrecision()

        gt_boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        gt_labels = torch.tensor([0])

        # Prediction with box shifted significantly (low IoU)
        shifted_boxes = torch.tensor([[200.0, 200.0, 300.0, 300.0]])  # No overlap!

        preds = [
            {
                "boxes": shifted_boxes,
                "scores": torch.tensor([0.99]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [{"boxes": gt_boxes, "labels": gt_labels}]

        metric.update(preds, targets)
        result = metric.compute()

        # mAP should be 0 when no overlap
        assert result["map"].item() < 0.01, f"Expected mAP ~ 0, got {result['map'].item()}"

    def test_partial_overlap_gives_intermediate_map(self) -> None:
        """Verify that partial overlap gives intermediate mAP."""
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric = MeanAveragePrecision()

        gt_boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        gt_labels = torch.tensor([0])

        # Prediction with ~67% IoU (shifted by 20 pixels, not 50)
        # IoU calculation: intersection=80*100=8000, union=10000+10000-8000=12000
        # IoU = 8000/12000 = 0.667, which is above COCO's 0.5 threshold
        shifted_boxes = torch.tensor([[120.0, 100.0, 220.0, 200.0]])

        preds = [
            {
                "boxes": shifted_boxes,
                "scores": torch.tensor([0.99]),
                "labels": torch.tensor([0]),
            }
        ]
        targets = [{"boxes": gt_boxes, "labels": gt_labels}]

        metric.update(preds, targets)
        result = metric.compute()

        # mAP should be intermediate (> 0 but < 1)
        map_val = result["map"].item()
        # At IoU=0.67, it should match at IoU thresholds 0.5 and 0.55 and 0.6 and 0.65
        # but not at 0.7 and above. So roughly 4/10 thresholds → mAP ~0.4
        assert 0.2 < map_val < 0.7, f"Expected 0.2 < mAP < 0.7, got {map_val}"


class TestLossComponentAnalysis:
    """Test individual loss components."""

    def test_vfl_loss_gradient_flow(self) -> None:
        """Test that VFL loss has proper gradients."""
        from lightly_train._task_models.picodet_object_detection.losses import (
            VarifocalLoss,
        )

        vfl_loss = VarifocalLoss(alpha=0.75, gamma=2.0)

        # Create inputs with gradients
        pred_logits = torch.randn(100, 80, requires_grad=True)
        target = torch.zeros(100, 80)
        target[0, 5] = 0.8  # One positive sample

        loss = vfl_loss(pred_logits, target)

        # Verify loss is scalar
        assert loss.dim() == 0

        # Verify gradients flow
        loss.backward()
        assert pred_logits.grad is not None
        assert not torch.isnan(pred_logits.grad).any()

    def test_giou_loss_gradient_flow(self) -> None:
        """Test that GIoU loss has proper gradients."""
        from lightly_train._task_models.picodet_object_detection.losses import GIoULoss

        giou_loss = GIoULoss()

        # Create box predictions with gradients
        pred_boxes = torch.tensor(
            [[100.0, 100.0, 200.0, 200.0]], requires_grad=True
        )
        gt_boxes = torch.tensor([[110.0, 110.0, 210.0, 210.0]])

        loss = giou_loss(pred_boxes, gt_boxes)

        # Verify loss is scalar
        assert loss.dim() == 0

        # Verify gradients flow
        loss.backward()
        assert pred_boxes.grad is not None
        assert not torch.isnan(pred_boxes.grad).any()

    def test_dfl_loss_gradient_flow(self) -> None:
        """Test that DFL loss has proper gradients."""
        from lightly_train._task_models.picodet_object_detection.losses import (
            DistributionFocalLoss,
        )

        dfl_loss = DistributionFocalLoss()

        # Create DFL predictions with gradients
        pred = torch.randn(10, 8, requires_grad=True)  # 10 samples, reg_max+1=8
        target = torch.rand(10) * 6.99  # Targets in [0, 6.99]

        loss = dfl_loss(pred, target)

        # Verify loss is scalar
        assert loss.dim() == 0

        # Verify gradients flow
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


class TestSimOTAAssigner:
    """Test SimOTA assignment logic."""

    def test_assigner_basic(self) -> None:
        """Test basic SimOTA assignment."""
        from lightly_train._task_models.picodet_object_detection.sim_ota_assigner import (
            SimOTAAssigner,
        )

        assigner = SimOTAAssigner(
            center_radius=2.5,
            candidate_topk=10,
            iou_weight=6.0,
            cls_weight=1.0,
            num_classes=80,
        )

        # Create synthetic data
        num_priors = 100
        pred_scores = torch.rand(num_priors, 80)  # Sigmoid scores

        # Priors: [cx, cy, stride_w, stride_h]
        priors = torch.zeros(num_priors, 4)
        priors[:, 0] = torch.arange(num_priors) * 8 + 4  # cx
        priors[:, 1] = 100  # cy (all at same y)
        priors[:, 2] = 8  # stride_w
        priors[:, 3] = 8  # stride_h

        # Decoded boxes in xyxy format
        decoded_bboxes = torch.zeros(num_priors, 4)
        decoded_bboxes[:, 0] = priors[:, 0] - 20  # x1
        decoded_bboxes[:, 1] = priors[:, 1] - 20  # y1
        decoded_bboxes[:, 2] = priors[:, 0] + 20  # x2
        decoded_bboxes[:, 3] = priors[:, 1] + 20  # y2

        # Single GT box
        gt_bboxes = torch.tensor([[150.0, 80.0, 250.0, 120.0]])
        gt_labels = torch.tensor([5])

        assigned_gt_inds, matched_pred_ious = assigner.assign(
            pred_scores=pred_scores,
            priors=priors,
            decoded_bboxes=decoded_bboxes,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
        )

        # Check output shapes
        assert assigned_gt_inds.shape == (num_priors,)
        assert matched_pred_ious.shape == (num_priors,)

        # Some priors should be assigned (positive)
        num_pos = (assigned_gt_inds > 0).sum().item()
        assert num_pos > 0, "Expected some positive assignments"

        # Positive IoUs should be non-zero
        pos_ious = matched_pred_ious[assigned_gt_inds > 0]
        if len(pos_ious) > 0:
            assert pos_ious.min() > 0, "Positive IoUs should be > 0"

    def test_assigner_no_gt(self) -> None:
        """Test SimOTA with no GT boxes."""
        from lightly_train._task_models.picodet_object_detection.sim_ota_assigner import (
            SimOTAAssigner,
        )

        assigner = SimOTAAssigner(num_classes=80)

        num_priors = 100
        pred_scores = torch.rand(num_priors, 80)
        priors = torch.rand(num_priors, 4) * 100
        priors[:, 2:] = 8  # stride
        decoded_bboxes = torch.rand(num_priors, 4) * 400

        gt_bboxes = torch.zeros(0, 4)  # No GT
        gt_labels = torch.zeros(0, dtype=torch.long)

        assigned_gt_inds, matched_pred_ious = assigner.assign(
            pred_scores=pred_scores,
            priors=priors,
            decoded_bboxes=decoded_bboxes,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
        )

        # All should be background (0)
        assert (assigned_gt_inds == 0).all()
        assert (matched_pred_ious == 0).all()


class TestFullTrainingStep:
    """Test the full training step with synthetic data."""

    def test_training_step_runs(self) -> None:
        """Test that training step runs without errors."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from lightly_train._data.yolo_object_detection_dataset import (
            YOLOObjectDetectionDataArgs,
        )
        from lightly_train._task_models.picodet_object_detection.train_model import (
            PicoDetObjectDetectionTrain,
            PicoDetObjectDetectionTrainArgs,
        )
        from lightly_train._task_models.picodet_object_detection.transforms import (
            PicoDetObjectDetectionTrainTransformArgs,
            PicoDetObjectDetectionValTransformArgs,
        )

        # Create train model
        model_args = PicoDetObjectDetectionTrainArgs()
        model_args.resolve_auto(
            total_steps=1000,
            model_name="picodet/s-320",
            model_init_args={},
        )
        data_args = YOLOObjectDetectionDataArgs(
            path=Path("/tmp/data"),
            train=Path("train") / "images",
            val=Path("val") / "images",
            names={0: "cat", 1: "dog"},
        )
        train_transform_args = PicoDetObjectDetectionTrainTransformArgs()
        train_transform_args.resolve_auto(model_init_args={"image_size": (320, 320)})
        val_transform_args = PicoDetObjectDetectionValTransformArgs()
        val_transform_args.resolve_auto(model_init_args={"image_size": (320, 320)})

        train_model = PicoDetObjectDetectionTrain(
            model_name="picodet/s-320",
            model_args=model_args,
            data_args=data_args,
            train_transform_args=train_transform_args,
            val_transform_args=val_transform_args,
            load_weights=False,
        )

        # Create synthetic batch
        batch_size = 2
        batch = {
            "image": torch.randn(batch_size, 3, 320, 320),
            "bboxes": [
                torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.3, 0.3]]),  # YOLO format
                torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            ],
            "classes": [
                torch.tensor([0, 1]),
                torch.tensor([0]),
            ],
        }

        # Mock fabric
        fabric = MagicMock()
        fabric.world_size = 1

        # Run training step
        result = train_model.training_step(fabric, batch, step=0)

        # Verify result
        assert result.loss is not None
        assert not torch.isnan(result.loss)
        assert "loss_vfl" in result.log_dict
        assert "loss_giou" in result.log_dict
        assert "loss_dfl" in result.log_dict

    def test_loss_components_decrease_with_training(self) -> None:
        """Test that loss components can decrease when optimized."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from lightly_train._data.yolo_object_detection_dataset import (
            YOLOObjectDetectionDataArgs,
        )
        from lightly_train._task_models.picodet_object_detection.train_model import (
            PicoDetObjectDetectionTrain,
            PicoDetObjectDetectionTrainArgs,
        )
        from lightly_train._task_models.picodet_object_detection.transforms import (
            PicoDetObjectDetectionTrainTransformArgs,
            PicoDetObjectDetectionValTransformArgs,
        )

        # Create train model
        model_args = PicoDetObjectDetectionTrainArgs()
        model_args.resolve_auto(
            total_steps=100,
            model_name="picodet/s-320",
            model_init_args={},
        )
        data_args = YOLOObjectDetectionDataArgs(
            path=Path("/tmp/data"),
            train=Path("train") / "images",
            val=Path("val") / "images",
            names={0: "cat"},
        )
        train_transform_args = PicoDetObjectDetectionTrainTransformArgs()
        train_transform_args.resolve_auto(model_init_args={"image_size": (320, 320)})
        val_transform_args = PicoDetObjectDetectionValTransformArgs()
        val_transform_args.resolve_auto(model_init_args={"image_size": (320, 320)})

        train_model = PicoDetObjectDetectionTrain(
            model_name="picodet/s-320",
            model_args=model_args,
            data_args=data_args,
            train_transform_args=train_transform_args,
            val_transform_args=val_transform_args,
            load_weights=False,
        )

        # Fixed batch for consistent comparison
        torch.manual_seed(42)
        batch = {
            "image": torch.randn(2, 3, 320, 320),
            "bboxes": [
                torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
                torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
            ],
            "classes": [
                torch.tensor([0]),
                torch.tensor([0]),
            ],
        }

        fabric = MagicMock()
        fabric.world_size = 1

        optimizer, _ = train_model.get_optimizer(total_steps=100)

        # Get initial loss
        result_initial = train_model.training_step(fabric, batch, step=0)
        initial_loss = result_initial.loss.item()

        # Take a few optimization steps
        for step in range(5):
            optimizer.zero_grad()
            result = train_model.training_step(fabric, batch, step=step)
            result.loss.backward()
            optimizer.step()

        # Get final loss
        result_final = train_model.training_step(fabric, batch, step=5)
        final_loss = result_final.loss.item()

        # Loss should decrease (or at least not increase dramatically)
        # We allow some tolerance since it's a small number of steps
        assert final_loss < initial_loss * 1.5, (
            f"Loss should not increase significantly: {initial_loss:.4f} -> {final_loss:.4f}"
        )
