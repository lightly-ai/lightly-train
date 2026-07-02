#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch

from lightly_train._task_models.depth_estimation.criterion import (
    GradientMatchingLoss,
    SILogLoss,
    SkyDistillLoss,
)


class TestSILogLoss:
    def test_forward__zero_when_pred_equals_target(self) -> None:
        pred = torch.rand(2, 1, 8, 8) + 0.5
        target = pred.clone()
        mask = torch.ones_like(pred, dtype=torch.bool)

        loss = SILogLoss(eps=0.0)(pred, target, mask)

        assert float(loss) == 0.0

    def test_forward__ignores_invalid_pixels(self) -> None:
        # Garbage predictions in the invalid region must not change the loss.
        pred = torch.rand(1, 1, 4, 4) + 0.5
        target = pred.clone()
        target[:, :, :2, :] = 0.0  # Mark the top half invalid.
        mask = target > 0
        pred_garbage = pred.clone()
        pred_garbage[:, :, :2, :] = 999.0

        loss = SILogLoss(eps=0.0)(pred_garbage, target, mask)

        assert float(loss) == 0.0

    def test_forward__all_invalid_returns_finite_with_grad(self) -> None:
        pred = (torch.rand(1, 1, 4, 4) + 0.5).requires_grad_()
        target = torch.zeros(1, 1, 4, 4)
        mask = target > 0

        loss = SILogLoss()(pred, target, mask)

        assert torch.isfinite(loss)
        assert loss.requires_grad
        assert float(loss.detach()) == 0.0

    def test_forward__positive_for_mismatch(self) -> None:
        pred = torch.rand(2, 1, 8, 8) + 0.5
        target = torch.rand(2, 1, 8, 8) + 0.5
        mask = torch.ones_like(pred, dtype=torch.bool)

        loss = SILogLoss()(pred, target, mask)

        assert float(loss) > 0.0


class TestGradientMatchingLoss:
    def test_forward__zero_when_pred_equals_target(self) -> None:
        pred = torch.rand(2, 1, 16, 16) + 0.5
        target = pred.clone()
        mask = torch.ones_like(pred, dtype=torch.bool)

        loss = GradientMatchingLoss()(pred, target, mask)

        assert float(loss) == 0.0

    def test_forward__penalizes_edge_mismatch(self) -> None:
        # A sharp edge in the prediction that is absent in a flat target is penalized.
        target = torch.ones(1, 1, 8, 8)
        pred = torch.ones(1, 1, 8, 8)
        pred[:, :, :, 4:] = 3.0
        mask = torch.ones_like(pred, dtype=torch.bool)

        loss = GradientMatchingLoss()(pred, target, mask)

        assert float(loss) > 0.0


class TestSkyDistillLoss:
    def test_forward__low_for_perfect_match(self) -> None:
        sky = torch.rand(2, 1, 8, 8)

        loss_match = SkyDistillLoss()(sky, sky)
        loss_mismatch = SkyDistillLoss()(sky, 1.0 - sky)

        assert float(loss_match) < float(loss_mismatch)
