#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pytest
import torch
import torch.distributed as dist
from _pytest.monkeypatch import MonkeyPatch

from lightly_train._methods.dinov2.dinov2_loss import DINOLoss, IBOTPatchLoss


@pytest.fixture
def no_dist(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    return


@pytest.mark.usefixtures("no_dist")
class TestDINOLoss:
    def test_softmax_center_teacher(self) -> None:
        """Test that the softmax_center_teacher method returns a tensor
        with the same shape as the input tensor and that each row sums to 1.
        """
        batch_size = 4
        out_dim = 2

        dino_loss = DINOLoss(out_dim=out_dim, student_temp=0.1, center_momentum=0.9)

        teacher_output = torch.randn(batch_size, out_dim)
        softmax = dino_loss.softmax_center_teacher(teacher_output, teacher_temp=0.04)

        sums = softmax.sum(dim=-1)

        assert torch.allclose(sums, torch.ones(batch_size))

    def test_sinkhorn_knopp_teacher(self) -> None:
        """Test that the sinkhorn_knopp_teacher method returns a tensor
        with the same shape as the input tensor and that each row sums to 1.
        """
        batch_size = 4
        out_dim = 2

        dino_loss = DINOLoss(out_dim=out_dim, student_temp=0.1, center_momentum=0.9)

        teacher_output = torch.randn(batch_size, out_dim)
        Q = dino_loss.sinkhorn_knopp_teacher(
            teacher_output, teacher_temp=0.04, n_iterations=4
        )

        # Q shape = [B, K]
        assert Q.shape == (batch_size, out_dim)

        # row sums ≈ 1
        row_sums = Q.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(batch_size))

    def test_update_center_momentum(self) -> None:
        """Test that the update_center method updates the center
        correctly with the given momentum.
        """
        batch_size = 4
        out_dim = 2
        center_momentum = 0.9
        mean = 2

        dino_loss = DINOLoss(
            out_dim=out_dim, student_temp=0.1, center_momentum=center_momentum
        )

        # call update & apply on a known tensor
        teacher_output = torch.ones(batch_size, out_dim) * mean
        dino_loss.update_center(teacher_output)
        dino_loss.apply_center_update()

        expected_center = mean * (1 - center_momentum) * torch.ones(out_dim)
        assert torch.allclose(dino_loss.center, expected_center)

    def test_forward(self) -> None:
        """Test that the forward method returns correct values"""
        out_dim = 2
        teacher_temp = 0.04

        dino_loss = DINOLoss(out_dim=out_dim, student_temp=0.1, center_momentum=0.9)

        teacher_output = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        student_output = [
            torch.tensor([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]) for _ in range(2)
        ]

        teacher_softmaxed = dino_loss.softmax_center_teacher(
            teacher_output, teacher_temp=teacher_temp
        )
        dino_loss.update_center(teacher_output)

        loss = dino_loss.forward(student_output, [teacher_softmaxed, teacher_softmaxed])

        assert loss == pytest.approx(1.5565, rel=0.0001)


@pytest.mark.usefixtures("no_dist")
class TestIBotPatchLoss:
    def test_softmax_center_teacher(self) -> None:
        """Test that the softmax_center_teacher method returns a tensor
        with the same shape as the input tensor and that each row sums to 1.
        """
        batch_size = 4
        patch_out_dim = 2

        ibot_loss = IBOTPatchLoss(
            patch_out_dim=patch_out_dim,
            student_temp=0.1,
            center_momentum=0.9,
        )

        teacher_output = torch.randn(batch_size, patch_out_dim)
        softmax = ibot_loss.softmax_center_teacher(teacher_output, teacher_temp=0.04)

        sums = softmax.sum(dim=-1)

        assert torch.allclose(sums, torch.ones(batch_size))

    def test_sinkhorn_knopp_teacher(self) -> None:
        """Test that the sinkhorn_knopp_teacher method returns a tensor
        with the same shape as the input tensor and that each row sums to 1.
        """
        batch_size = 4
        patch_out_dim = 2

        ibot_loss = IBOTPatchLoss(
            patch_out_dim=patch_out_dim,
            student_temp=0.1,
            center_momentum=0.9,
        )

        teacher_output = torch.randn(batch_size, patch_out_dim)
        n_masked_patches_tensor = torch.randint(low=1, high=batch_size, size=(1,))
        Q = ibot_loss.sinkhorn_knopp_teacher(
            teacher_output,
            teacher_temp=0.04,
            n_masked_patches_tensor=n_masked_patches_tensor,
        )

        # Q shape = [B, K]
        assert Q.shape == (batch_size, patch_out_dim)

        # row sums ≈ 1
        row_sums = Q.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(batch_size))

    def test_update_center_momentum(self) -> None:
        """Test that the update_center method updates the center
        correctly with the given momentum.
        """
        batch_size = 4
        patch_out_dim = 2
        center_momentum = 0.9
        mean = 2

        ibot_loss = IBOTPatchLoss(
            patch_out_dim=patch_out_dim,
            student_temp=0.1,
            center_momentum=center_momentum,
        )

        # call update & apply on a known tensor
        teacher_output = torch.ones(batch_size, patch_out_dim) * mean
        ibot_loss.update_center(teacher_output)
        ibot_loss.apply_center_update()

        expected_center = mean * (1 - center_momentum) * torch.ones(patch_out_dim)
        assert torch.allclose(ibot_loss.center, expected_center)

    def test_forward_masked(self) -> None:
        """Test that the forward method returns correct values"""
        out_dim = 2
        teacher_temp = 0.1

        ibot_loss = IBOTPatchLoss(
            patch_out_dim=out_dim,
            student_temp=0.2,
            center_momentum=0.9,
        )

        masked_teacher_output = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        masked_student_output = torch.tensor([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]])

        mask = torch.tensor(
            [
                [True, False, True, False],
                [False, False, False, True],
                [False, False, False, False],
            ]
        )

        masked_teacher_softmaxed = ibot_loss.softmax_center_teacher(
            masked_teacher_output.unsqueeze(0), teacher_temp=teacher_temp
        )
        ibot_loss.update_center(masked_teacher_output.unsqueeze(0))

        loss = ibot_loss.forward_masked(
            teacher_patch_tokens_masked=masked_teacher_softmaxed,
            student_patch_tokens_masked=masked_student_output,
            student_masks_flat=mask,
        )
        assert loss == pytest.approx(0.4057, rel=0.0001)
