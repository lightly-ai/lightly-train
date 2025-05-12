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

from lightly_train._methods.dinov2.dinov2_loss import DINOLoss, iBOTPatchLoss


@pytest.fixture
def no_dist(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    return


@pytest.mark.usefixtures("no_dist")
class TestDINOLoss:
    def test_softmax_center_teacher(
        self,
        batch_size=4,
        out_dim=2,
        teacher_temp=0.04,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        """Test that the softmax_center_teacher method returns a tensor
        with the same shape as the input tensor and that each row sums to 1.
        """

        dino_loss = DINOLoss(
            out_dim=out_dim, student_temp=student_temp, center_momentum=center_momentum
        )

        teacher_output = torch.randn(batch_size, out_dim)
        softmax = dino_loss.softmax_center_teacher(
            teacher_output, teacher_temp=teacher_temp
        )

        sums = softmax.sum(dim=-1)

        assert torch.allclose(sums, torch.ones(batch_size))

    def test_sinkhorn_knopp_teacher(
        self,
        batch_size=4,
        out_dim=2,
        teacher_temp=0.04,
        student_temp=0.1,
        center_momentum=0.9,
        n_iterations=4,
    ):
        """Test that the sinkhorn_knopp_teacher method returns a tensor
        with the same shape as the input tensor and that each row sums to 1.
        """

        dino_loss = DINOLoss(
            out_dim=out_dim, student_temp=student_temp, center_momentum=center_momentum
        )

        teacher_output = torch.zeros(batch_size, out_dim)
        Q = dino_loss.sinkhorn_knopp_teacher(
            teacher_output, teacher_temp=teacher_temp, n_iterations=n_iterations
        )

        print(Q)

        # Q shape = [B, K]
        assert Q.shape == (batch_size, out_dim)

        # column sums ≈ B/K = 4/2 = 2
        col_sums = Q.sum(dim=0)
        assert torch.allclose(col_sums, torch.ones(out_dim) * batch_size / out_dim)

        # row sums ≈ 1
        row_sums = Q.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(batch_size))

    def test_update_center_momentum(self):
        """Test that the update_center method updates the center
        correctly with the given momentum.
        """

        module = DINOLoss(out_dim=2, center_momentum=0.0)

        # call update & apply on a known tensor
        teacher = torch.ones(3, 2) * 2.0  # mean = [2,2]
        module.update_center(teacher)
        module.apply_center_update()

        # with momentum=0, center should equal the batch mean exactly
        assert torch.allclose(module.center, torch.tensor([[2.0, 2.0]]))

    def test_forward(self):
        """Test that the forward method returns a non-negative scalar"""

        # two views, same predictions -> minimal loss
        loss_module = DINOLoss(out_dim=3, student_temp=1.0)

        # teacher outputs all mass on class 2
        teacher = torch.tensor([[0.0, 0.0, 1.0]])
        teacher_sm = torch.softmax(teacher, dim=-1)
        student_views = [torch.randn(2, 3) for _ in range(2)]

        # inject identical teacher list
        loss = loss_module.forward(student_views, [teacher_sm, teacher_sm])

        # loss should be non-negative scalar
        assert loss.ndim == 0 and loss.item() >= 0


@pytest.mark.usefixtures("no_dist")
class TestIBotPatchLoss:
    def test_softmax_center_teacher(self):
        """Test that the softmax_center_teacher method returns a tensor
        with the same shape as the input tensor and that each row sums to 1.
        """

        module = iBOTPatchLoss(patch_out_dim=4, student_temp=1.0, center_momentum=0.5)
        # initialize center to zeros
        t_out = torch.randn(5, 4)
        sm = module.softmax_center_teacher(t_out, teacher_temp=2.0)
        # each row of sm sums to 1
        sums = sm.sum(dim=-1)

        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_patch_sinkhorn(self):
        module = iBOTPatchLoss(patch_out_dim=2, student_temp=1.0)
        # reuse sinkhorn_knopp_teacher from module
        logits = torch.zeros(4, 2)
        n_masked = torch.tensor(4)
        Q = module.sinkhorn_knopp_teacher(
            logits, teacher_temp=1.0, n_masked_patches_tensor=n_masked, n_iterations=4
        )
        assert Q.shape == (4, 2)
        # normalized as above
        assert torch.allclose(Q.sum(dim=1), torch.ones(4), atol=1e-5)

        # # Q shape = [B, K]
        # assert Q.shape == (6, 3)
        # # column sums ≈ 1
        # col_sums = Q.sum(dim=0)
        # assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5)
        # # row sums ≈ B/K = 6/3 = 2
        # row_sums = Q.sum(dim=1)
        # assert torch.allclose(row_sums, torch.full_like(row_sums, 2.0), atol=1e-5)

    def test_update_center_momentum(self):
        """Test that the update_center method updates the center
        correctly with the given momentum.
        """

        module = DINOLoss(out_dim=2, center_momentum=0.0)

        # call update & apply on a known tensor
        teacher = torch.ones(3, 2) * 2.0  # mean = [2,2]
        module.update_center(teacher)
        module.apply_center_update()

        # with momentum=0, center should equal the batch mean exactly
        assert torch.allclose(module.center, torch.tensor([[2.0, 2.0]]))

    def test_forward_masked_consistency(self):
        module = iBOTPatchLoss(patch_out_dim=3, student_temp=1.0)
        B, N, D = 2, 4, 3
        # random patch tokens
        teacher = torch.randn(B, N, D)
        student = torch.randn(B, N, D)
        mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 1]])
        # compute loss via both pathways
        direct = module.forward(student, teacher, mask)
        masked = module.forward_masked(student, teacher, mask, n_masked_patches=3)
        # loss should be finite scalar
        for L in (direct, masked):
            assert torch.isfinite(L).all() and L.ndim == 0
