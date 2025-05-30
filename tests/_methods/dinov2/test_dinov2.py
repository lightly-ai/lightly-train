import torch
from pytest_mock import MockerFixture
from torch import Size

from lightly_train._methods.dinov2.dinov2 import (
    DINOv2,
    DINOv2AdamWViTSBArgs,
    DINOv2Args,
)
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._scaling import ScalingInfo
from lightly_train.types import Batch

from ...helpers import DummyVitModel

#TODO test the utils

#TODO: test the method, test the args
def setup_dinov2_helper(dinov2_args: DINOv2Args, mocker: MockerFixture, emb_model: EmbeddingModel, batch_size: int) -> DINOv2:
    optimizer_args = DINOv2AdamWViTSBArgs()
    scaling_info = ScalingInfo(dataset_size=1000, epochs=100)
    dinov2_args.resolve_auto(
        scaling_info=scaling_info,
        optimizer_args=optimizer_args,
        model=emb_model.wrapped_model.get_model(),
    )

    dinov2 = DINOv2(
        method_args=dinov2_args,
        optimizer_args=optimizer_args,
        embedding_model=emb_model,
        global_batch_size=batch_size,
    )

    trainer_mock = mocker.Mock()
    trainer_mock.global_step = 0
    trainer_mock.max_epochs = 1
    trainer_mock.estimated_stepping_batches = 1

    dinov2.trainer = trainer_mock

    return dinov2


class TestDINOv2:
    def test_train_step_impl__default(self, mocker: MockerFixture) -> None:
        emb_model = EmbeddingModel(wrapped_model=DummyVitModel())
        b = 16

        views = [torch.rand(b, 3, 8, 8) for _ in range(2)] + [torch.rand(b, 3, 4, 4) for _ in range(8)]
        batch: Batch = {
            "views": views,
            "filename": [f"img_{i}" for i in range(b)],
        }

        # run DistillationV2
        dinov2_args = DINOv2Args()
        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, b)

        out = dinov2.training_step_impl(batch, 0)

        # check thatthe ibot and dino heads are the same
        assert dinov2.student_dino_head == dinov2.student_ibot_head
        assert len(list(dinov2.student_dino_head.parameters())) == len(list(dinov2.student_ibot_head.parameters()))
        for (name_dino, param_dino), (name_ibot, param_ibot) in zip(
            dinov2.student_dino_head.named_parameters(), dinov2.student_ibot_head.named_parameters()
        ):
            assert name_dino == name_ibot
            assert param_dino.dtype == param_ibot.dtype
            assert param_dino.requires_grad == param_ibot.requires_grad
            assert torch.allclose(param_dino, param_ibot, rtol=1e-3, atol=1e-4)
        assert out.loss.shape == Size([])
        assert out.dino_global_loss.shape == Size([])
        assert out.dino_local_loss.shape == Size([])
        assert out.ibot_loss.shape == Size([])
        assert out.koleo_loss.shape == Size([])
    
    def test_train_step_impl__no_local(self, mocker: MockerFixture) -> None:
        emb_model = EmbeddingModel(wrapped_model=DummyVitModel())
        b = 16

        views = [torch.rand(b, 3, 8, 8) for _ in range(2)]
        batch: Batch = {
            "views": views,
            "filename": [f"img_{i}" for i in range(b)],
        }

        dinov2_args = DINOv2Args()
        dinov2_args.n_local_crops = 0

        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, b)

        out = dinov2.training_step_impl(batch, 0)
        assert out.loss.shape == Size([])
        assert out.dino_global_loss.shape == Size([])
        assert out.dino_local_loss.shape == Size([])
        assert out.ibot_loss.shape == Size([])
        assert out.koleo_loss.shape == Size([])
        assert out.dino_local_loss == torch.tensor(0.0)
    
    def test_train_step_impl__seprate_ibot_head(self, mocker: MockerFixture) -> None:
        emb_model = EmbeddingModel(wrapped_model=DummyVitModel())
        b = 16

        views = [torch.rand(b, 3, 8, 8) for _ in range(2)] + [torch.rand(b, 3, 4, 4) for _ in range(8)]
        batch: Batch = {
            "views": views,
            "filename": [f"img_{i}" for i in range(b)],
        }

        # run DistillationV2
        dinov2_args = DINOv2Args()
        dinov2_args.ibot_separate_head = True
        # To make sure the heads are different
        dinov2_args.bottleneck_dim_ibot = dinov2_args.bottleneck_dim // 2

        dinov2 = setup_dinov2_helper(dinov2_args, mocker, emb_model, b)

        out = dinov2.training_step_impl(batch, 0)

        # check that the ibot and dino heads are NOT the same
        assert dinov2.student_dino_head != dinov2.student_ibot_head
        assert out.loss.shape == Size([])
        assert out.dino_global_loss.shape == Size([])
        assert out.dino_local_loss.shape == Size([])
        assert out.ibot_loss.shape == Size([])
        assert out.koleo_loss.shape == Size([])
    