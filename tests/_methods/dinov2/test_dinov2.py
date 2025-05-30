from lightly_train._scaling import ScalingInfo
import pytest

import torch
from torch import Size


from lightly_train._methods.dinov2.dinov2 import DINOv2, DINOv2Args, DINOv2AdamWViTSBArgs
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train.types import Batch

from ...helpers import DummyVitModel


#TODO test the utils

#TODO: test the method, test the args
class TestDINOv2:
    def test_train_step_impl__default(self) -> None:
        emb_model = EmbeddingModel(wrapped_model=DummyVitModel())
        b = 16

        views = [torch.rand(b, 3, 8, 8) for _ in range(8)]
        batch: Batch = {
            "views": views,
        }

        # run DistillationV2
        scaling_info = ScalingInfo(dataset_size=1000, epochs=100)
        dinov2_args = DINOv2Args()

        optimizer_args = DINOv2AdamWViTSBArgs()
        dinov2_args.resolve_auto(
            scaling_info=scaling_info,
            optimizer_args=optimizer_args,
            model=emb_model.wrapped_model.get_model(),
        )

        dinov2 = DINOv2(
            method_args=dinov2_args,
            optimizer_args=optimizer_args,
            embedding_model=emb_model,
            global_batch_size=b,
        )

        out = dinov2.training_step_impl(batch, 0)
        assert out.loss.shape == Size([])
        assert out.dino_global_loss == Size([])
        assert out.dino_local_loss == Size([])
        assert out.ibot_loss == Size([])
        assert out.koleo_loss == Size([])
    
    def test_train_step_impl__no_local(self) -> None:
        emb_model = EmbeddingModel(wrapped_model=DummyVitModel())
        b = 16

        views = [torch.rand(b, 3, 8, 8) for _ in range(2)]
        batch: Batch = {
            "views": views,
        }


        scaling_info = ScalingInfo(dataset_size=1000, epochs=100)
        dinov2_args = DINOv2Args()
        dinov2_args.n_local_crops = 0

        optimizer_args = DINOv2AdamWViTSBArgs()
        dinov2_args.resolve_auto(
            scaling_info=scaling_info,
            optimizer_args=optimizer_args,
            model=emb_model.wrapped_model.get_model(),
        )

        dinov2 = DINOv2(
            method_args=dinov2_args,
            optimizer_args=optimizer_args,
            embedding_model=emb_model,
            global_batch_size=b,
        )

        out = dinov2.training_step_impl(batch, 0)
        assert out.loss.shape == Size([])
        assert out.dino_global_loss == Size([])
        assert out.dino_local_loss == Size([])
        assert out.ibot_loss == Size([])
        assert out.koleo_loss == Size([])
        assert out.dino_local_loss == torch.tensor(0.0)
        