#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""DenseCL

- [0]: 2021, DenseCL: https://arxiv.org/abs/2011.09157
"""

from __future__ import annotations

import copy
from typing import Literal

import torch
from lightly.loss import NTXentLoss
from lightly.models import utils
from lightly.models.modules.heads import DenseCLProjectionHead
from lightly.models.modules.memory_bank import MemoryBankModule
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
)
from lightly.utils.scheduler import cosine_schedule
from torch import Tensor
from torch.nn import AdaptiveAvgPool2d, Module
from torch.nn import functional as F

from lightly_train import _scaling
from lightly_train._configs.validate import no_auto
from lightly_train._methods.densecl.densecl_loss import DenseCLLoss
from lightly_train._methods.densecl.densecl_transform import (
    DenseCLTransform,
)
from lightly_train._methods.method import Method, TrainingStepResult
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.sgd_args import SGDArgs
from lightly_train._optim.trainable_modules import TrainableModules
from lightly_train._scaling import ScalingInfo
from lightly_train._transforms.transform import (
    MethodTransform,
)
from lightly_train.types import Batch


class DenseCLSGDArgs(SGDArgs):
    # Paper uses 0.3 for COCO with ResNet50 and 0.03 for ImageNet with ResNet50.
    # COCO with YOLO works well with values around 0.1.
    lr: float = 0.1


class DenseCLArgs(MethodArgs):
    # Default values for ImageNet1k pretraining from paper.

    # Projection head
    hidden_dim: int = 2048
    output_dim: int = 128

    # Loss
    lambda_: float = 0.5
    temperature: float = 0.2
    memory_bank_size: int | Literal["auto"] = "auto"
    gather_distributed: bool = True

    # Momentum
    momentum_start: float = 0.999
    momentum_end: float = 0.999

    def resolve_auto(
        self,
        scaling_info: ScalingInfo,
        optimizer_args: OptimizerArgs,
        wrapped_model: ModelWrapper,
    ) -> None:
        if self.memory_bank_size == "auto":
            # Reduce memory bank size for smaller datasets, otherwise training is
            # unstable.
            self.memory_bank_size = _scaling.get_bucket_value(
                input=scaling_info.dataset_size,
                buckets=[
                    # (dataset_size, memory_bank_size)
                    # Memory bank size is roughly 50% of the minimal dataset size and
                    # 25% of the maximal dataset size for the given bucket. For example,
                    # a bucket with 100-250 images has a memory bank size of 64.
                    # Note that the DenseCL paper uses a memory bank size of 65536 for
                    # COCO (118k images) and ImageNet (1.3M images).
                    (50, 16),
                    (100, 32),
                    (250, 64),
                    (500, 128),
                    (1_000, 256),
                    (2_000, 512),
                    (4_000, 1024),
                    (10_000, 2048),
                    (20_000, 4096),
                    (40_000, 8192),
                    (60_000, 16384),
                    (200_000, 32768),
                    (float("inf"), 65536),
                ],
            )


class DenseCLEncoder(Module):
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self.local_projection_head = DenseCLProjectionHead(
            input_dim=embedding_model.embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.global_projection_head = DenseCLProjectionHead(
            input_dim=embedding_model.embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.pool = AdaptiveAvgPool2d((1, 1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # B = batch size, C = number of channels, H = image height, W = image width, D = output_dim
        # (B, C, H, W)
        features = self.embedding_model(x, pool=False)
        # (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        global_proj = self.pool(features).flatten(start_dim=1)
        # (B, C) -> (B, D)
        global_proj = self.global_projection_head(global_proj)
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        features = features.flatten(start_dim=2).permute(0, 2, 1)
        # (B, H*W, C) -> (B, H*W, D)
        local_proj = self.local_projection_head(features)
        # (B, H*W, D) -> (B, D)
        local_proj_pooled = local_proj.mean(dim=1)
        # Return: (B, H*W, C), (B, D), (B, H*W, D), (B, D)
        return features, global_proj, local_proj, local_proj_pooled


class DenseCL(Method):
    """DenseCL based on MoCo v2."""

    def __init__(
        self,
        method_args: DenseCLArgs,
        optimizer_args: OptimizerArgs,
        embedding_model: EmbeddingModel,
        global_batch_size: int,
    ):
        super().__init__(
            method_args=method_args,
            optimizer_args=optimizer_args,
            embedding_model=embedding_model,
            global_batch_size=global_batch_size,
        )
        self.method_args = method_args
        self.query_encoder = DenseCLEncoder(
            embedding_model=embedding_model,
            hidden_dim=method_args.hidden_dim,
            output_dim=method_args.output_dim,
        )
        self.key_encoder = copy.deepcopy(self.query_encoder)
        self.memory_bank = MemoryBankModule(
            size=no_auto(method_args.memory_bank_size),
            gather_distributed=method_args.gather_distributed,
        )
        self.local_criterion = DenseCLLoss(temperature=method_args.temperature)
        self.global_criterion = NTXentLoss(
            temperature=method_args.temperature,
            memory_bank_size=(
                no_auto(method_args.memory_bank_size),
                method_args.output_dim,
            ),
            gather_distributed=method_args.gather_distributed,
        )

    def training_step_impl(self, batch: Batch, batch_idx: int) -> TrainingStepResult:
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=self.method_args.momentum_start,
            end_value=self.method_args.momentum_end,
        )
        utils.update_momentum(
            model=self.query_encoder, model_ema=self.key_encoder, m=momentum
        )
        views = batch["views"]
        query_features, query_global, query_local, _ = self.query_encoder(views[0])
        query_features = F.normalize(query_features, dim=-1)
        query_global = F.normalize(query_global, dim=-1)
        query_local = F.normalize(query_local, dim=-1)
        with torch.no_grad():
            distributed = self.trainer.num_devices > 1
            key_images, shuffle = batch_shuffle(batch=views[1], distributed=distributed)
            key_features, key_global, key_local, key_local_pooled = self.key_encoder(
                key_images
            )
            key_features = batch_unshuffle(
                F.normalize(key_features, dim=-1).contiguous(),
                shuffle=shuffle,
                distributed=distributed,
            )
            key_global = batch_unshuffle(
                F.normalize(key_global, dim=-1).contiguous(),
                shuffle=shuffle,
                distributed=distributed,
            )
            key_local = batch_unshuffle(
                F.normalize(key_local, dim=-1).contiguous(),
                shuffle=shuffle,
                distributed=distributed,
            )
            key_local_pooled = batch_unshuffle(
                F.normalize(key_local_pooled, dim=-1).contiguous(),
                shuffle=shuffle,
                distributed=distributed,
            )
            key_local = utils.select_most_similar(
                query_features.detach(), key_features, key_local
            )

        query_local = query_local.flatten(end_dim=1)
        key_local = key_local.flatten(end_dim=1)

        _, negatives = self.memory_bank(output=key_local_pooled, update=True)
        local_loss = self.local_criterion(
            out0=query_local,
            out1=key_local,
            negatives=negatives,
        )
        global_loss = self.global_criterion(out0=query_global, out1=key_global)
        lambda_ = self.method_args.lambda_
        loss = (1 - lambda_) * global_loss + lambda_ * local_loss
        return TrainingStepResult(loss=loss)

    @staticmethod
    def method_args_cls() -> type[DenseCLArgs]:
        return DenseCLArgs

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[OptimizerArgs]:
        classes: dict[OptimizerType | Literal["auto"], type[OptimizerArgs]] = {
            "auto": DenseCLSGDArgs,
            OptimizerType.SGD: DenseCLSGDArgs,
        }
        return classes.get(optim_type, Method.optimizer_args_cls(optim_type=optim_type))

    def trainable_modules(self) -> TrainableModules:
        return TrainableModules(modules=[self.query_encoder])

    @staticmethod
    def transform_cls() -> type[MethodTransform]:
        return DenseCLTransform
