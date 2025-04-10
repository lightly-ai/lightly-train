#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
from typing import Literal

import lightly.models.utils as lm_utils
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as torchvision_functional
from lightly.loss import DetConBLoss, DetConSLoss
from lightly.models.modules.heads import (
    BYOLPredictionHead,
    BYOLProjectionHead,
    SimCLRProjectionHead,
)
from torch import Tensor
from torch.distributions import Categorical
from torchvision.transforms import InterpolationMode

from lightly_train._methods.detcon.detcon_transform import (
    DetConBTransform,
    DetConSTransform,
)
from lightly_train._methods.method import Method, TrainingStepResult
from lightly_train._methods.method_args import MethodArgs
from lightly_train._models.embedding_model import EmbeddingModel
from lightly_train._optim.optimizer_args import OptimizerArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train._optim.sgd_args import SGDArgs
from lightly_train._optim.trainable_modules import TrainableModules
from lightly_train._transforms.transform import (
    MethodTransform,
)
from lightly_train.types import Batch


class DetConSArgs(MethodArgs):
    """Args for DetConS method."""

    hidden_dim: int = 2048
    output_dim: int = 128
    num_layers: int = 2
    batch_norm: bool = True
    temperature: float = 0.1
    num_sampled_cls: int = 16
    num_total_cls: int = 256


class DetConBArgs(MethodArgs):
    """Args for DetConB method."""

    proj_hidden_dim: int = 2048
    proj_output_dim: int = 256
    pred_hidden_dim: int = 2048
    pred_output_dim: int = 256
    temperature: float = 0.1
    num_sampled_cls: int = 16
    num_total_cls: int = 256


class DetConSSGDArgs(SGDArgs):
    """Parameters for ImageNet pretraining with batch size 4096. Usually optimized with
    LARS and linear scaling of learning rate by batch size.
    """

    lr: float = 0.3
    weight_decay: float = 0.0001


class DetConBSGDArgs(SGDArgs):
    """Parameters for ImageNet pretraining with batch size 4096 (300 epochs).
    For 1000 epoch pretraining, values are lr=0.2 and weight_decay=1.5e-6. Usually
    optimized with LARS and linear scaling of learning rate by batch size.
    """

    lr: float = 0.3
    weight_decay: float = 0.0001


def _subsample_mask_indices(
    masks: Tensor,
    n_total_cls: int,
    n_sampled_cls: int,
    class_weighted_sampling: bool = False,
) -> Tensor:
    """From a variable number of classes/masks appearing in each image of the batch,
    subsample a fixed number of masks, with repetition.

    Args:
        masks: Integer tensor of shape (B, H, W) containing the masks.
        n_total_cls: Total number of classes in the dataset.
        n_sampled_cls: Number of masks to subsample.
        class_weighted_sampling: Whether to sample classes with probability proportional
            to the number of pixels in the mask.

    Returns:
        An integer tensor of shape (B, n_sampled_cls) containing the subsampled mask indices.
    """
    b, h, w = masks.shape
    masks = masks.view(b, h * w)  # (b, h*w)
    bin_masks = (  # TODO: make more memory efficient
        F.one_hot(masks, num_classes=n_total_cls).permute(0, 2, 1).to(torch.bool)
    )  # (b, n_total_cls, h*w)

    if class_weighted_sampling:
        mask_unnormed_prob = bin_masks.sum(dim=-1).to(torch.float32)  # (b, n_total_cls)
    else:
        mask_unnormed_prob = (
            torch.greater(bin_masks.sum(dim=-1), 1e-3).to(torch.float32) + 0.00000000001
        )

    # create batch of Categorical distributions, one for each image
    # probabilities will be normalized in the Categorical distribution
    indices: Tensor = (
        # type ignore because torch distributions are not typed
        Categorical(probs=mask_unnormed_prob).sample([n_sampled_cls]).permute(1, 0)  # type: ignore[no-untyped-call]
    )  # (b, n_sampled_cls)
    return indices


def _subsample_pooled_features(x: Tensor, indices: Tensor) -> Tensor:
    """From masked-pooled features, only keep the features corresponding to the sampled
    masks. Don't use this function with double batches, i.e. when concatenating views!

    Args:
        x: Float tensor of shape (B, n_total_cls, C) containing the pooled features of
            all appearing classes in the dataset, i.e. dim=1 contains all classes from 0
            to max_class.
        indices: Integer tensor of shape (B, num_sampled_cls) containing the subsampled mask
            indices.

    Returns:
        Float tensor of shape (B, num_sampled_cls, C) containing the pooled
            features of the subsampled masks.
    """
    return torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


class DetConS(Method):
    def __init__(
        self,
        method_args: DetConSArgs,
        optimizer_args: OptimizerArgs,
        embedding_model: EmbeddingModel,
        global_batch_size: int,
    ) -> None:
        super().__init__(
            method_args=method_args,
            optimizer_args=optimizer_args,
            embedding_model=embedding_model,
            global_batch_size=global_batch_size,
        )
        self.method_args = method_args
        self.embedding_model = embedding_model
        self.projection_head = SimCLRProjectionHead(
            input_dim=self.embedding_model.embed_dim,
            hidden_dim=self.method_args.hidden_dim,
            output_dim=self.method_args.output_dim,
            num_layers=self.method_args.num_layers,
            batch_norm=self.method_args.batch_norm,
        )
        self.criterion = DetConSLoss(
            temperature=self.method_args.temperature,
            # TODO (Lionel, 15.01.25): use only dist.is_available() if lightly dependency is updated
            gather_distributed=dist.is_available() and dist.is_initialized(),
        )

    def training_step_impl(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> TrainingStepResult:
        views: list[Tensor] = batch["views"]
        masks_list: list[Tensor] = batch["masks"]
        x = self.embedding_model(
            torch.cat(views, dim=0), pool=False
        )  # (2*B, C, l_h, l_w)

        # some constants
        _, c, l_h, l_w = x.shape
        b = x.shape[0] // 2
        n_total_cls = self.method_args.num_total_cls
        n_sampled_cls = self.method_args.num_sampled_cls
        proj_output_dim = self.method_args.output_dim

        # resize masks to match the feature map size
        masks: Tensor = torchvision_functional.resize(
            torch.cat(masks_list, dim=0),
            (l_h, l_w),
            interpolation=InterpolationMode.NEAREST,
        ).to(torch.int64)  # (2*B, l_h, l_w)

        # choose only a subset of the masks/classes
        indices = _subsample_mask_indices(
            masks=masks,
            n_total_cls=n_total_cls,
            n_sampled_cls=n_sampled_cls,
        )  # int (2*B, n_sampled_cls)

        # pool the features over the sampled masks
        x_pooled = lm_utils.pool_masked(
            source=x, mask=masks, num_cls=n_total_cls, reduce="mean"
        )  # float (2*B, C, n_total_cls)
        x_pooled = x_pooled.permute(0, 2, 1)  # (2*B, n_total_cls, C)

        indices0, indices1 = indices.chunk(2, dim=0)  # both (B, n_sampled_cls)
        x_pooled0, x_pooled1 = x_pooled.chunk(2, dim=0)  # both (B, n_total_cls, C)
        x_pooled0 = _subsample_pooled_features(
            x_pooled0, indices0
        )  # (B, n_sampled_cls, C)
        x_pooled1 = _subsample_pooled_features(
            x_pooled1, indices1
        )  # (B, n_sampled_cls, C)
        indices = torch.cat([indices0, indices1], dim=0)  # (2*B, n_sampled_cls)
        x_pooled = torch.cat([x_pooled0, x_pooled1], dim=0)  # (2*B, n_sampled_cls, C)

        # BatchNorm1d in projection_head requires folding of class dimension into batch
        x_pooled = self.projection_head(x_pooled.view(2 * b * n_sampled_cls, c)).view(
            2 * b, n_sampled_cls, proj_output_dim
        )  # (2*B, n_sampled_cls, projection_output_dim)

        view0, view1 = x_pooled.chunk(
            2, dim=0
        )  # both (B, n_sampled_cls, projection_output_dim)
        mask0, mask1 = indices.chunk(2, dim=0)  # both (B, n_sampled_cls)
        loss = self.criterion(
            view0=view0,
            view1=view1,
            mask_view0=mask0,
            mask_view1=mask1,
        )
        return TrainingStepResult(loss=loss)

    @staticmethod
    def method_args_cls() -> type[DetConSArgs]:
        return DetConSArgs

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[OptimizerArgs]:
        classes: dict[OptimizerType | Literal["auto"], type[OptimizerArgs]] = {
            "auto": DetConSSGDArgs,
            OptimizerType.SGD: DetConSSGDArgs,
        }
        return classes.get(optim_type, Method.optimizer_args_cls(optim_type=optim_type))

    def trainable_modules(self) -> TrainableModules:
        return TrainableModules(modules=[self.embedding_model, self.projection_head])

    @staticmethod
    def transform_cls() -> type[MethodTransform]:
        return DetConSTransform


class DetConB(Method):
    def __init__(
        self,
        method_args: DetConBArgs,
        optimizer_args: OptimizerArgs,
        embedding_model: EmbeddingModel,
        global_batch_size: int,
    ) -> None:
        super().__init__(
            method_args=method_args,
            optimizer_args=optimizer_args,
            embedding_model=embedding_model,
            global_batch_size=global_batch_size,
        )
        self.method_args = method_args

        # prediction branch
        self.embedding_model = embedding_model
        self.projection_head = BYOLProjectionHead(
            input_dim=self.embedding_model.embed_dim,
            hidden_dim=self.method_args.proj_hidden_dim,
            output_dim=self.method_args.proj_output_dim,
        )
        self.prediction_head = BYOLPredictionHead(
            input_dim=self.method_args.proj_output_dim,
            hidden_dim=self.method_args.pred_hidden_dim,
            output_dim=self.method_args.pred_output_dim,
        )

        # target branch
        self.embedding_model_momentum = copy.deepcopy(self.embedding_model)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        self.criterion = DetConBLoss(
            temperature=self.method_args.temperature,
            # TODO (Lionel, 15.01.25): use only dist.is_available() if lightly dependency is updated
            gather_distributed=dist.is_available() and dist.is_initialized(),
        )

    def training_step_impl(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> TrainingStepResult:
        views: list[Tensor] = batch["views"]
        masks_list: list[Tensor] = batch["masks"]
        prediction = self.embedding_model(
            torch.cat(views, dim=0), pool=False
        )  # (2*B, C, l_h, l_w)
        with torch.no_grad():
            target = self.embedding_model_momentum(
                torch.cat(views, dim=0), pool=False
            )  # (2*B, C, l_h, l_w)

        # some constants
        _, c, l_h, l_w = prediction.shape
        b = prediction.shape[0] // 2
        n_total_cls = self.method_args.num_total_cls
        n_sampled_cls = self.method_args.num_sampled_cls
        proj_output_dim = self.method_args.proj_output_dim
        pred_output_dim = self.method_args.pred_output_dim
        target_output_dim = proj_output_dim

        # resize masks to match the feature map size
        masks: Tensor = torchvision_functional.resize(
            torch.cat(masks_list, dim=0),
            (l_h, l_w),
            interpolation=InterpolationMode.NEAREST,
        ).to(torch.int64)

        # choose only a subset of the masks/classes
        indices = _subsample_mask_indices(
            masks=masks,
            n_total_cls=n_total_cls,
            n_sampled_cls=n_sampled_cls,
        )

        # pool the features over the sampled masks
        prediction_pooled = lm_utils.pool_masked(
            source=prediction, mask=masks, num_cls=n_total_cls, reduce="mean"
        )
        target_pooled = lm_utils.pool_masked(
            source=target, mask=masks, num_cls=n_total_cls, reduce="mean"
        )
        prediction_pooled = prediction_pooled.permute(0, 2, 1)  # (2*B, n_total_cls, C)
        target_pooled = target_pooled.permute(0, 2, 1)  # (2*B, n_total_cls, C)

        # subsampling requires splitting into proper batch size
        indices0, indices1 = indices.chunk(2, dim=0)  # both (B, n_sampled_cls)
        prediction_pooled0, prediction_pooled1 = prediction_pooled.chunk(
            2, dim=0
        )  # both (B, n_total_cls, C)
        target_pooled0, target_pooled1 = target_pooled.chunk(
            2, dim=0
        )  # both (B, n_total_cls, C)
        prediction_pooled0 = _subsample_pooled_features(
            prediction_pooled0, indices0
        )  # (B, n_sampled_cls, C)
        prediction_pooled1 = _subsample_pooled_features(
            prediction_pooled1, indices1
        )  # (B, n_sampled_cls, C)
        target_pooled0 = _subsample_pooled_features(
            target_pooled0, indices0
        )  # (B, n_sampled_cls, C)
        target_pooled1 = _subsample_pooled_features(
            target_pooled1, indices1
        )  # (B, n_sampled_cls, C)
        indices = torch.cat([indices0, indices1], dim=0)  # (2*B, n_sampled_cls)
        prediction_pooled = torch.cat(
            [prediction_pooled0, prediction_pooled1], dim=0
        )  # (2*B, n_sampled_cls, C)
        target_pooled = torch.cat(
            [target_pooled0, target_pooled1], dim=0
        )  # (2*B, n_sampled_cls, C)

        # send through projection / prediction heads
        prediction_pooled = self.projection_head(
            prediction_pooled.view(2 * b * n_sampled_cls, c)
        )  # (2 * B * n_sampled_cls, proj_output_dim)
        prediction_pooled = self.prediction_head(prediction_pooled).view(
            2 * b, n_sampled_cls, pred_output_dim
        )  # (2*B, n_sampled_cls, pred_output_dim)

        with torch.no_grad():
            target_pooled = self.projection_head_momentum(
                target_pooled.view(2 * b * n_sampled_cls, c)
            ).view(
                2 * b, n_sampled_cls, target_output_dim
            )  # (2*B, n_sampled_cls, target_output_dim)

        pred0, pred1 = prediction_pooled.chunk(
            2, dim=0
        )  # both (B, n_sampled_cls, pred_output_dim)
        target0, target1 = target_pooled.chunk(
            2, dim=0
        )  # both (B, n_sampled_cls, target_output_dim)
        mask0, mask1 = indices.chunk(2, dim=0)  # both (B, n_sampled_cls)

        loss0 = self.criterion(
            pred_view0=pred0,
            pred_view1=pred1,
            target_view0=target0,
            target_view1=target1,
            mask_view0=mask0,
            mask_view1=mask1,
        )
        loss1 = self.criterion(
            pred_view0=pred1,
            pred_view1=pred0,
            target_view0=target1,
            target_view1=target0,
            mask_view0=mask1,
            mask_view1=mask0,
        )
        loss = (loss0 + loss1) / 2
        return TrainingStepResult(loss=loss)

    @staticmethod
    def method_args_cls() -> type[DetConBArgs]:
        return DetConBArgs

    @staticmethod
    def optimizer_args_cls(
        optim_type: OptimizerType | Literal["auto"],
    ) -> type[OptimizerArgs]:
        classes: dict[OptimizerType | Literal["auto"], type[OptimizerArgs]] = {
            "auto": DetConBSGDArgs,
            OptimizerType.SGD: DetConBSGDArgs,
        }
        return classes.get(optim_type, Method.optimizer_args_cls(optim_type=optim_type))

    def trainable_modules(self) -> TrainableModules:
        return TrainableModules(
            modules=[self.embedding_model, self.projection_head, self.prediction_head]
        )

    @staticmethod
    def transform_cls() -> type[MethodTransform]:
        return DetConBTransform
