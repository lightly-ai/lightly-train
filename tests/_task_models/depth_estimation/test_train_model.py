#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import torch

from lightly_train._data.depth_estimation_dataset import (
    DepthEstimationDataArgs,
    SplitArgs,
)
from lightly_train._metrics.depth_estimation.task_metric import (
    DepthEstimationTaskMetricArgs,
)
from lightly_train._task_models.depth_estimation.train_model import (
    DepthEstimationTrain,
    DepthEstimationTrainArgs,
)
from lightly_train._task_models.depth_estimation.transforms import (
    DepthEstimationTrainTransformArgs,
    DepthEstimationValTransformArgs,
)
from lightly_train.types import DepthEstimationBatch

# Test-only V3 config: real ViT-S backbone with a tiny DPT head and a 70px processing
# resolution so depth fine-tuning runs fast on CPU.
_MODEL_NAME = "dinov2/_vittest14-dav3"


def _make_train_model() -> DepthEstimationTrain:
    data_args = DepthEstimationDataArgs(
        train=SplitArgs(
            images="/tmp/train/images", depth="/tmp/train/depth", sky="/tmp/train/sky"
        ),
        val=SplitArgs(
            images="/tmp/val/images", depth="/tmp/val/depth", sky="/tmp/val/sky"
        ),
    )
    train_transform_args = DepthEstimationTrainTransformArgs()
    train_transform_args.resolve_auto(model_init_args={"model_name": _MODEL_NAME})
    val_transform_args = DepthEstimationValTransformArgs()
    val_transform_args.resolve_auto(model_init_args={"model_name": _MODEL_NAME})
    model_args = DepthEstimationTrainArgs()
    model_args.resolve_auto(
        total_steps=10,
        gradient_accumulation_steps=1,
        train_num_batches=4,
        model_name=_MODEL_NAME,
        model_init_args={},
        data_args=data_args,
    )
    return DepthEstimationTrain(
        model_name=_MODEL_NAME,
        model_args=model_args,
        data_args=data_args,
        train_transform_args=train_transform_args,
        val_transform_args=val_transform_args,
        load_weights=False,
        metric_args=DepthEstimationTaskMetricArgs(),
        gradient_accumulation_steps=1,
    )


def _make_batch(*, sky_depth_value: float) -> DepthEstimationBatch:
    """Builds a fixed (B=1) batch whose top half is sky.

    The sky region of the depth target is filled with ``sky_depth_value`` so a test can
    vary only the (garbage) sky depth while everything else stays identical.
    """
    h = w = 70
    image = torch.zeros(1, 3, h, w)
    depth = torch.ones(1, 1, h, w)  # Valid positive depth everywhere.
    sky = torch.zeros(1, 1, h, w)
    sky[:, :, : h // 2, :] = 1.0  # Top half is sky.
    depth[:, :, : h // 2, :] = sky_depth_value
    return {
        "image_path": ["0.png"],
        "image": image,
        "depth": depth,
        "sky": sky,
    }


class TestDepthEstimationTrain:
    def test__step__sky_depth_excluded_from_loss(self) -> None:
        # The depth in sky regions is garbage, so it must not affect the depth losses.
        # Two batches that differ only in the sky-region depth value must produce the
        # same loss. The model is in eval mode with fixed weights, so the forward pass
        # is deterministic.
        train_model = _make_train_model()
        train_model.eval()
        metrics = train_model.val_metrics

        with torch.no_grad():
            result_a = train_model._step(
                batch=_make_batch(sky_depth_value=1e3),
                metrics=metrics,
                compute_metrics=False,
            )
            result_b = train_model._step(
                batch=_make_batch(sky_depth_value=1e-3),
                metrics=metrics,
                compute_metrics=False,
            )

        assert torch.equal(result_a.loss, result_b.loss)

    def test__step__non_sky_depth_changes_loss(self) -> None:
        # Sanity check that the loss is not trivially constant: changing the depth in the
        # valid (non-sky) region must change the loss.
        train_model = _make_train_model()
        train_model.eval()
        metrics = train_model.val_metrics

        batch = _make_batch(sky_depth_value=1.0)
        batch_changed = _make_batch(sky_depth_value=1.0)
        # Perturb the non-sky (bottom half) depth, which is included in the loss.
        assert isinstance(batch_changed["depth"], torch.Tensor)
        batch_changed["depth"][:, :, 35:, :] = 5.0

        with torch.no_grad():
            result = train_model._step(
                batch=batch, metrics=metrics, compute_metrics=False
            )
            result_changed = train_model._step(
                batch=batch_changed, metrics=metrics, compute_metrics=False
            )

        assert not torch.equal(result.loss, result_changed.loss)
