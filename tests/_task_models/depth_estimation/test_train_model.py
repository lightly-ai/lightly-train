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

# Test-only V3 configs: real ViT-S backbone with a tiny DPT head and a 70px processing
# resolution so depth fine-tuning runs fast on CPU. The metric variant only differs by
# `scale_mode="focal"`, which switches on the scale-aware loss branch.
_MODEL_NAME = "dinov2/_vittest14-dav3"
_METRIC_MODEL_NAME = "dinov2/_vittest14-dav3-metric"


def _make_train_model(
    model_name: str = _MODEL_NAME,
    *,
    use_ema: bool = False,
    abs_l1_loss_weight: float = 0.0,
) -> DepthEstimationTrain:
    data_args = DepthEstimationDataArgs(
        train=SplitArgs(
            images="/tmp/train/images", depth="/tmp/train/depth", sky="/tmp/train/sky"
        ),
        val=SplitArgs(
            images="/tmp/val/images", depth="/tmp/val/depth", sky="/tmp/val/sky"
        ),
    )
    train_transform_args = DepthEstimationTrainTransformArgs()
    train_transform_args.resolve_auto(model_init_args={"model_name": model_name})
    val_transform_args = DepthEstimationValTransformArgs()
    val_transform_args.resolve_auto(model_init_args={"model_name": model_name})
    model_args = DepthEstimationTrainArgs(
        use_ema=use_ema, abs_l1_loss_weight=abs_l1_loss_weight
    )
    model_args.resolve_auto(
        total_steps=10,
        gradient_accumulation_steps=1,
        train_num_batches=4,
        model_name=model_name,
        model_init_args={},
        data_args=data_args,
    )
    return DepthEstimationTrain(
        model_name=model_name,
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
                model=train_model.model,
                batch=_make_batch(sky_depth_value=1e3),
                metrics=metrics,
                compute_metrics=False,
            )
            result_b = train_model._step(
                model=train_model.model,
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
                model=train_model.model,
                batch=batch,
                metrics=metrics,
                compute_metrics=False,
            )
            result_changed = train_model._step(
                model=train_model.model,
                batch=batch_changed,
                metrics=metrics,
                compute_metrics=False,
            )

        assert not torch.equal(result.loss, result_changed.loss)


class TestDepthEstimationTrainMetric:
    def test___init___relative_is_scale_invariant(self) -> None:
        # The relative student keeps the scale-invariant SILog (default lambd) and the
        # scale-and-shift-aligned validation metric.
        train_model = _make_train_model(_MODEL_NAME)

        assert train_model._is_metric is False
        assert train_model.silog_criterion.lambd == train_model.model_args.silog_lambda
        assert train_model.val_metrics.metrics.align is True

    def test___init___metric_is_scale_aware(self) -> None:
        # The metric student switches SILog to scale-aware (lambd=0, i.e. log-space L2)
        # and turns off metric alignment so the metrics reflect true metric accuracy.
        train_model = _make_train_model(_METRIC_MODEL_NAME)

        assert train_model._is_metric is True
        assert train_model.silog_criterion.lambd == 0.0
        assert train_model.val_metrics.metrics.align is False
        assert train_model.train_metrics.metrics.align is False

    def test__step__metric_loss_is_scale_aware(self) -> None:
        # For a metric model, scaling the prediction by a global factor must change the
        # loss (the opposite of the relative, scale-invariant case). The prediction is
        # scaled by scaling the target instead: with fixed eval weights the forward pass
        # is deterministic, so a target scaled by a constant is not free to be absorbed.
        train_model = _make_train_model(_METRIC_MODEL_NAME)
        train_model.eval()
        metrics = train_model.val_metrics

        batch = _make_batch(sky_depth_value=1.0)
        batch_scaled = _make_batch(sky_depth_value=1.0)
        assert isinstance(batch_scaled["depth"], torch.Tensor)
        # Scale only the valid (non-sky) region so the depth loss sees the scale change.
        batch_scaled["depth"][:, :, 35:, :] *= 4.0

        with torch.no_grad():
            result = train_model._step(
                model=train_model.model,
                batch=batch,
                metrics=metrics,
                compute_metrics=False,
            )
            result_scaled = train_model._step(
                model=train_model.model,
                batch=batch_scaled,
                metrics=metrics,
                compute_metrics=False,
            )

        assert torch.isfinite(result.loss)
        assert not torch.equal(result.loss, result_scaled.loss)

    def test__step__abs_l1_term_added_only_for_metric(self) -> None:
        # The relative-L1 term is added to the total loss for a metric model with a
        # positive weight, and its per-pixel value is logged in every case.
        base = _make_train_model(_METRIC_MODEL_NAME, abs_l1_loss_weight=0.0)
        weighted = _make_train_model(_METRIC_MODEL_NAME, abs_l1_loss_weight=1.0)
        # Share weights so the two models produce the same forward pass, isolating the
        # loss-composition difference.
        weighted.model.load_state_dict(base.model.state_dict())
        base.eval()
        weighted.eval()
        batch = _make_batch(sky_depth_value=1.0)
        image = batch["image"]
        depth = batch["depth"]
        sky = batch["sky"]
        assert isinstance(image, Tensor)
        assert isinstance(depth, Tensor)
        assert isinstance(sky, Tensor)

        with torch.no_grad():
            result_base = base._step(
                model=base.model,
                batch=batch,
                metrics=base.val_metrics,
                compute_metrics=False,
            )
            result_weighted = weighted._step(
                model=weighted.model,
                batch=batch,
                metrics=weighted.val_metrics,
                compute_metrics=False,
            )

        # With a positive weight the metric total loss is larger by exactly the term.
        abs_l1 = weighted.abs_l1_criterion(
            weighted._forward(model=weighted.model, images=image)["depth"],
            depth,
            (depth > 0) & (sky < 0.5),
        ).detach()
        assert float(abs_l1) > 0.0
        assert float(result_weighted.loss) == pytest.approx(
            float(result_base.loss) + float(abs_l1), abs=1e-5
        )

    def test__step__abs_l1_term_not_applied_for_relative(self) -> None:
        # For a relative model the AbsRel term is removed entirely: a positive weight does
        # not change the total loss and the logged abs_l1_loss stays 0.
        base = _make_train_model(_MODEL_NAME, abs_l1_loss_weight=0.0)
        weighted = _make_train_model(_MODEL_NAME, abs_l1_loss_weight=1.0)
        # Share weights so the two models produce the same forward pass, isolating the
        # loss-composition difference.
        weighted.model.load_state_dict(base.model.state_dict())
        base.eval()
        weighted.eval()
        batch = _make_batch(sky_depth_value=1.0)
        image = batch["image"]
        depth = batch["depth"]
        sky = batch["sky"]
        assert isinstance(image, Tensor)
        assert isinstance(depth, Tensor)
        assert isinstance(sky, Tensor)

        with torch.no_grad():
            result_base = base._step(
                model=base.model,
                batch=batch,
                metrics=base.val_metrics,
                compute_metrics=False,
            )
            result_weighted = weighted._step(
                model=weighted.model,
                batch=batch,
                metrics=weighted.val_metrics,
                compute_metrics=False,
            )

        # A positive weight leaves the relative total loss unchanged: the term is not added.
        assert float(result_weighted.loss) == pytest.approx(
            float(result_base.loss), abs=1e-5
        )
        # The logged abs_l1_loss is a constant 0 for relative models.
        assert (
            weighted.val_metrics.loss_metrics.compute()["val_loss/abs_l1_loss"] == 0.0
        )


class TestDepthEstimationTrainEMA:
    def test___init___ema_disabled_by_default(self) -> None:
        train_model = _make_train_model()

        assert train_model.ema_model is None
        # With EMA off, get_task_model returns the live model and the export has no
        # ema_model keys.
        assert train_model.get_task_model() is train_model.model
        assert not any(
            k.startswith("ema_model.") for k in train_model.get_export_state_dict()
        )

    def test___init___ema_enabled(self) -> None:
        train_model = _make_train_model(use_ema=True)

        assert train_model.ema_model is not None
        # Inference/export use the EMA model when enabled.
        assert train_model.get_task_model() is train_model.ema_model.model

    def test_on_train_batch_end__updates_ema_toward_live_weights(self) -> None:
        # After a live-weight change, an EMA update must move the shadow toward the live
        # weights (but not all the way, since decay < 1).
        train_model = _make_train_model(use_ema=True)
        assert train_model.ema_model is not None

        # Pick a representative decoder weight and force live/EMA apart.
        (name, live_param) = next(
            (n, p) for n, p in train_model.model.named_parameters()
        )
        with torch.no_grad():
            live_param.add_(1.0)
        ema_param = dict(train_model.ema_model.model.named_parameters())[name]
        before = ema_param.clone()
        assert not torch.allclose(before, live_param)

        train_model.on_train_batch_end()

        # The EMA weight moved toward the (larger) live weight but did not reach it.
        assert torch.all(ema_param >= before)
        assert not torch.allclose(ema_param, live_param)

    def test_get_export_state_dict__holds_ema_weights_under_normal_keys(self) -> None:
        # The export must contain the EMA weights, stored under the live-model `model.`
        # keys, with no `ema_model.` keys — indistinguishable from a non-EMA export.
        train_model = _make_train_model(use_ema=True)
        assert train_model.ema_model is not None

        # Drive live and EMA apart so the two weight sets are distinguishable.
        with torch.no_grad():
            for p in train_model.model.parameters():
                p.add_(1.0)

        export = train_model.get_export_state_dict()

        assert not any(k.startswith("ema_model.") for k in export)
        ema_sd = train_model.ema_model.model.state_dict()
        live_sd = train_model.model.state_dict()
        # A representative weight in the export matches the EMA value, not the live value.
        key = "model.decoder.scratch.output_conv1.weight"
        assert torch.equal(export[key], ema_sd["decoder.scratch.output_conv1.weight"])
        assert not torch.equal(
            export[key], live_sd["decoder.scratch.output_conv1.weight"]
        )

    def test_load_train_state_dict__seeds_ema_from_ema_free_checkpoint(self) -> None:
        # Loading an export/relative checkpoint (which has no ema_model. keys) into an
        # EMA-enabled model must succeed and seed the EMA shadow from the loaded weights.
        # This is the metric-fine-tune-from-a-relative-checkpoint path.
        source = _make_train_model()  # EMA off -> export has no ema_model keys.
        with torch.no_grad():
            for p in source.model.parameters():
                p.add_(0.5)
        export = source.get_export_state_dict()
        assert not any(k.startswith("ema_model.") for k in export)

        target = _make_train_model(use_ema=True)
        target.load_train_state_dict(export, strict=True)

        assert target.ema_model is not None
        # Live model loaded from the checkpoint, and the EMA shadow was seeded to match.
        for name, live_p in target.model.named_parameters():
            ema_p = dict(target.ema_model.model.named_parameters())[name]
            assert torch.equal(live_p, ema_p)
