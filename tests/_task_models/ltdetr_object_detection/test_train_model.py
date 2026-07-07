#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._task_models.ltdetr_object_detection.train_model import (
    DINOv2LTDETRObjectDetectionTrainArgsV2,
)
from lightly_train._task_models.ltdetr_object_detection.transforms import (
    DINOv2LTDETRObjectDetectionTrainTransformArgsV2,
    DINOv2LTDETRObjectDetectionValTransformArgsV2,
)


def test_dinov2_train_args_v2_defaults() -> None:
    args = DINOv2LTDETRObjectDetectionTrainArgsV2()

    assert DINOv2LTDETRObjectDetectionTrainArgsV2.default_batch_size == 16
    assert DINOv2LTDETRObjectDetectionTrainArgsV2.default_steps == 100_000 // 16 * 72
    assert args.patch_size == 14
    assert args.decoder_name == "rtdetrv2"
    assert args.lr == 1e-4
    assert args.backbone_lr_factor == 1e-2
    assert args.scheduler_name == "linear"
    assert args.lr_warmup_steps == 2000
    assert args.scheduler_flat_steps == "auto"
    assert args.scheduler_no_aug_steps == "auto"
    assert args.use_ema_model is True
    assert args.ema_momentum == 0.9999
    assert args.ema_warmup_steps == 2000
    assert args.matcher_weight_dict == {
        "cost_class": 2.0,
        "cost_bbox": 5.0,
        "cost_giou": 2.0,
    }
    assert args.loss_weight_dict == {
        "loss_vfl": 1.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
    }
    assert args.losses == ["vfl", "boxes"]
    assert args.loss_alpha == 0.75
    assert args.loss_gamma == 2.0
    assert args.gradient_clip_val == 0.1
    assert args.weight_decay == 1e-4
    assert args.optimizer_betas == (0.9, 0.999)
    assert args.scheduler_start_factor == 0.01


def test_dinov2_transform_args_v2_defaults() -> None:
    train_args = DINOv2LTDETRObjectDetectionTrainTransformArgsV2()
    val_args = DINOv2LTDETRObjectDetectionValTransformArgsV2()

    assert train_args.image_size == "auto"
    assert train_args.normalize == "auto"
    assert train_args.num_channels == "auto"
    assert train_args.channel_drop is None
    assert train_args.scale_jitter is not None
    assert train_args.scale_jitter.divisible_by is None
    assert train_args.scale_jitter.sizes == [
        (476, 476),
        (504, 504),
        (532, 532),
        (560, 560),
        (588, 588),
        (616, 616),
        *([(644, 644)] * 20),
        (672, 672),
        (700, 700),
        (728, 728),
        (756, 756),
        (784, 784),
        (812, 812),
    ]
    assert train_args.mosaic is not None
    assert train_args.mosaic.output_size == 320
    assert train_args.mixup is not None
    assert train_args.mixup.prob == 0.5
    assert train_args.copyblend is not None
    assert train_args.copyblend.prob == 0.5

    assert val_args.image_size == "auto"
    assert val_args.normalize == "auto"
    assert val_args.num_channels == "auto"
    assert val_args.photometric_distort is None
    assert val_args.random_zoom_out is None
    assert val_args.random_iou_crop is None
    assert val_args.random_flip is None
    assert val_args.scale_jitter is None
    assert val_args.mosaic is None
    assert val_args.mixup is None
    assert val_args.copyblend is None
