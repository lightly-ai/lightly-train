#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import pytest
from torch import nn
from torch.optim.lr_scheduler import LinearLR

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._metrics.detection.task_metric import ObjectDetectionTaskMetricArgs
from lightly_train._models.dinov3.dinov3_package import DINOV3_PACKAGE
from lightly_train._models.dinov3.dinov3_src.hub import backbones
from lightly_train._task_models.dinov3_ltdetr_object_detection.task_model import (
    DINOv3LTDETRObjectDetection,
    _RTDETRTransformerv2Config,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.train_model import (
    DINOv3LTDETRObjectDetectionTrain,
    DINOv3LTDETRObjectDetectionTrainArgs,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.transforms import (
    DINOv3LTDETRObjectDetectionTrainTransformArgs,
    DINOv3LTDETRObjectDetectionValTransformArgs,
)
from lightly_train._task_models.object_detection_components.flat_cosine import (
    FlatCosineLRScheduler,
)


@pytest.mark.parametrize("use_ema_model", [True, False])
def test_load_train_state_dict__from_exported(use_ema_model: bool) -> None:
    model_args = DINOv3LTDETRObjectDetectionTrainArgs(use_ema_model=use_ema_model)
    train_model = _create_train_model(model_args)
    task_model = train_model.model
    state_dict = train_model.get_export_state_dict()
    task_model.load_train_state_dict(state_dict)


def test_load_train_state_dict__no_ema_weights() -> None:
    model_args = DINOv3LTDETRObjectDetectionTrainArgs(use_ema_model=True)
    train_model = _create_train_model(model_args)
    task_model = train_model.model
    state_dict = train_model.state_dict()
    # Drop all EMA weights from the state dict. This is for backwards compatibility
    # with older checkpoints. The model should still be able to load the weights by
    # copying the non-EMA weights to the EMA model.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ema_model.")}
    task_model.load_train_state_dict(state_dict)


def _is_module_frozen(m: nn.Module) -> bool:
    return all(not param.requires_grad for param in m.parameters())


@pytest.mark.parametrize("should_freeze", [True, False])
def test_freeze_backbone_on_set_train_mode(should_freeze: bool) -> None:
    model_args = DINOv3LTDETRObjectDetectionTrainArgs(
        use_ema_model=True,
        backbone_freeze=should_freeze,
    )
    train_model = _create_train_model(model_args)
    task_model_backbone = train_model.model.backbone
    assert isinstance(task_model_backbone, nn.Module), "Backbone should be a nn.Module"

    train_model.set_train_mode()

    assert _is_module_frozen(task_model_backbone) == should_freeze, (
        f"Backbone should be frozen: {should_freeze}, but got frozen={_is_module_frozen(task_model_backbone)}"
    )
    assert not task_model_backbone.training == should_freeze, (
        "Backbone should be in eval mode after set_train_mode()"
    )


def test_resolve_auto__uses_vit_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    model_args = DINOv3LTDETRObjectDetectionTrainArgs()
    train_model, calls = _create_train_model_with_capture(
        model_args,
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        monkeypatch=monkeypatch,
    )

    assert model_args.patch_size == 16
    assert calls[0]["model_args"]["patch_size"] == 16
    assert train_model.model.backbone.patch_size == 16


def test_resolve_auto__uses_model_init_args_patch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_args = DINOv3LTDETRObjectDetectionTrainArgs()
    train_model, calls = _create_train_model_with_capture(
        model_args,
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        model_init_args={"patch_size": 14},
        monkeypatch=monkeypatch,
    )

    assert model_args.patch_size == 14
    assert calls[0]["model_args"]["patch_size"] == 14
    assert train_model.model.backbone.patch_size == 14


def test_resolve_auto__uses_explicit_patch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_args = DINOv3LTDETRObjectDetectionTrainArgs(patch_size=14)
    train_model, calls = _create_train_model_with_capture(
        model_args,
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        monkeypatch=monkeypatch,
    )

    assert model_args.patch_size == 14
    assert calls[0]["model_args"]["patch_size"] == 14
    assert train_model.model.backbone.patch_size == 14


def test_task_model_init_args_roundtrip_preserves_patch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _create_task_model(monkeypatch=monkeypatch, patch_size=14)

    assert model.init_args["patch_size"] == 14

    roundtrip_model = DINOv3LTDETRObjectDetection(
        **model.init_args,
        load_weights=False,
    )
    assert roundtrip_model.init_args == model.init_args
    assert roundtrip_model.init_args["patch_size"] == 14


def test_resolve_auto__uses_explicit_patch_size_for_convnext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_args = DINOv3LTDETRObjectDetectionTrainArgs(patch_size=14)
    train_model, calls = _create_train_model_with_capture(
        model_args,
        model_name="dinov3/convnext-small-ltdetr",
        monkeypatch=monkeypatch,
    )

    assert model_args.patch_size == 14
    assert calls[0]["model_args"]["patch_size"] == 14
    assert train_model.model.backbone.patch_size == 14


@pytest.mark.parametrize("patch_size", [14, 16, 64])
def test_train_transform_args__resolve_auto__scale_jitter_divisible_by_patch_size(
    patch_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_model = _create_task_model(
        monkeypatch=monkeypatch,
        patch_size=patch_size,
    )
    train_transform_args = DINOv3LTDETRObjectDetectionTrainTransformArgs()
    train_transform_args.resolve_auto(task_model.init_args)

    assert train_transform_args.scale_jitter.divisible_by == patch_size * 2


def _create_train_model(
    train_model_args: DINOv3LTDETRObjectDetectionTrainArgs,
    *,
    model_name: str = "dinov3/vitt16-notpretrained-ltdetr",
    model_init_args: dict[str, Any] | None = None,
) -> DINOv3LTDETRObjectDetectionTrain:
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
    )
    train_model_args.resolve_auto(
        total_steps=1000,
        model_name=model_name,
        model_init_args={} if model_init_args is None else model_init_args,
        data_args=data_args,
    )
    train_transform_args = DINOv3LTDETRObjectDetectionTrainTransformArgs()
    train_transform_args.resolve_auto(model_init_args={})
    val_transform_args = DINOv3LTDETRObjectDetectionValTransformArgs()
    val_transform_args.resolve_auto(model_init_args={})

    train_model = DINOv3LTDETRObjectDetectionTrain(
        model_name=model_name,
        model_args=train_model_args,
        data_args=data_args,
        train_transform_args=train_transform_args,
        val_transform_args=val_transform_args,
        metric_args=ObjectDetectionTaskMetricArgs(),
        load_weights=False,
        gradient_accumulation_steps=1,
    )
    return train_model


def _create_task_model(
    *,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str = "dinov3/vitt16-notpretrained-ltdetr",
    patch_size: int = 16,
) -> DINOv3LTDETRObjectDetection:
    monkeypatch.setattr(DINOV3_PACKAGE, "get_model", _fake_get_model)
    return DINOv3LTDETRObjectDetection(
        model_name=model_name,
        classes={0: "class_0", 1: "class_1"},
        image_size=(640, 640),
        patch_size=patch_size,
        image_normalize=None,
        backbone_freeze=False,
        backbone_weights=None,
        backbone_args=None,
        load_weights=False,
    )


def _fake_get_model(*args: Any, **kwargs: Any) -> Any:
    model_name = kwargs.get("model_name", args[0] if args else "")
    model_args = kwargs.get("model_args", {}) or {}

    if str(model_name).startswith("convnext"):
        convnext_kwargs: dict[str, Any] = {"pretrained": False}
        if "patch_size" in model_args:
            convnext_kwargs["patch_size"] = int(model_args["patch_size"])
        return backbones._dinov3_convnext_test(**convnext_kwargs)

    return backbones._dinov3_vit_test(
        pretrained=False,
        patch_size=int(model_args.get("patch_size", 16)),
    )


def _create_train_model_with_capture(
    train_model_args: DINOv3LTDETRObjectDetectionTrainArgs,
    *,
    model_name: str = "dinov3/vitt16-notpretrained-ltdetr",
    model_init_args: dict[str, Any] | None = None,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DINOv3LTDETRObjectDetectionTrain, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake_get_model(*args: Any, **kwargs: Any) -> Any:
        calls.append(dict(kwargs))
        return _fake_get_model(*args, **kwargs)

    monkeypatch.setattr(DINOV3_PACKAGE, "get_model", fake_get_model)

    train_model = _create_train_model(
        train_model_args,
        model_name=model_name,
        model_init_args=model_init_args,
    )
    return train_model, calls


@pytest.mark.parametrize(
    ("scheduler_name", "scheduler_cls"),
    [
        ("linear", LinearLR),
        ("flat-cosine", FlatCosineLRScheduler),
    ],
)
def test_get_optimizer__scheduler_modes(
    scheduler_name: Literal["linear", "flat-cosine"],
    scheduler_cls: type[LinearLR] | type[FlatCosineLRScheduler],
) -> None:
    train_model = _create_train_model(
        DINOv3LTDETRObjectDetectionTrainArgs(
            scheduler_name=scheduler_name,
            lr_warmup_steps=500,
        )
    )
    optimizer, scheduler = train_model.get_optimizer(
        total_steps=1000,
        global_batch_size=16,
    )

    assert isinstance(scheduler, scheduler_cls)
    optimizer.step()
    scheduler.step()
    assert len(scheduler.get_last_lr()) == len(optimizer.param_groups)
    scheduler.load_state_dict(scheduler.state_dict())  # type: ignore[no-untyped-call]


def test_get_optimizer__flat_cosine_raises_when_cosine_phase_collapses() -> None:
    train_model = _create_train_model(
        DINOv3LTDETRObjectDetectionTrainArgs(
            scheduler_name="flat-cosine",
            lr_warmup_steps=1000,
        )
    )

    with pytest.raises(ValueError, match="non-empty cosine phase"):
        train_model.get_optimizer(total_steps=1000, global_batch_size=16)


def test_get_optimizer__linear_warns_when_warmup_exceeds_training(
    caplog: pytest.LogCaptureFixture,
) -> None:
    train_model = _create_train_model(
        DINOv3LTDETRObjectDetectionTrainArgs(
            scheduler_name="linear",
            lr_warmup_steps=1001,
        )
    )

    with caplog.at_level(
        logging.WARNING,
        logger="lightly_train._task_models.dinov3_ltdetr_object_detection.train_model",
    ):
        train_model.get_optimizer(total_steps=1000, global_batch_size=16)

    assert "the schedule will not complete as intended" in caplog.text


@pytest.mark.parametrize(
    ("patch_size", "feat_strides", "num_levels"),
    [
        (16, [8, 16, 32, 64], 4),
        (14, [7, 14, 28, 56], 4),
        (64, [32, 64, 128, 256], 4),
        (16, [8, 16, 32], 3),
        (14, [7, 14, 28], 3),
        (64, [32, 64, 128], 3),
    ],
)
def test_rtdetr_transformer_v2_config__resolve_auto__patch_size(
    patch_size: int, feat_strides: list[int], num_levels: int
):
    config = _RTDETRTransformerv2Config(
        num_levels=num_levels, feat_channels=[-1] * num_levels
    )

    config.resolve_auto(patch_size=patch_size)

    assert config.feat_strides == feat_strides
