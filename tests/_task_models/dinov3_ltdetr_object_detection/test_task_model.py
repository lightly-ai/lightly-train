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
import torch
from lightning_utilities.core.imports import RequirementCache
from pytest_mock import MockerFixture
from torch import nn
from torch.optim.lr_scheduler import LinearLR

from lightly_train._data.yolo_object_detection_dataset import (
    YOLOObjectDetectionDataArgs,
)
from lightly_train._metrics.detection.task_metric import ObjectDetectionTaskMetricArgs
from lightly_train._task_models.dinov3_ltdetr.task_model import (
    _RTDETRTransformerv2Config,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.config import (
    LTDETR_MODEL_REGISTRY,
    DFINETransformerConfig,
    RTDETRTransformerv2Config,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.task_model import (
    LTDETRObjectDetection,
    _resolve_transformer_config,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.train_model import (
    LTDETRObjectDetectionTrain,
    LTDETRObjectDetectionTrainArgs,
)
from lightly_train._task_models.dinov3_ltdetr_object_detection.transforms import (
    LTDETRObjectDetectionTrainTransformArgs,
    LTDETRObjectDetectionValTransformArgs,
)
from lightly_train._task_models.object_detection_components.flat_cosine import (
    FlatCosineLRScheduler,
)
from lightly_train._task_models.object_detection_components.rtdetrv2_decoder import (
    RTDETRTransformerv2,
)


def _is_module_frozen(m: nn.Module) -> bool:
    return all(not param.requires_grad for param in m.parameters())


def _create_train_model(
    train_model_args: LTDETRObjectDetectionTrainArgs,
    *,
    model_name: str = "dinov3/vitt16-notpretrained-ltdetr",
    model_init_args: dict[str, Any] | None = None,
) -> LTDETRObjectDetectionTrain:
    data_args = YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
    )
    train_model_args.resolve_auto(
        total_steps=1000,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name=model_name,
        model_init_args={} if model_init_args is None else model_init_args,
        data_args=data_args,
    )
    train_transform_args = LTDETRObjectDetectionTrainTransformArgs()
    train_transform_args.resolve_auto(model_init_args={})
    val_transform_args = LTDETRObjectDetectionValTransformArgs()
    val_transform_args.resolve_auto(model_init_args={})

    train_model = LTDETRObjectDetectionTrain(
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


@pytest.fixture()
def dummy_yolo_detection_data_args() -> YOLOObjectDetectionDataArgs:
    return YOLOObjectDetectionDataArgs(
        path=Path("/tmp/data"),
        train=Path("train") / "images",
        val=Path("val") / "images",
        names={0: "class_0", 1: "class_1"},
    )


@pytest.mark.parametrize("use_ema_model", [True, False])
def test_load_train_state_dict__from_exported(use_ema_model: bool) -> None:
    model_args = LTDETRObjectDetectionTrainArgs(use_ema_model=use_ema_model)
    train_model = _create_train_model(model_args)
    task_model = train_model.model
    state_dict = train_model.get_export_state_dict()
    task_model.load_train_state_dict(state_dict)


def test_load_train_state_dict__no_ema_weights() -> None:
    model_args = LTDETRObjectDetectionTrainArgs(use_ema_model=True)
    train_model = _create_train_model(model_args)
    task_model = train_model.model
    state_dict = train_model.state_dict()
    # Drop all EMA weights from the state dict. This is for backwards compatibility
    # with older checkpoints. The model should still be able to load the weights by
    # copying the non-EMA weights to the EMA model.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ema_model.")}
    task_model.load_train_state_dict(state_dict)


@pytest.mark.parametrize("should_freeze", [True, False])
def test_freeze_backbone_on_set_train_mode(should_freeze: bool) -> None:
    model_args = LTDETRObjectDetectionTrainArgs(
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


@pytest.mark.parametrize(
    ("model_name", "expected_patch_size"),
    [("dinov3/vitt16-ltdetr-coco", 16), ("dinov3/convnext-tiny-ltdetr-coco", None)],
)
def test_resolve_auto__uses_vit_model_name(
    model_name: str,
    expected_patch_size: int | str,
    dummy_yolo_detection_data_args: YOLOObjectDetectionDataArgs,
) -> None:
    model_args = LTDETRObjectDetectionTrainArgs()

    model_args.resolve_auto(
        total_steps=1000,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name=model_name,
        model_init_args={},
        data_args=dummy_yolo_detection_data_args,
    )

    assert model_args.patch_size == expected_patch_size


@pytest.mark.parametrize(
    "model_name",
    ["ltdetrv2-s", "ltdetrv2-m", "ltdetrv2-l", "ltdetrv2-x"],
)
def test_resolve_auto__uses_ltdetrv2_alias(
    model_name: str,
    dummy_yolo_detection_data_args: YOLOObjectDetectionDataArgs,
) -> None:
    # Regression test for TRN-2187: ``resolve_auto`` runs before the task-model
    # constructor canonicalizes short LT-DETRv2 aliases, so it must resolve
    # them itself (via ``parse_model_name``) rather than raising
    # ``Unable to resolve patch_size='auto'``. All aliases map to EdgeCrafter
    # (ECViT) backbones with a fixed patch_size of 16.
    model_args = LTDETRObjectDetectionTrainArgs()

    model_args.resolve_auto(
        total_steps=1000,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name=model_name,
        model_init_args={},
        data_args=dummy_yolo_detection_data_args,
    )

    assert model_args.patch_size == 16


@pytest.mark.parametrize(
    ("model_name", "expected_patch_size"),
    [("dinov3/vitt16-ltdetr-coco", 36), ("dinov3/convnext-tiny-ltdetr-coco", 47)],
)
def test_resolve_auto__uses_model_init_args_patch_size(
    model_name: str,
    expected_patch_size: int,
    dummy_yolo_detection_data_args: YOLOObjectDetectionDataArgs,
) -> None:
    model_args = LTDETRObjectDetectionTrainArgs()

    model_args.resolve_auto(
        total_steps=1000,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name=model_name,
        model_init_args={"patch_size": expected_patch_size},
        data_args=dummy_yolo_detection_data_args,
    )

    assert model_args.patch_size == expected_patch_size


@pytest.mark.parametrize(
    ("model_name", "expected_patch_size"),
    [("dinov3/vitt16-ltdetr-coco", 36), ("dinov3/convnext-tiny-ltdetr-coco", 47)],
)
def test_resolve_auto__uses_model_explicit_patch_size_arg(
    model_name: str,
    expected_patch_size: int,
    dummy_yolo_detection_data_args: YOLOObjectDetectionDataArgs,
) -> None:
    model_args = LTDETRObjectDetectionTrainArgs(patch_size=expected_patch_size)

    model_args.resolve_auto(
        total_steps=1000,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name=model_name,
        model_init_args={},
        data_args=dummy_yolo_detection_data_args,
    )

    assert model_args.patch_size == expected_patch_size


def test_checkpoint_roundtrip__rtdetrv2_decoder_preserved_when_not_explicit() -> None:
    # Backwards compatibility: a checkpoint trained with the previous
    # RTDETRv2 default must reconstruct with the RTDETRv2 architecture when
    # the user does not explicitly set ``decoder_name``, even though the new
    # default is dfine. This round-trip simulates loading such a checkpoint:
    # we save the task model's ``init_args`` and ``state_dict`` (mirroring
    # what ``lightly_train`` persists in the checkpoint file), build a fresh
    # train model from defaults, feed it the saved ``init_args`` as the
    # checkpoint payload, and verify the state dict loads cleanly into an
    # RTDETRv2 decoder.
    model_name = "dinov3/vitt16-notpretrained-ltdetr"

    # Source: an old-style RTDETRv2 checkpoint.
    source_args = LTDETRObjectDetectionTrainArgs(
        decoder_name="rtdetrv2",
        use_ema_model=False,
    )
    source_train_model = _create_train_model(
        source_args,
        model_name=model_name,
    )
    checkpoint_model_init_args = source_train_model.get_task_model().init_args
    assert checkpoint_model_init_args["decoder_name"] == "rtdetrv2"
    checkpoint_state_dict = source_train_model.state_dict()

    # Pick a stable decoder tensor to compare before/after the round trip.
    decoder_keys = sorted(k for k in checkpoint_state_dict if "decoder" in k)
    assert decoder_keys, "expected at least one decoder tensor in the state dict"
    source_decoder_tensor = checkpoint_state_dict[decoder_keys[0]].clone()

    # Target: a fresh train model built from the new defaults. The user does
    # not explicitly set ``decoder_name``; ``model_init_args`` carries the
    # checkpoint payload.
    target_args = LTDETRObjectDetectionTrainArgs(use_ema_model=False)
    target_train_model = _create_train_model(
        target_args,
        model_name=model_name,
        model_init_args=dict(checkpoint_model_init_args),
    )

    # Architecture was reconstructed as RTDETRv2 via the compatibility shim.
    assert target_args.decoder_name == "rtdetrv2"
    target_task_model = target_train_model.get_task_model()
    assert isinstance(target_task_model.decoder, RTDETRTransformerv2)

    # State dict loads cleanly into the reconstructed architecture.
    incompatible = target_train_model.load_train_state_dict(checkpoint_state_dict)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    # Decoder weights actually landed on the target.
    loaded_decoder_tensor = target_train_model.state_dict()[decoder_keys[0]]
    torch.testing.assert_close(loaded_decoder_tensor, source_decoder_tensor)


@pytest.mark.parametrize(
    ("model_name", "decoder_name", "expected_config_type"),
    [
        ("dinov3/vitt16-notpretrained-ltdetr", None, RTDETRTransformerv2Config),
        ("dinov3/vitt16-notpretrained-ltdetr", "dfine", DFINETransformerConfig),
        ("ltdetrv2-s", None, DFINETransformerConfig),
        ("ltdetrv2-s", "rtdetrv2", RTDETRTransformerv2Config),
    ],
)
def test_resolve_transformer_config__selects_decoder_family(
    model_name: str,
    decoder_name: Literal["rtdetrv2", "dfine"] | None,
    expected_config_type: type[RTDETRTransformerv2Config | DFINETransformerConfig],
) -> None:
    config = LTDETR_MODEL_REGISTRY.get(alias=model_name)()

    transformer_config = _resolve_transformer_config(
        config=config, decoder_name=decoder_name
    )

    assert isinstance(transformer_config, expected_config_type)


def test_resolve_auto__warns_on_explicit_checkpoint_decoder_conflict(
    caplog: pytest.LogCaptureFixture,
    dummy_yolo_detection_data_args: YOLOObjectDetectionDataArgs,
) -> None:
    model_args = LTDETRObjectDetectionTrainArgs(decoder_name="dfine")

    with caplog.at_level(
        logging.WARNING,
        logger="lightly_train._task_models.dinov3_ltdetr_object_detection.train_model",
    ):
        model_args.resolve_auto(
            total_steps=1000,
            gradient_accumulation_steps=1,
            train_num_batches=100,
            model_name="dinov3/vitt16-ltdetr-coco",
            model_init_args={"decoder_name": "rtdetrv2"},
            data_args=dummy_yolo_detection_data_args,
        )

    assert model_args.decoder_name == "dfine"
    assert "checkpoint's decoder_name='rtdetrv2'" in caplog.text


def test_resolve_auto__auto_lr_warmup_steps_short_run(
    dummy_yolo_detection_data_args: YOLOObjectDetectionDataArgs,
) -> None:
    # ``lr_warmup_steps`` defaults to ``"auto"`` so short default runs do not
    # collapse the flat-cosine phase to zero (see P2 Codex review on PR #798).
    model_args = LTDETRObjectDetectionTrainArgs()

    model_args.resolve_auto(
        total_steps=100,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name="ltdetrv2-s",
        model_init_args={},
        data_args=dummy_yolo_detection_data_args,
    )

    assert isinstance(model_args.lr_warmup_steps, int)
    assert 0 <= model_args.lr_warmup_steps < 100


def test_task_model_init_args_roundtrip_preserves_patch_size() -> None:
    model = LTDETRObjectDetection(
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        classes={0: "class_0", 1: "class_1"},
        image_size=(640, 640),
        patch_size=14,
        image_normalize=None,
        backbone_freeze=False,
        backbone_weights=None,
        backbone_args=None,
        load_weights=False,
    )

    assert model.init_args["patch_size"] == 14

    roundtrip_model = LTDETRObjectDetection(
        **model.init_args,
        load_weights=False,
    )
    assert roundtrip_model.init_args == model.init_args
    assert roundtrip_model.init_args["patch_size"] == 14


@pytest.mark.parametrize(
    ("patch_size", "expected_image_size"),
    [
        (14, (644, 644)),
        (16, (640, 640)),
        (24, (672, 672)),
        (64, (640, 640)),
    ],
)
def test_train_transform_args__resolve_auto__image_size_is_2x_patch_size_compatible(
    patch_size: int,
    expected_image_size: tuple[int, int],
) -> None:
    train_transform_args = LTDETRObjectDetectionTrainTransformArgs()
    train_transform_args.resolve_auto(model_init_args={"patch_size": patch_size})

    assert train_transform_args.image_size == expected_image_size


@pytest.mark.parametrize("patch_size", [14, 16, 64])
def test_train_transform_args__resolve_auto__scale_jitter_divisible_by_patch_size(
    patch_size: int,
) -> None:
    train_transform_args = LTDETRObjectDetectionTrainTransformArgs()
    train_transform_args.resolve_auto(model_init_args={"patch_size": patch_size})

    assert train_transform_args.scale_jitter is not None, (
        "scale_jitter should not be None after resolve_auto"
    )

    assert train_transform_args.scale_jitter.divisible_by == patch_size * 2


@pytest.mark.parametrize(
    ("patch_size", "expected_image_size"),
    [
        (14, (644, 644)),
        (16, (640, 640)),
        (24, (672, 672)),
        (64, (640, 640)),
    ],
)
def test_val_transform_args__resolve_auto__image_size_is_2x_patch_size_compatible(
    patch_size: int,
    expected_image_size: tuple[int, int],
) -> None:
    val_transform_args = LTDETRObjectDetectionValTransformArgs()
    val_transform_args.resolve_auto(model_init_args={"patch_size": patch_size})

    assert val_transform_args.image_size == expected_image_size


@pytest.mark.parametrize(
    "transform_args_cls",
    [
        LTDETRObjectDetectionTrainTransformArgs,
        LTDETRObjectDetectionValTransformArgs,
    ],
)
def test_transform_args__resolve_auto__preserves_explicit_image_size(
    transform_args_cls: type[
        LTDETRObjectDetectionTrainTransformArgs | LTDETRObjectDetectionValTransformArgs
    ],
) -> None:
    transform_args = transform_args_cls()
    transform_args.resolve_auto(
        model_init_args={"patch_size": 14, "image_size": (672, 672)}
    )

    assert transform_args.image_size == (672, 672)


@pytest.mark.parametrize(
    "transform_args_cls",
    [
        LTDETRObjectDetectionTrainTransformArgs,
        LTDETRObjectDetectionValTransformArgs,
    ],
)
def test_transform_args__resolve_auto__rejects_incompatible_explicit_image_size(
    transform_args_cls: type[
        LTDETRObjectDetectionTrainTransformArgs | LTDETRObjectDetectionValTransformArgs
    ],
) -> None:
    transform_args = transform_args_cls()

    with pytest.raises(ValueError, match=r"2 \* the patch size"):
        transform_args.resolve_auto(
            model_init_args={"patch_size": 14, "image_size": (658, 658)}
        )


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
        LTDETRObjectDetectionTrainArgs(
            scheduler_name=scheduler_name,
            lr_warmup_steps=500,
            scheduler_flat_steps=550,
            scheduler_no_aug_steps=150,
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
        LTDETRObjectDetectionTrainArgs(
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
        LTDETRObjectDetectionTrainArgs(
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
) -> None:
    config = _RTDETRTransformerv2Config(
        num_levels=num_levels, feat_channels=[-1] * num_levels
    )

    config.resolve_auto(patch_size=patch_size)


def test_predict_batch__composes_stages_in_order(mocker: MockerFixture) -> None:
    model = LTDETRObjectDetection(
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        classes={0: "class_0", 1: "class_1"},
        image_size=(256, 256),
        load_weights=False,
    )

    preprocess_image_spy = mocker.spy(model, "preprocess_image")
    preprocess_batch_spy = mocker.spy(model, "preprocess_batch")
    forward_backend_spy = mocker.spy(model, "forward_backend")
    postprocess_spy = mocker.spy(model, "postprocess")

    images = [torch.rand(3, 480, 640), torch.rand(3, 720, 1280)]
    result = model.predict_batch(images=images)

    # Each input image goes through preprocess_image once.
    assert preprocess_image_spy.call_count == 2

    # The stacked batch is preprocessed in a single call with shape (B, C, H, W).
    assert preprocess_batch_spy.call_count == 1
    (batch_in,) = preprocess_batch_spy.call_args.args
    assert batch_in.shape == (2, 3, 256, 256)

    # forward_backend receives the output of preprocess_batch.
    assert forward_backend_spy.call_count == 1
    (forward_in,) = forward_backend_spy.call_args.args
    assert forward_in is preprocess_batch_spy.spy_return

    # postprocess receives forward_backend's output and per-image metadata.
    assert postprocess_spy.call_count == 1
    raw_in, metadata = postprocess_spy.call_args.args
    assert raw_in is forward_backend_spy.spy_return
    assert len(metadata) == 2

    # predict_batch returns whatever postprocess produced.
    assert result is postprocess_spy.spy_return


def test_dinov2_backbone__uses_dinov3_ltdetr_wrapper() -> None:
    from lightly_train._task_models.dinov3_ltdetr_object_detection.dinov3_vit_wrapper import (
        DINOSTAs,
    )

    model = LTDETRObjectDetection(
        model_name="dinov2/vits14-ltdetr",
        classes={0: "class_0", 1: "class_1"},
        image_size=(224, 224),
        load_weights=False,
    )

    assert isinstance(model.backbone, DINOSTAs)
    assert model.backbone.use_sta is False

    model.eval()
    with torch.no_grad():
        outputs = model.forward_backend(torch.rand(1, 3, 224, 224))

    assert outputs["pred_logits"].shape[0] == 1
    assert outputs["pred_boxes"].shape[0] == 1


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__dynamic_batch_size(tmp_path: Path) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = LTDETRObjectDetection(
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        classes={0: "car", 1: "person"},
        image_size=(256, 256),
        load_weights=False,
    )

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, simplify=False, verify=True)

    onnx_model = onnx.load(out)
    input_batch_dim = onnx_model.graph.input[0].type.tensor_type.shape.dim[0]
    assert input_batch_dim.dim_param == "N"

    import torch

    inputs = np.random.randn(3, 3, 256, 256).astype(np.float32)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": inputs})

    with torch.no_grad():
        torch_outputs = model(torch.from_numpy(inputs))

    for onnx_out, torch_out in zip(onnx_outputs, torch_outputs):
        onnx_tensor = torch.from_numpy(onnx_out)
        if torch_out.is_floating_point():
            close = torch.isclose(onnx_tensor, torch_out, atol=2e-2, rtol=1e-1)
            assert close.float().mean() > 0.95
        else:
            assert (onnx_tensor == torch_out).float().mean() > 0.95


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
def test_export_onnx__static_batch_size(tmp_path: Path) -> None:
    model = LTDETRObjectDetection(
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        classes={0: "car", 1: "person"},
        image_size=(256, 256),
        load_weights=False,
    )

    out = tmp_path / "model.onnx"
    model.export_onnx(
        out=out, batch_size=3, dynamic_batch_size=False, simplify=False, verify=True
    )


@pytest.mark.skipif(not RequirementCache("onnx"), reason="onnx not installed")
@pytest.mark.skipif(
    not RequirementCache("onnxruntime"), reason="onnxruntime not installed"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
@pytest.mark.parametrize("decoder_name", ["rtdetrv2", "dfine"])
def test_export_onnx__fp16(
    tmp_path: Path, decoder_name: Literal["rtdetrv2", "dfine"]
) -> None:
    import onnx

    model = LTDETRObjectDetection(
        model_name="dinov3/vitt16-notpretrained-ltdetr",
        classes={0: "car", 1: "person"},
        image_size=(256, 256),
        load_weights=False,
    )

    out = tmp_path / "model.onnx"
    model.export_onnx(out=out, precision="fp16", simplify=True, verify=True)

    model_onnx = onnx.load(str(out))
    # Verify the model has fp16 tensors.
    has_fp16 = any(
        init.data_type == onnx.TensorProto.FLOAT16
        for init in model_onnx.graph.initializer
    )
    assert has_fp16


# ---------------------------------------------------------------------------
# EdgeCrafter (ECViT) backbone tests
# ---------------------------------------------------------------------------
#
# The ECViT backbones are exposed under the ``edgecrafter/`` package prefix
# (e.g. ``edgecrafter/ecvitt-ltdetr``) and are dispatched inside the DINOv3
# LTDETR task model. These tests verify the wiring without depending on the
# pretrained weight download.


ECVIT_LTDETR_MODEL_NAMES = [
    "edgecrafter/ecvitt-ltdetr",
    "edgecrafter/ecvittplus-ltdetr",
    "edgecrafter/ecvits-ltdetr",
    "edgecrafter/ecvitsplus-ltdetr",
]


@pytest.mark.parametrize("model_name", ECVIT_LTDETR_MODEL_NAMES)
def test_is_supported_model__ecvit(model_name: str) -> None:
    assert LTDETRObjectDetection.is_supported_model(model_name) is True


@pytest.mark.parametrize("model_name", ECVIT_LTDETR_MODEL_NAMES)
def test_parse_model_name__ecvit(model_name: str) -> None:
    parsed = LTDETRObjectDetection.parse_model_name(model_name)
    assert parsed["package_name"] == "edgecrafter"
    assert parsed["model_name"] == model_name
    # backbone_name is the bare ECViT preset (no package prefix, no -ltdetr
    # suffix).
    assert parsed["backbone_name"] in {
        "ecvitt",
        "ecvittplus",
        "ecvits",
        "ecvitsplus",
    }


@pytest.mark.parametrize("model_name", ECVIT_LTDETR_MODEL_NAMES)
def test_create_train_model__ecvit(
    model_name: str,
    dummy_yolo_detection_data_args: YOLOObjectDetectionDataArgs,
) -> None:
    from lightly_train._task_models.dinov3_ltdetr_object_detection.ecvit_vit_wrapper import (
        ECViTBackboneWrapper,
    )

    model_args = LTDETRObjectDetectionTrainArgs()
    model_args.resolve_auto(
        total_steps=1000,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name=model_name,
        model_init_args={},
        data_args=dummy_yolo_detection_data_args,
    )
    # ECViT always resolves to patch_size=16 (the ECViT-NN ConvPyramidPatchEmbed
    # uses a fixed patch size of 16).
    assert model_args.patch_size == 16

    train_model = _create_train_model(
        model_args, model_name=model_name, model_init_args={"patch_size": 16}
    )
    task_model = train_model.model
    # The backbone must be an ECViTBackboneWrapper, not a DINOv3 wrapper.
    assert isinstance(task_model.backbone, ECViTBackboneWrapper)
    # ECViT has no mask_token; the constructor must not have tried to freeze
    # one (which would AttributeError).
    assert not hasattr(task_model.backbone, "mask_token")
    # The wrapped ECViTModelWrapper itself must not have a mask_token either.
    assert not hasattr(task_model.backbone.backbone_model, "mask_token")


# ---------------------------------------------------------------------------
# Short LT-DETRv2 alias tests
# ---------------------------------------------------------------------------
#
# ``ltdetrv2-{s,m,l,x}`` is a public alias that resolves to the canonical
# EdgeCrafter (ECViT) LT-DETR object-detection model name. These tests verify
# that the alias is accepted by ``is_supported_model`` and resolves to the
# correct canonical name in ``parse_model_name``.

LTDETR_V2_ALIAS_MODEL_NAMES = [
    "ltdetrv2-s",
    "ltdetrv2-m",
    "ltdetrv2-l",
    "ltdetrv2-x",
]

LTDETR_V2_ALIAS_TO_CANONICAL: dict[str, str] = {
    "ltdetrv2-s": "edgecrafter/ecvitt-ltdetr",
    "ltdetrv2-m": "edgecrafter/ecvittplus-ltdetr",
    "ltdetrv2-l": "edgecrafter/ecvits-ltdetr",
    "ltdetrv2-x": "edgecrafter/ecvitsplus-ltdetr",
}


@pytest.mark.parametrize("model_name", LTDETR_V2_ALIAS_MODEL_NAMES)
def test_is_supported_model__ltdetrv2_alias(model_name: str) -> None:
    assert LTDETRObjectDetection.is_supported_model(model_name) is True


@pytest.mark.parametrize(
    ("alias", "canonical"),
    list(LTDETR_V2_ALIAS_TO_CANONICAL.items()),
)
def test_parse_model_name__ltdetrv2_alias(alias: str, canonical: str) -> None:
    parsed = LTDETRObjectDetection.parse_model_name(alias)
    assert parsed["package_name"] == "edgecrafter"
    assert parsed["model_name"] == canonical
    # backbone_name must be the bare ECViT preset (no package prefix, no
    # -ltdetr suffix) so the EdgeCrafter package can load the weights.
    expected_backbone_name = canonical
    if expected_backbone_name.startswith("edgecrafter/"):
        expected_backbone_name = expected_backbone_name[len("edgecrafter/") :]
    if expected_backbone_name.endswith("-ltdetr"):
        expected_backbone_name = expected_backbone_name[: -len("-ltdetr")]
    assert parsed["backbone_name"] == expected_backbone_name


def test_list_model_names__includes_ltdetrv2_aliases() -> None:
    names = LTDETRObjectDetection.list_model_names()
    for alias in LTDETR_V2_ALIAS_MODEL_NAMES:
        assert alias in names, f"Expected alias {alias!r} in list_model_names()"


@pytest.mark.parametrize("should_freeze", [True, False])
def test_freeze_backbone_on_set_train_mode__ecvit(should_freeze: bool) -> None:
    # ECViT's backbone has no mask_token, so the constructor's DINOv3-ViT
    # branch (which would call ``backbone.mask_token.requires_grad = False``)
    # must be skipped. This test exercises the full construction + set_train_mode
    # path to confirm there is no AttributeError on ``mask_token``.
    model_args = LTDETRObjectDetectionTrainArgs(
        use_ema_model=True,
        backbone_freeze=should_freeze,
    )
    train_model = _create_train_model(
        model_args,
        model_name="edgecrafter/ecvitt-ltdetr",
        model_init_args={"patch_size": 16},
    )
    task_model_backbone = train_model.model.backbone
    assert isinstance(task_model_backbone, nn.Module)
    assert not hasattr(task_model_backbone, "mask_token")

    train_model.set_train_mode()

    assert _is_module_frozen(task_model_backbone) == should_freeze


def test_resolve_auto__ecvit_patch_size_is_16(
    dummy_yolo_detection_data_args: YOLOObjectDetectionDataArgs,
) -> None:
    # Belt-and-braces: the explicit resolve_auto test for the ECViT package
    # branch, separate from the parametrized _create_train_model test above.
    model_args = LTDETRObjectDetectionTrainArgs()
    model_args.resolve_auto(
        total_steps=1000,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name="edgecrafter/ecvitsplus-ltdetr",
        model_init_args={},
        data_args=dummy_yolo_detection_data_args,
    )
    assert model_args.patch_size == 16


def test_resolve_auto__ecvit_model_init_args_patch_size_wins(
    dummy_yolo_detection_data_args: YOLOObjectDetectionDataArgs,
) -> None:
    # An explicit ``patch_size`` in ``model_init_args`` must override the
    # ECViT default of 16 (same precedence as the DINOv3 path).
    model_args = LTDETRObjectDetectionTrainArgs()
    model_args.resolve_auto(
        total_steps=1000,
        gradient_accumulation_steps=1,
        train_num_batches=100,
        model_name="edgecrafter/ecvitt-ltdetr",
        model_init_args={"patch_size": 32},
        data_args=dummy_yolo_detection_data_args,
    )
    assert model_args.patch_size == 32


@pytest.mark.parametrize("model_name", ECVIT_LTDETR_MODEL_NAMES)
def test_task_model__ecvit_rejects_non_16_patch_size(model_name: str) -> None:
    # The ECViT-NN ConvPyramidPatchEmbed only supports patch_size=16. The
    # task model must reject any other value with a ValueError naming the
    # constraint; otherwise the decoder's `config.resolve_auto` would build
    # strides at the wrong scale and anchors would be misaligned.
    with pytest.raises(ValueError, match=r"patch_size=16"):
        LTDETRObjectDetection(
            model_name=model_name,
            classes={0: "class_0", 1: "class_1"},
            image_size=(640, 640),
            patch_size=32,
            load_weights=False,
        )


@pytest.mark.parametrize("model_name", ECVIT_LTDETR_MODEL_NAMES)
@pytest.mark.parametrize("patch_size", [16, None])
def test_task_model__ecvit_accepts_patch_size_16_or_default(
    model_name: str, patch_size: int | None
) -> None:
    # Either an explicit patch_size=16 or the default None must construct
    # successfully, and the decoder strides must be `[8, 16, 32]` (matching
    # what the ECViTBackboneWrapper actually emits).
    model = LTDETRObjectDetection(
        model_name=model_name,
        classes={0: "class_0", 1: "class_1"},
        image_size=(640, 640),
        patch_size=patch_size,
        load_weights=False,
    )
    assert model.decoder.feat_strides == [8, 16, 32]


@pytest.mark.parametrize("model_name", ECVIT_LTDETR_MODEL_NAMES)
def test_get_optimizer__ecvit_splits_pretrained_backbone_from_projector(
    model_name: str,
) -> None:
    # The pretrained ECViT VisionTransformer must get the low
    # `backbone_lr_factor`, but the freshly initialized projector must train
    # at the full detector LR. Otherwise the connector never converges
    # during fine-tuning.
    from lightly_train._models.ecvit.ecvit import ECViTModelWrapper
    from lightly_train._task_models.dinov3_ltdetr_object_detection.ecvit_vit_wrapper import (
        ECViTBackboneWrapper,
    )

    # Pin the linear scheduler here: this test asserts optimizer param-group
    # membership and LRs (the backbone/detector split), not the scheduler.
    # The default `flat-cosine` would otherwise auto-resolve flat/no_aug steps
    # from the test's `total_steps=1000` and collide with the default
    # `lr_warmup_steps=2000`.
    # `scheduler_start_factor=1.0` neutralizes LinearLR's init `step()` scaling.
    train_model = _create_train_model(
        LTDETRObjectDetectionTrainArgs(
            scheduler_name="linear",
            scheduler_start_factor=1.0,
        ),
        model_name=model_name,
        model_init_args={"patch_size": 16},
    )
    backbone = train_model.model.backbone
    assert isinstance(backbone, ECViTBackboneWrapper)
    ecvit_wrapper = backbone.backbone_model
    assert isinstance(ecvit_wrapper, ECViTModelWrapper)

    pretrained_param_ids = {id(p) for p in ecvit_wrapper.backbone.parameters()}
    projector_param_ids = {id(p) for p in ecvit_wrapper.projector.parameters()}
    assert pretrained_param_ids.isdisjoint(projector_param_ids)
    assert len(pretrained_param_ids) > 0
    assert len(projector_param_ids) > 0

    optimizer, _ = train_model.get_optimizer(
        total_steps=1000,
        global_batch_size=train_model.model_args.default_batch_size,
    )
    by_name = {g["name"]: g for g in optimizer.param_groups}

    expected_backbone_lr = (
        train_model.model_args.lr * train_model.model_args.backbone_lr_factor
    )
    expected_detector_lr = train_model.model_args.lr

    # Params get split between the weight-decay and no-weight-decay variants
    # of each LR group based on `optimizer_helpers.get_weight_decay_parameters`.
    # Assert membership on the union of the two variants per side.
    backbone_union_ids: set[int] = set()
    for group_name in ("backbone", "backbone_no_wd"):
        group = by_name[group_name]
        assert group["lr"] == pytest.approx(expected_backbone_lr)
        backbone_union_ids.update(id(p) for p in group["params"])
    assert backbone_union_ids == pretrained_param_ids, (
        "backbone groups together must contain exactly the pretrained "
        "ECViT VisionTransformer params (and nothing else)"
    )

    detector_union_ids: set[int] = set()
    for group_name in ("detector", "detector_no_wd"):
        group = by_name[group_name]
        assert group["lr"] == pytest.approx(expected_detector_lr)
        detector_union_ids.update(id(p) for p in group["params"])
    # The freshly initialized projector must end up in the detector group
    # (not the backbone group) so it trains at the full detector LR.
    assert projector_param_ids <= detector_union_ids, (
        "detector groups together must include the freshly initialized ECViT projector"
    )
    assert projector_param_ids.isdisjoint(backbone_union_ids), (
        "projector params must not be in the backbone groups"
    )
