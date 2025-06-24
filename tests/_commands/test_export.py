#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import re
import sys
from importlib import util as importlib_util
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torchvision import models as torchvision_models

from lightly_train._checkpoint import Checkpoint
from lightly_train._commands import export
from lightly_train._commands.export import CLIExportConfig, ExportConfig
from lightly_train._models.model_wrapper import ModelWrapper
from lightly_train._models.super_gradients.super_gradients_package import (
    SUPER_GRADIENTS_PACKAGE,
)
from lightly_train._models.timm.timm_package import TIMM_PACKAGE
from lightly_train._models.torchvision.torchvision_package import TORCHVISION_PACKAGE
from lightly_train._models.ultralytics.ultralytics_package import ULTRALYTICS_PACKAGE

from .. import helpers

if importlib_util.find_spec("ultralytics") is not None:
    from ultralytics import YOLO
else:
    YOLO = None

if importlib_util.find_spec("super_gradients") is not None:
    from super_gradients.training import models as super_gradient_models
else:
    super_gradient_models = None


def test_export__torch_state_dict(tmp_path: Path) -> None:
    """Check that exporting a model's state dict works as expected."""
    ckpt_path, ckpt = _get_checkpoint(tmp_path)
    wrapped_model = ckpt.lightly_train.models.wrapped_model
    embedding_model = ckpt.lightly_train.models.embedding_model
    part_expected = [
        ("model", wrapped_model.get_model().state_dict()),
        ("wrapped_model", wrapped_model.state_dict()),
        ("embedding_model", embedding_model.state_dict()),
    ]

    for part, expected in part_expected:
        out_path = tmp_path / f"{part}.pt"
        export.export(
            out=out_path,
            checkpoint=ckpt_path,
            part=part,
            format="torch_state_dict",
        )
        _assert_state_dict_equal(torch.load(out_path, weights_only=True), expected)


def test_export__torch_model(tmp_path: Path) -> None:
    """Check that exporting a model works as expected."""
    ckpt_path, ckpt = _get_checkpoint(tmp_path)
    wrapped_model = ckpt.lightly_train.models.wrapped_model
    embedding_model = ckpt.lightly_train.models.embedding_model
    part_expected = [
        ("model", wrapped_model.get_model()),
        ("wrapped_model", wrapped_model),
        ("embedding_model", embedding_model),
    ]

    for part, expected in part_expected:
        out_path = tmp_path / f"{part}.pt"
        export.export(
            out=out_path,
            checkpoint=ckpt_path,
            part=part,
            format="torch_model",
        )
        loaded_model = torch.load(out_path)
        assert isinstance(loaded_model, type(expected))
        _assert_state_dict_equal(loaded_model.state_dict(), expected.state_dict())


def test_export__torchvision(tmp_path: Path) -> None:
    """Check that exporting in torchvision format works as expected."""
    model = torchvision_models.resnet18()
    ckpt_path, ckpt = _get_checkpoint(
        tmp_path=tmp_path, wrapped_model=TORCHVISION_PACKAGE.get_model_wrapper(model)
    )
    out = tmp_path / "out.pt"
    export.export(out=out, checkpoint=ckpt_path)

    loaded_model = torchvision_models.resnet18()
    loaded_model.load_state_dict(torch.load(out, weights_only=True))
    assert isinstance(loaded_model, type(model))
    assert torch.equal(loaded_model.conv1.weight, model.conv1.weight)


@pytest.mark.skipif(YOLO is None, reason="ultralytics is not installed")
def test_export__ultralytics(tmp_path: Path) -> None:
    model = YOLO("yolov8n.yaml")
    ckpt_path, ckpt = _get_checkpoint(
        tmp_path=tmp_path, wrapped_model=ULTRALYTICS_PACKAGE.get_model_wrapper(model)
    )
    out = tmp_path / "out.pt"
    export.export(out=out, checkpoint=ckpt_path)
    loaded_model = YOLO(out)
    assert isinstance(loaded_model, YOLO)


@pytest.mark.skipif(YOLO is None, reason="ultralytics is not installed")
def test_export__ultralytics_option__deprecation_warning(tmp_path: Path) -> None:
    model = YOLO("yolov8n.yaml")
    ckpt_path, ckpt = _get_checkpoint(
        tmp_path=tmp_path, wrapped_model=ULTRALYTICS_PACKAGE.get_model_wrapper(model)
    )
    out = tmp_path / "out.pt"
    with pytest.warns(
        FutureWarning,
        match=re.escape(
            "The 'ultralytics' format is deprecated and will be removed in version "
            "0.5.0., instead the format can be omitted since it is mapped to the "
            "default format.",
        ),
    ):
        export.export(
            out=out,
            checkpoint=ckpt_path,
            part="model",
            format="ultralytics",
        )


@pytest.mark.skipif(
    super_gradient_models is None, reason="super_gradients is not installed"
)
def test_export__super_gradients(tmp_path: Path) -> None:
    model = super_gradient_models.get(model_name="yolo_nas_s", num_classes=3)
    model.backbone.stem.conv.branch_3x3.conv.weight.data.fill_(1.234)

    ckpt_path, ckpt = _get_checkpoint(
        tmp_path=tmp_path,
        wrapped_model=SUPER_GRADIENTS_PACKAGE.get_model_wrapper(model),
    )
    out = tmp_path / "out.pt"
    export.export(out=out, checkpoint=ckpt_path)

    loaded_model = super_gradient_models.get(
        model_name="yolo_nas_s", num_classes=3, checkpoint_path=str(out)
    )
    assert isinstance(loaded_model, type(model))
    assert torch.all(loaded_model.backbone.stem.conv.branch_3x3.conv.weight == 1.234)


def test_export__custom(tmp_path: Path) -> None:
    ckpt_path, ckpt = _get_checkpoint(tmp_path)
    model = ckpt.lightly_train.models.model

    out_path = tmp_path / "model.pt"
    export.export(out=out_path, checkpoint=ckpt_path)
    _assert_state_dict_equal(
        torch.load(out_path, weights_only=True), model.state_dict()
    )


@pytest.mark.skipif(
    importlib_util.find_spec("timm") is None, reason="timm is not installed"
)
def test_export__timm(tmp_path: Path) -> None:
    import timm

    model = timm.create_model("resnet18", pretrained=False)

    ckpt_path, ckpt = _get_checkpoint(
        tmp_path=tmp_path, wrapped_model=TIMM_PACKAGE.get_model_wrapper(model)
    )
    out = tmp_path / "out.pt"
    export.export(out=out, checkpoint=ckpt_path)

    loaded_model = timm.create_model(
        "resnet18", pretrained=False, checkpoint_path=str(out)
    )
    assert isinstance(loaded_model, type(model))


def test_export__invalid_part() -> None:
    """Check that an error is raised when an invalid part is provided."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid model part: 'invalid_part'. Valid parts are: "
            "['model', 'wrapped_model', 'embedding_model']"
        ),
    ):
        export.export(
            out="out.pt",
            checkpoint="checkpoint.pt",
            part="invalid_part",
            format="torch_state_dict",
        )


def test_export__invalid_format() -> None:
    """Check that an error is raised when an invalid format is provided."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid model format: 'invalid_format'. Valid formats are: "
            "['package_default', 'torch_model', 'torch_state_dict']"
        ),
    ):
        export.export(
            out="out.pt",
            checkpoint="checkpoint.pt",
            part="model",
            format="invalid_format",
        )


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Requires Python 3.10 or higher for typing."
)
def test_export__parameters() -> None:
    """Tests that export function and configs have the same parameters and default
    values.

    This test is here to make sure we don't forget to update the parameters in all
    places.
    """
    helpers.assert_same_params(a=ExportConfig, b=export.export)
    helpers.assert_same_params(a=ExportConfig, b=CLIExportConfig, assert_type=False)


@pytest.mark.parametrize(
    "part",
    [
        "model",
        "wrapped_model",
        "embedding_model",
    ],
)
def test_export_from_dictconfig(tmp_path: Path, part: str) -> None:
    ckpt_path, ckpt = _get_checkpoint(tmp_path)
    out_path = tmp_path / "model.pt"
    model = getattr(ckpt.lightly_train.models, part)
    config = OmegaConf.create(
        dict(
            checkpoint=str(ckpt_path),
            out=str(out_path),
            part=part,
            format="torch_state_dict",
        )
    )
    export.export_from_dictconfig(config=config)
    _assert_state_dict_equal(
        torch.load(out_path, weights_only=True), model.state_dict()
    )


def _assert_state_dict_equal(a: dict[str, Tensor], b: dict[str, Tensor]) -> None:
    assert a.keys() == b.keys()
    for key in a.keys():
        assert torch.equal(a[key], b[key])


def _get_checkpoint(
    tmp_path: Path, wrapped_model: ModelWrapper | None = None
) -> tuple[Path, Checkpoint]:
    checkpoint = helpers.get_checkpoint(wrapped_model=wrapped_model)
    ckpt_path = tmp_path / "last.ckpt"
    checkpoint.save(ckpt_path)
    return ckpt_path, checkpoint
