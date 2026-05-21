#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lightning_fabric.loggers.logger import Logger as FabricLogger
from PIL import Image as PILImageModule
from pytest_mock import MockerFixture
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from lightly_train._loggers import logger_helpers
from lightly_train._loggers.jsonl import JSONLLogger, JSONLLoggerArgs
from lightly_train._loggers.logger_args import LoggerArgs
from lightly_train._loggers.mlflow import MLFlowLogger, MLFlowLoggerArgs
from lightly_train._loggers.tensorboard import TensorBoardLogger, TensorBoardLoggerArgs
from lightly_train._loggers.wandb import WandbLogger, WandbLoggerArgs
from lightly_train.errors import ConfigValidationError

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]


@pytest.mark.parametrize(
    "loggers, expected",
    [
        # Test case for default empty dictionary
        ({}, LoggerArgs()),
        # Test case for None input
        (None, LoggerArgs()),
        # Test case for user config
        (
            {"jsonl": {"flush_logs_every_n_steps": 5}},
            LoggerArgs(jsonl=JSONLLoggerArgs(flush_logs_every_n_steps=5)),
        ),
        # Test case for disabling logger
        (
            {"jsonl": None},
            LoggerArgs(jsonl=None),
        ),
        # Test case for passing LoggerArgs
        (
            LoggerArgs(jsonl=JSONLLoggerArgs(flush_logs_every_n_steps=5)),
            LoggerArgs(jsonl=JSONLLoggerArgs(flush_logs_every_n_steps=5)),
        ),
    ],
)
def test_get_logger_args__success(
    loggers: dict[str, Any] | None, expected: LoggerArgs
) -> None:
    logger_args = logger_helpers.get_logger_args(loggers=loggers)
    assert logger_args == expected


def test_get_logger_args__failure() -> None:
    with pytest.raises(ConfigValidationError):
        logger_helpers.get_logger_args({"nonexisting_arg": 1})


def test_get_loggers__default(tmp_path: Path) -> None:
    loggers = logger_helpers.get_loggers(
        logger_args=LoggerArgs(),
        out=tmp_path,
    )
    assert any(isinstance(logger, JSONLLogger) for logger in loggers)
    assert any(isinstance(logger, TensorBoardLogger) for logger in loggers)
    assert not any(isinstance(logger, WandbLogger) for logger in loggers)
    assert not any(isinstance(logger, MLFlowLogger) for logger in loggers)


def test_get_loggers__disable(tmp_path: Path) -> None:
    loggers = logger_helpers.get_loggers(
        logger_args=LoggerArgs(
            jsonl=None,
            mlflow=None,
            tensorboard=None,
            wandb=None,
        ),
        out=tmp_path,
    )
    assert loggers == []


def test_get_callbacks__jsonl_user_config(tmp_path: Path) -> None:
    loggers = logger_helpers.get_loggers(
        logger_args=LoggerArgs(
            jsonl=JSONLLoggerArgs(flush_logs_every_n_steps=5),
            mlflow=None,
            tensorboard=None,
            wandb=None,
        ),
        out=tmp_path,
    )
    logger = loggers[0]
    assert isinstance(logger, JSONLLogger)
    assert logger._flush_logs_every_n_steps == 5
    assert Path(logger.log_dir) == tmp_path


def test_get_callbacks__tensorboard_user_config(tmp_path: Path) -> None:
    loggers = logger_helpers.get_loggers(
        logger_args=LoggerArgs(
            jsonl=None,
            mlflow=None,
            tensorboard=TensorBoardLoggerArgs(name="abc"),
            wandb=None,
        ),
        out=tmp_path,
    )
    logger = loggers[0]
    assert isinstance(logger, TensorBoardLogger)
    assert logger.name == "abc"
    assert Path(logger.save_dir) == tmp_path
    assert Path(logger.log_dir) == tmp_path / "abc"  # Name is used as subdirectory.


@pytest.mark.skipif(wandb is None, reason="Wandb not available")
def test_get_callbacks__wandb_user_config(tmp_path: Path) -> None:
    loggers = logger_helpers.get_loggers(
        logger_args=LoggerArgs(
            jsonl=None,
            mlflow=None,
            tensorboard=None,
            wandb=WandbLoggerArgs(project="abc"),
        ),
        out=tmp_path,
    )
    logger = loggers[0]
    assert isinstance(logger, WandbLogger)
    assert logger.name == "abc"  # WandbLogger uses the project as the name.
    assert logger.save_dir is not None
    assert Path(logger.save_dir) == tmp_path  # Cannot check for log_dir as it is None.


@pytest.mark.skipif(mlflow is None, reason="Mlflow not available")
def test_get_callbacks__mlflow_user_config(tmp_path: Path) -> None:
    loggers = logger_helpers.get_loggers(
        logger_args=LoggerArgs(
            jsonl=None,
            mlflow=MLFlowLoggerArgs(experiment_name="abc"),
            tensorboard=None,
            wandb=None,
        ),
        out=tmp_path,
    )
    logger = loggers[0]
    assert isinstance(logger, MLFlowLogger)
    temp_experiment = logger.experiment.get_experiment(logger.name)
    assert (
        temp_experiment.name == "abc"
    )  # MLFlowLogger uses the experiment id (will be 1).
    assert logger.save_dir is not None
    assert Path(logger.save_dir) == tmp_path  # Cannot check for log_dir as it is None.


def test_log_image_to_loggers__tensorboard(tmp_path: Path) -> None:
    tb_logger = TensorBoardLogger(save_dir=tmp_path)
    image = PILImageModule.new("RGB", (4, 4), color=(128, 0, 0))

    logger_helpers.log_image_to_loggers(
        loggers=[tb_logger], key="train/labels", image=image, step=7
    )
    tb_logger.save()

    acc = EventAccumulator(tb_logger.log_dir)
    acc.Reload()
    assert "train/labels" in acc.Tags()["images"]
    events = acc.Images("train/labels")
    assert len(events) == 1
    assert events[0].step == 7


def test_log_image_to_loggers__jsonl_skipped(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    jsonl_logger = JSONLLogger(save_dir=tmp_path)
    log_metrics_spy = mocker.spy(jsonl_logger, "log_metrics")
    image = PILImageModule.new("RGB", (4, 4))

    logger_helpers.log_image_to_loggers(
        loggers=[jsonl_logger], key="train/labels", image=image, step=0
    )

    log_metrics_spy.assert_not_called()


@pytest.mark.skipif(mlflow is None, reason="Mlflow not available")
def test_log_image_to_loggers__mlflow(tmp_path: Path) -> None:
    tracking_uri = (tmp_path / "mlruns").as_uri()
    mlflow_logger = MLFlowLogger(
        experiment_name="test_exp",
        tracking_uri=tracking_uri,
        save_dir=tmp_path,
    )
    image = PILImageModule.new("RGB", (4, 4), color=(128, 0, 0))

    logger_helpers.log_image_to_loggers(
        loggers=[mlflow_logger], key="train/labels", image=image, step=7
    )

    client = mlflow_logger.experiment
    png_artifacts = [
        a
        for a in client.list_artifacts(mlflow_logger.run_id, "images")
        if a.path.endswith(".png")
    ]
    assert len(png_artifacts) == 1

    local_path = client.download_artifacts(mlflow_logger.run_id, png_artifacts[0].path)
    stored = PILImageModule.open(local_path)
    assert stored.size == image.size
    assert stored.getpixel((0, 0)) == (128, 0, 0)


@pytest.mark.skipif(wandb is None, reason="Wandb not available")
def test_log_image_to_loggers__wandb(tmp_path: Path) -> None:
    wandb_logger = WandbLogger(
        save_dir=str(tmp_path),
        project="test_proj",
        name="test_run",
        offline=True,
    )
    image = PILImageModule.new("RGB", (4, 4), color=(128, 0, 0))

    try:
        logger_helpers.log_image_to_loggers(
            loggers=[wandb_logger], key="train/labels", image=image, step=7
        )
    finally:
        wandb_logger.experiment.finish()

    # Wandb offline mode writes a PNG per logged image under
    # wandb/offline-run-*/files/media/images/<key>/<filename>.png.
    pngs = list(tmp_path.rglob("media/images/train/labels_*.png"))
    assert len(pngs) == 1
    stored = PILImageModule.open(pngs[0])
    assert stored.size == image.size
    assert stored.getpixel((0, 0)) == (128, 0, 0)


def test_log_image_to_loggers__unrecognized_logger(mocker: MockerFixture) -> None:
    unknown_logger = mocker.MagicMock(spec=FabricLogger)
    image = PILImageModule.new("RGB", (4, 4))
    with pytest.raises(ValueError, match="Unrecognized logger type"):
        logger_helpers.log_image_to_loggers(
            loggers=[unknown_logger], key="train/labels", image=image, step=0
        )
