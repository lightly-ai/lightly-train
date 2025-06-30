#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os
from pathlib import Path

from pytest import LogCaptureFixture
from pytest_mock import MockerFixture
from pytorch_lightning.utilities import rank_zero_info

from lightly_train import _logging
from lightly_train._logging import RegexFilter


def test_set_up_console_logging() -> None:
    _logging.set_up_console_logging()
    _logging.set_up_console_logging()
    # Should only have a single console handler even after multiple calls
    # to set up console logging.
    logger = logging.getLogger("lightly_train")
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == logging.INFO


def test_set_up_console_logging__custom_log_level(mocker: MockerFixture) -> None:
    mocker.patch.dict(os.environ, {"LIGHTLY_TRAIN_LOG_LEVEL": "WARNING"})
    _logging.set_up_console_logging()
    logger = logging.getLogger("lightly_train")
    assert len(logger.handlers) == 1
    assert logger.handlers[0].level == logging.WARNING


def test__set_console_handler() -> None:
    lightly_logger = logging.getLogger("lightly_train")
    lightly_logger.addHandler(logging.StreamHandler())
    lightning_logger = logging.getLogger("pytorch_lightning")
    lightning_logger.addHandler(logging.StreamHandler())
    torch_logger = logging.getLogger("torch")
    torch_logger.addHandler(logging.StreamHandler())
    new_handler = logging.StreamHandler()
    # Should remove the existing handler and add the new handler.
    _logging._set_console_handler(new_handler)
    assert len(lightly_logger.handlers) == 1
    assert lightly_logger.handlers[0] == new_handler
    assert len(lightning_logger.handlers) == 1
    assert lightning_logger.handlers[0] == new_handler
    assert len(torch_logger.handlers) == 1
    assert torch_logger.handlers[0] == new_handler


def test__remove_handlers() -> None:
    logger = logging.getLogger("test_remove_handlers")
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.StreamHandler())
    assert len(logger.handlers) == 2
    _logging._remove_handlers(logger)
    assert len(logger.handlers) == 0


def test__remove_handlers_by_type() -> None:
    logger = logging.getLogger("test_remove_handlers_by_type")
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.NullHandler())
    logger.addHandler(logging.Handler())
    assert len(logger.handlers) == 3
    _logging._remove_handlers(logger, logging.StreamHandler)
    assert len(logger.handlers) == 2
    for handler in logger.handlers:
        assert not isinstance(handler, logging.StreamHandler)


def test_set_up_file_logging(tmp_path: Path) -> None:
    _logging.set_up_file_logging(log_file_path=tmp_path / "test.log")
    logging.getLogger("lightly_train").debug("debug message")
    logging.getLogger("lightly_train").info("info message")
    logging.getLogger("lightly_train").warning("warning message")
    logging.getLogger("lightly_train").error("error message")
    logging.getLogger("lightly_train").critical("critical message")
    logs = (tmp_path / "test.log").read_text()
    assert "debug message" in logs
    assert "info message" in logs
    assert "warning message" in logs
    assert "error message" in logs
    assert "critical message" in logs


class TestRegexFilter:
    def test(self, caplog: LogCaptureFixture) -> None:
        logger = logging.getLogger("test_regex_filter")
        logger.addFilter(RegexFilter("my test message"))
        with caplog.at_level(logging.DEBUG):
            logger.debug("abc my test message abc")
            logger.debug("other message")
        assert "my test message" not in caplog.text
        assert "other message" in caplog.text


def test_set_up_filters(caplog: LogCaptureFixture) -> None:
    _logging.set_up_filters()
    with caplog.at_level(logging.DEBUG):
        rank_zero_info(
            "You are using a CUDA device ('NVIDIA GeForce RTX 4090') that has Tensor "
            "Cores. To properly utilize them, you should set "
            "`torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off "
            "precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"
        )
    assert "You are using a CUDA device" not in caplog.text
