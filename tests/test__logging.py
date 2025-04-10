#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os
import pathlib
import tempfile
from pathlib import Path

from lightly_train import _logging


def test_set_up_console_logging() -> None:
    _logging.set_up_console_logging()
    _logging.set_up_console_logging()
    # Should only have a single console handler even after multiple calls
    # to set up console logging.
    logger = logging.getLogger("lightly_train")
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_set_up_console_logging__custom_log_level() -> None:
    os.environ[_logging.LIGHTLY_TRAIN_LOG_LEVEL_ENV_VAR] = str(logging.WARNING)
    _logging.set_up_console_logging()
    logger = logging.getLogger("lightly_train")
    assert len(logger.handlers) == 1
    assert logger.handlers[0].level == logging.WARNING
    os.environ.pop(_logging.LIGHTLY_TRAIN_LOG_LEVEL_ENV_VAR)


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


def test_set_up_file_logging() -> None:
    with tempfile.NamedTemporaryFile() as file:
        _logging.set_up_file_logging(log_file_path=Path(file.name))
        logging.getLogger("lightly_train").debug("debug message")
        logging.getLogger("lightly_train").info("info message")
        logging.getLogger("lightly_train").warning("warning message")
        logging.getLogger("lightly_train").error("error message")
        logging.getLogger("lightly_train").critical("critical message")
        logs = pathlib.Path(file.name).read_text()
        assert "debug message" in logs
        assert "info message" in logs
        assert "warning message" in logs
        assert "error message" in logs
        assert "critical message" in logs
