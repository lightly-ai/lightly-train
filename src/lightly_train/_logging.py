#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import copy
import logging
import re
from logging import Filter, Logger, LogRecord
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

from pytorch_lightning.utilities import rank_zero_only

from lightly_train._env import Env

# Capture and send warnings to the `py.warnings`.
# Keep this import at the top to ensure that warnings are captured from the beginning.
# Users can disable this behavior by setting `capture=False` when calling `logging.captureWarnings`.
logging.captureWarnings(capture=True)

# Set up the logger for the lightly_train package.
lightly_logger = logging.getLogger("lightly_train")
lightly_logger.setLevel(logging.DEBUG)


class ConsoleFormatter(logging.Formatter):
    """Custom formatter for console logging.

    This formatter uses ANSI escape codes to color log messages based on their level.
    * DEBUG     No color
    * INFO      No color
    * WARNING   Yellow
    * ERROR     Red
    * CRITICAL  Red
    The reset code is appended to each message to ensure that the color does not leak.

    The formatter does not print timestamps or log levels so as to not clutter the console.
    """

    reset = "\x1b[0m"
    log_entry_structure = "%(message)s"

    FORMATS = {
        logging.DEBUG: "\033[1;34m[debug] " + log_entry_structure + reset,
        logging.INFO: "" + log_entry_structure + reset,
        logging.WARNING: "\033[93m" + log_entry_structure + reset,
        logging.ERROR: "\033[91m" + log_entry_structure + reset,
        logging.CRITICAL: "\033[91m" + log_entry_structure + reset,
    }

    def __init__(self) -> None:
        self.formatters = {
            level: logging.Formatter(level_format)
            for level, level_format in ConsoleFormatter.FORMATS.items()
        }
        self.default_formatter = logging.Formatter(fmt=self.log_entry_structure)

    def format(self, record: logging.LogRecord) -> str:
        new_record = copy.copy(record)
        formatter = self.formatters.get(new_record.levelno, self.default_formatter)
        return formatter.format(new_record)


@rank_zero_only  # type: ignore[misc]
def set_up_console_logging() -> None:
    """Sets up console logging and ensures a single handler per console logger."""
    level = Env.LIGHTLY_TRAIN_LOG_LEVEL.value
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(ConsoleFormatter())
    _set_console_handler(ch)


def _set_console_handler(handler: "logging.StreamHandler[TextIO]") -> None:
    """Sets this handler as the only handler printing to the console.

    Removes all existing stream handlers.
    """
    console_loggers = [
        lightly_logger,
        logging.getLogger("pytorch_lightning"),
        logging.getLogger("torch"),
        logging.getLogger("py.warnings"),
    ]

    for console_logger in console_loggers:
        # Remove any existing handlers that could print to the console.
        _remove_handlers(console_logger, logging.StreamHandler)
        console_logger.addHandler(handler)


def _remove_handlers(
    logger: logging.Logger,
    handler_cls_to_remove: type[logging.Handler] = logging.Handler,
) -> None:
    """Removes all handlers of the given class from the logger."""
    new_handlers = []
    for handler in logger.handlers:
        # NullHandler should never be removed as it prevents messages from being
        # printed to stderr. See https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
        if isinstance(handler, logging.NullHandler) or not isinstance(
            handler, handler_cls_to_remove
        ):
            new_handlers.append(handler)
    logger.handlers = new_handlers


class FileFormatter(logging.Formatter):
    """Custom formatter for logging to a file.

    This formatter prints timestamps, log levels, and additional information such as
    the name of the logger, the source file, the line number, and the function name.

    """

    def __init__(self) -> None:
        self.info_formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s"
        )
        self.error_formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s\n%(exc_info)s"
        )

    def format(self, record: logging.LogRecord) -> str:
        if record.exc_info is None:
            return self.info_formatter.format(record)
        else:
            return self.error_formatter.format(record)


class LightlyTrainRotatingFileHandler(RotatingFileHandler):
    """Custom rotating file handler for logging."""

    pass


@rank_zero_only  # type: ignore[misc]
def set_up_file_logging(log_file_path: Path) -> None:
    """Sets up logging to a file by.

    Args:
        log_file_path:
            Path to the log file.
    """
    fh = _get_file_handler(log_file_path)
    file_loggers = [
        lightly_logger,
        logging.getLogger("pytorch_lightning"),
        logging.getLogger("torch"),
        logging.getLogger("py.warnings"),
    ]

    for file_logger in file_loggers:
        _remove_handlers(file_logger, LightlyTrainRotatingFileHandler)
        file_logger.addHandler(fh)


def _get_file_handler(log_file_path: Path) -> LightlyTrainRotatingFileHandler:
    fh = LightlyTrainRotatingFileHandler(
        str(log_file_path),
        mode="a",
        maxBytes=1024 * 1024 * 1024,
        backupCount=4,
        encoding=None,
        delay=False,
    )
    fh.setFormatter(FileFormatter())
    return fh


class RegexFilter(Filter):
    """Filter to exclude messages based on a regex pattern."""

    def __init__(self, pattern: str, name: str = "") -> None:
        super().__init__(name)
        self.regex = re.compile(pattern)

    def filter(self, record: LogRecord) -> bool:
        return not self.regex.search(record.getMessage())


@rank_zero_only  # type: ignore[misc]
def set_up_filters() -> None:
    """Sets up filters to exclude specific log messages."""
    lightning_logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
    _remove_filters(lightning_logger)

    # Ignore torch.set_float32_matmul_precision logs as we handle this in our code.
    lightning_logger.addFilter(
        RegexFilter(
            r"To properly utilize them, you should set "
            r"`torch.set_float32_matmul_precision\('medium' \| 'high'\)` which will "
            r"trade-off precision for performance"
        )
    )


def _remove_filters(logger: Logger) -> None:
    """Removes all filters from the logger."""
    for filter in logger.filters:
        if isinstance(filter, RegexFilter):
            logger.removeFilter(filter)
