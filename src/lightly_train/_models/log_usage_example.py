#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from os import linesep


def format_log_msg_model_usage_example(log_message_code_block: list[str]) -> str:
    log_message_header = (
        f"Example: How to use the exported model{linesep}{'-' * 88}{linesep}"
    )

    log_message_footer = f"{'-' * 88}{linesep}"

    def format_code_lines(lines: list[str]) -> str:
        str_out = ""
        for line in lines:
            str_out += f"{line}{linesep}"
        return str_out

    log_message = (
        log_message_header
        + format_code_lines(log_message_code_block)
        + log_message_footer
    )

    return log_message
