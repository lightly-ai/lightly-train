#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import time
from typing import Any, TypeVar

import torch

logger = logging.getLogger(__name__)


_T = TypeVar("_T")


def disable_compile(fn: _T, recursive: bool = True, ) -> _T:
    """Same as torch.compiler.disable but handles missing torch.compile gracefully.

    Usage:
        @_torch_compile.disable_compile
        def my_function(...):
            ...
    """
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
        return torch.compiler.disable(fn, recursive=recursive)  # type: ignore
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
        return torch._dynamo.disable(fn, recursive=recursive)  # type: ignore
    return fn


def try_compile(
    fn: _T, name: str, torch_compile_args: dict[str, Any] | None = None
) -> _T:
    logger.info(f"Compiling {name} with torch.compile")
    if not hasattr(torch, "compile"):
        return fn

    if torch_compile_args is None:
        torch_compile_args = {}  # {"dynamic": True}

    if torch_compile_args.get("disable"):
        return fn

    # try:
    #     torch._dynamo.config.capture_scalar_outputs = True
    # except AttributeError:
    #     return fn

    start_time = time.perf_counter()
    try:
        fn = torch.compile(fn, **torch_compile_args)  # type: ignore
    except Exception as ex:
        logger.warning(
            f"Compilation failed, falling back to uncompiled version. Error: {ex}"
        )
        return fn
    total_time = time.perf_counter() - start_time
    logger.info(f"Compilation completed in {total_time:.1f} seconds")
    return fn
