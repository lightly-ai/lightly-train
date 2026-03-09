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
from typing import TypeVar

import torch
from pydantic import ConfigDict

from lightly_train._configs.config import PydanticConfig

logger = logging.getLogger(__name__)


_T = TypeVar("_T")


def disable_compile(fn: _T, recursive: bool = True) -> _T:
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


# ## How to enable torch.compile for a model
#
# 1. Create a <ModelName>TorchCompileArgs class that inherits from TorchCompileArgs
#    in the train_model.py file
# 2. Set disable: bool = False in the new class
# 3. Set TrainModel.torch_compile_args_cls to the new class
# 4. Implement TrainModel.forward as this is the only method that will be compiled
#
# Compilation is disabled by default as most models do not yet support it.
#
# training_step and validation_step are not compiled as they contain a lot of logic that
# is not supported by torch.compile. For example, most batches contain the image
# filenames which triggers a recompilation for every new filename. Metrics don't support
# compilation either and are also called inside the train/val steps. Therefore it is
# easier to move all the compilable code to the forward method and call it there.


class TorchCompileArgs(PydanticConfig):
    # Disable torch.compile for all models by default as most models do not support
    # compilation out of the box.
    # See ImageClassificationTrainModel on how to enable it for a specific model.
    disable: bool = True

    # Allow extra fields as torch.compile accepts many arguments
    model_config = ConfigDict(extra="allow")


def try_compile(fn: _T, name: str, torch_compile_args: TorchCompileArgs) -> _T:
    """Tries to compile a torch model or function and falls back to the uncompiled
    version if any error occurs."""
    if torch_compile_args.disable:
        return fn
    if not hasattr(torch, "compile"):
        return fn

    logger.info(f"Compiling {name} with torch.compile")
    start_time = time.perf_counter()
    try:
        fn = torch.compile(fn, **torch_compile_args.model_dump())  # type: ignore
    except Exception as ex:
        logger.warning(
            f"Compilation failed, falling back to uncompiled version. Error: {ex}"
        )
        return fn
    total_time = time.perf_counter() - start_time
    logger.info(f"Compilation completed in {total_time:.1f} seconds")
    return fn
