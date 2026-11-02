#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import logging
import sys
import time
from types import MethodType
from typing import Any, Callable, TypeVar

import torch
from pydantic import ConfigDict

from lightly_train import _distributed
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
# See this discussion for more details: https://github.com/Lightning-AI/pytorch-lightning/discussions/21569


class TorchCompileArgs(PydanticConfig):
    # Disable torch.compile for all models by default as most models do not support
    # compilation out of the box.
    # See ImageClassificationTrainModel on how to enable it for a specific model.
    disable: bool = True

    # Compilation mode passed to torch.compile, e.g. "default", "reduce-overhead"
    # or "max-autotune". Can't be combined with options.
    mode: str | None = None

    # Backend options passed to torch.compile. Can't be combined with mode.
    options: dict[str, Any] | None = None

    # Allow extra fields as torch.compile accepts many arguments
    model_config = ConfigDict(extra="allow")


def try_compile(fn: _T, name: str, torch_compile_args: TorchCompileArgs) -> _T:
    """Tries to compile a function and falls back to the uncompiled version if a
    compilation error occurs."""
    if torch_compile_args.disable:
        return fn
    if not hasattr(torch, "compile"):
        return fn
    if sys.platform.startswith("win"):
        # Compilation is currently unstable on Windows.
        return fn

    logger.info(f"Compiling {name} with torch.compile")
    start_time = time.perf_counter()
    try:
        compiled_fn = torch.compile(fn, **torch_compile_args.model_dump())  # type: ignore
    except Exception as ex:
        logger.warning(
            f"Compilation failed for {name}, falling back to eager. Error: {ex}"
        )
        return fn
    total_time = time.perf_counter() - start_time
    logger.info(f"Compilation completed in {total_time:.1f} seconds")

    return _CompiledOrEager(fn, compiled_fn, name)  # type: ignore[arg-type, return-value]


def _return_fn(fn: Callable[..., Any]) -> Callable[..., Any]:
    return fn


class _CompiledOrEager:
    """Calls the compiled function and switches to eager on a compilation error.

    Pickles and copies as the original function.
    """

    def __init__(
        self, fn: Callable[..., Any], compiled_fn: Callable[..., Any], name: str
    ) -> None:
        self._fn = fn
        self._compiled_fn = compiled_fn
        self._name = name
        self._compile_failed = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from torch._dynamo.exc import TorchDynamoException

        if self._compile_failed:
            return self._fn(*args, **kwargs)
        try:
            return self._compiled_fn(*args, **kwargs)
        # torch.compile is lazy, so compilation errors are raised on call.
        except TorchDynamoException as ex:
            self._compile_failed = True
            logger.warning(
                f"Compilation failed for {self._name} on rank "
                f"{_distributed.get_global_rank()}, falling back to eager for "
                f"subsequent calls. Error: {ex}"
            )
            return self._fn(*args, **kwargs)

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        # Binds self like a function when assigned to a class.
        return self if obj is None else MethodType(self, obj)

    def __reduce__(self) -> tuple[Any, ...]:
        return _return_fn, (self._fn,)
