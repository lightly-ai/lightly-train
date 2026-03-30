#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import contextlib
import os
from typing import Any, Callable, Generator

import torch
from torch import Tensor
from torch.nn import Module


# TODO(Guarin, 12/25): When you remove this context manager, also remove
# the corresponding weights_only warning in _warnings.py
@contextlib.contextmanager
def _torch_weights_only_false() -> Generator[None, None, None]:
    """All torch.load calls within this context will run with weights_only=False."""
    previous_state = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    try:
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        yield
    finally:
        if previous_state is not None:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_state
        else:
            del os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]


def register_load_state_dict_pre_hook(
    module: Module,
    hook: Callable[[Module, dict[str, Any], str, Any, Any], None],
) -> None:
    """Registers a load_state_dict pre-hook on the module.

    Handles backwards compatibility for PyTorch <= 2.4.

    Args:
        module: The module to register the hook on.
        hook:
            The hook function to register. Should have the signature:
            hook(module: Module, state_dict: dict[str, Any], prefix: str, *args: Any, **kwargs: Any) -> None
    """
    if hasattr(module, "register_load_state_dict_pre_hook"):
        module.register_load_state_dict_pre_hook(hook)  # type: ignore[no-untyped-call]
    else:
        # Backwards compatibility for PyTorch <= 2.4
        module._register_load_state_dict_pre_hook(hook, with_module=True)  # type: ignore[no-untyped-call]


def set_warn_on_accumulate_grad_stream_mismatch(value: bool) -> None:
    # Avoids the following warning when using DDP:
    #
    # UserWarning: The AccumulateGrad node's stream does not match the stream of the
    # node that produced the incoming gradient. This may incur unnecessary
    # synchronization and break CUDA graph capture if the AccumulateGrad node's stream
    # is the default stream. This mismatch is caused by an AccumulateGrad node created
    # prior to the current iteration being kept alive. This can happen if the autograd
    # graph is still being kept alive by tensors such as the loss, or if you are using
    # DDP, which will stash a reference to the node. To resolve the mismatch, delete all
    # references to the autograd graph or ensure that DDP initialization is performed
    # under the same stream as subsequent forwards. If the mismatch is intentional, you
    # can use torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False) to
    # suppress this warning.
    if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(value)  # type: ignore


@torch.no_grad()
def update_ema_tensors(
    tensors: list[Tensor],
    tensors_ema: list[Tensor],
    m: float,
) -> None:
    """Updates tensors with an exponential moving average using foreach ops."""
    if not tensors_ema:
        return

    torch._foreach_mul_(tensors_ema, m)
    torch._foreach_add_(tensors_ema, tensors, alpha=1.0 - m)


@torch.no_grad()
def update_momentum(model: Module, model_ema: Module, m: float) -> None:
    """Updates parameters of `model_ema` with the EMA of `model`."""
    update_ema_tensors(
        tensors=list(model.parameters()),
        tensors_ema=list(model_ema.parameters()),
        m=m,
    )
