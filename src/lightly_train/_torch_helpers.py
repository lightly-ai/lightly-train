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
