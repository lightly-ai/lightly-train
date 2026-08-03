#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""8-bit AdamW optimizer arguments (bitsandbytes).

Keeps the Adam ``m``/``v`` optimizer state in int8 with fp32 per-block
quantization scales (~4-8x smaller than fp32 AdamW). Requires the optional
``bitsandbytes`` dependency, imported lazily so it stays optional.
"""

from __future__ import annotations

from torch.optim.optimizer import Optimizer as TorchOptimizer

from lightly_train._optim.adamw_args import AdamWArgs
from lightly_train._optim.optimizer_type import OptimizerType
from lightly_train.types import ParamsT


class AdamW8bitArgs(AdamWArgs):
    """Arguments for the bitsandbytes 8-bit AdamW optimizer.

    Same fields/defaults as :class:`AdamWArgs`; only the constructed optimizer
    differs. Per-group keys emitted by ``get_optimizer_with_decay`` (e.g.
    ``name``, ``foreach``) are tolerated by bitsandbytes (stored as group
    attributes, ignored in ``step``), so no key stripping is needed and
    ``name`` — which DINOv2 reads in ``on_before_optimizer_step`` — is preserved.
    """

    @staticmethod
    def type() -> OptimizerType:
        return OptimizerType.ADAMW8BIT

    def get_optimizer(self, params: ParamsT, lr_scale: float) -> TorchOptimizer:
        try:
            from bitsandbytes.optim import (  # type: ignore[attr-defined]
                AdamW8bit,
            )
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "optim_type='adamw8bit' requires the 'bitsandbytes' package. "
                "Install it with `pip install bitsandbytes`."
            ) from err

        kwargs = self.model_dump()
        kwargs["lr"] *= lr_scale
        return AdamW8bit(params=params, **kwargs)  # type: ignore[no-untyped-call]
