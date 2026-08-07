#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Callable, TypeVar, cast

import torch
from pydantic import Field

from lightly_train._configs.config import PydanticConfig

_T = TypeVar("_T")


class ActivationCheckpointingArgs(PydanticConfig):
    """Configuration for activation checkpointing (gradient checkpointing).

    Activation checkpointing trades compute for memory by discarding intermediate
    activations during the forward pass and recomputing them during the backward
    pass. This can significantly reduce peak GPU memory usage when pretraining
    large vision transformer backbones.

    Attributes:
        enabled:
            Whether to enable activation checkpointing.
        every_n_blocks:
            Apply checkpointing every N blocks. 1 means all blocks are checkpointed.
            Backbones with chunked blocks (DINOv2 with ``block_chunks > 0``, used by
            ViT-L/14 and ViT-g/14) checkpoint whole chunks, so there ``every_n_blocks``
            counts chunks and the effective granularity is
            ``every_n_blocks * chunksize`` blocks.
    """

    enabled: bool = False
    every_n_blocks: int = Field(default=1, ge=1)


def maybe_checkpoint(
    block: Callable[..., _T],
    *args: Any,
    use_activation_checkpointing: bool,
    block_index: int,
    every_n_blocks: int,
    **kwargs: Any,
) -> _T:
    """Conditionally apply activation checkpointing to a block.

    When checkpointing is enabled and the block index is a multiple of
    ``every_n_blocks``, the block's forward pass is wrapped with
    ``torch.utils.checkpoint.checkpoint``. Otherwise, the block is called
    directly.

    Args:
        block: The module or callable to execute.
        *args: Positional arguments forwarded to ``block``.
        use_activation_checkpointing: Whether checkpointing is enabled.
        block_index: Zero-based index of the current block.
        every_n_blocks: Apply checkpointing every N blocks.
        **kwargs: Keyword arguments forwarded to ``block``.
    """
    if use_activation_checkpointing and block_index % every_n_blocks == 0:
        return cast(
            _T,
            torch.utils.checkpoint.checkpoint(
                block, *args, use_reentrant=False, **kwargs
            ),
        )
    return block(*args, **kwargs)
