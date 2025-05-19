#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import os

from lightly_train import _lightning_rank_zero

get_global_rank = _lightning_rank_zero.get_global_rank


def get_local_rank() -> int | None:
    """Get the local rank of the current process."""
    rank_keys = ("LOCAL_RANK", "SLURM_LOCALID", "JSM_NAMESPACE_LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return None


def get_node_rank() -> int | None:
    """Get the node rank of the current process."""
    rank_keys = ("NODE_RANK", "GROUP_RANK", "SLURM_NODEID")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return None


def is_global_rank_zero() -> bool:
    """Check if the current process is running on the global rank zero."""
    global_rank = get_global_rank()
    # Check node rank because process might be assigned to a node but not yet
    # a global rank.
    return global_rank == 0 or (global_rank is None and is_node_rank_zero())


def is_local_rank_zero() -> bool:
    """Check if the current process is running on the local rank zero."""
    local_rank = get_local_rank()
    return local_rank == 0 or local_rank is None


def is_node_rank_zero() -> bool:
    """Check if the current process is running on the node rank zero."""
    node_rank = get_node_rank()
    return node_rank == 0 or node_rank is None
