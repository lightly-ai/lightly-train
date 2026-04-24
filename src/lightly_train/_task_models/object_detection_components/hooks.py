#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
"""Load-state-dict pre-hooks shared by RT-DETRv2 / D-FINE decoders.

These hooks adapt a pretrained checkpoint to a module configured with a
different number of classes, by truncating or zero-/re-initializing the
classification-related weights so ``load_state_dict`` succeeds.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch
from torch.nn import Module, ModuleList

logger = logging.getLogger(__name__)


def denoising_class_embed_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Adjust denoising class embeddings when checkpoint has different number of classes.

    If the checkpoint and module have different numbers of classes, this hook reuses
    available weights and initializes missing ones from the module's initialization.
    This allows loading checkpoints trained on different datasets.

    Args:
        module: The module being loaded.
        state_dict: The checkpoint state dictionary.
        prefix: Prefix for parameter names in state_dict.
    """
    weight_key = f"{prefix}denoising_class_embed.weight"
    checkpoint_weight = state_dict.get(weight_key)
    if checkpoint_weight is None:
        return

    embed_module = getattr(module, "denoising_class_embed", None)
    if embed_module is None:
        return

    num_classes_checkpoint = checkpoint_weight.shape[0]
    num_classes_module = embed_module.num_embeddings
    if num_classes_checkpoint == num_classes_module:
        return

    logger.info(
        f"Checkpoint has {num_classes_checkpoint - 1} classes, module expects "
        f"{num_classes_module - 1} classes. Adjusting denoising class embeddings."
    )

    device = embed_module.weight.device

    # Last class is padding_idx
    num_user_classes_checkpoint = num_classes_checkpoint - 1
    num_user_classes_module = num_classes_module - 1

    if num_classes_checkpoint > num_classes_module:
        # Checkpoint has more classes: reuse checkpoint and discard excess
        adjusted_weight = torch.cat(
            [
                checkpoint_weight[:num_user_classes_module].to(device),
                checkpoint_weight[-1:].to(device),  # padding class
            ],
            dim=0,
        )
    else:
        # Checkpoint has fewer classes: reuse checkpoint and initialize missing from
        # module
        adjusted_weight = torch.cat(
            [
                checkpoint_weight[:num_user_classes_checkpoint].to(device),
                embed_module.weight[num_user_classes_checkpoint:].detach().clone(),  # type: ignore[index]
            ],
            dim=0,
        )

    state_dict[weight_key] = adjusted_weight


def score_head_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    _score_head_reuse_or_reinit_hook(
        module,
        state_dict,
        prefix,
        enc_or_dec="enc",
    )
    _score_head_reuse_or_reinit_hook(
        module,
        state_dict,
        prefix,
        enc_or_dec="dec",
    )


def _score_head_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    enc_or_dec: Literal["enc", "dec"],
) -> None:
    """Adjust score head weights when checkpoint has different number of classes.

    Handles both single score head (e.g., encoder) and multiple score heads (e.g., decoder layers).

    Args:
        module: The module being loaded.
        state_dict: The checkpoint state dictionary.
        prefix: Prefix for parameter names in state_dict.
        enc_or_dec: Whether this is for encoder ("enc") or decoder ("dec") score head.
    """
    module_name = f"{enc_or_dec}_score_head"
    score_head = getattr(module, module_name, None)
    if score_head is None:
        return

    # Handle both single head and multiple heads (ModuleList)
    heads_to_process = (
        enumerate(score_head)
        if isinstance(score_head, ModuleList)
        else [(None, score_head)]
    )

    any_adjusted = False
    for idx, head_module in heads_to_process:
        # Construct parameter keys based on whether this is a list or single head
        if idx is not None:
            weight_key = f"{prefix}{module_name}.{idx}.weight"
            bias_key = f"{prefix}{module_name}.{idx}.bias"
        else:
            weight_key = f"{prefix}{module_name}.weight"
            bias_key = f"{prefix}{module_name}.bias"

        was_adjusted = _reuse_or_reinit(
            head_module, state_dict, weight_key=weight_key, bias_key=bias_key
        )
        any_adjusted = any_adjusted or was_adjusted

    if any_adjusted:
        logger.info(
            f"Checkpoint has different number of classes for {module_name}. "
            f"Adjusted weights/biases to match module configuration."
        )


def _reuse_or_reinit(
    head_module: Module,
    state_dict: dict[str, Any],
    *,
    weight_key: str,
    bias_key: str,
) -> bool:
    """Adjust linear head weights/biases when checkpoint has different number of classes.

    Enables loading checkpoints trained on different number of classes by either:
    - Truncating weights if checkpoint has more classes (excess classes discarded)
    - Padding weights if checkpoint has fewer classes (new classes initialized from module)

    Args:
        head_module: The linear classification head module.
        state_dict: The checkpoint state dictionary.
        weight_key: Key to the weight parameter in state_dict.
        bias_key: Key to the bias parameter in state_dict.

    Returns:
        True if weights/biases were adjusted, False otherwise.
    """
    checkpoint_weight = state_dict.get(weight_key)
    checkpoint_bias = state_dict.get(bias_key)
    if checkpoint_weight is None:
        return False

    num_classes_checkpoint = checkpoint_weight.shape[0]
    num_classes_module = getattr(head_module, "out_features", None)
    if num_classes_module is None or num_classes_checkpoint == num_classes_module:
        return False

    device = head_module.weight.device

    if num_classes_checkpoint > num_classes_module:
        # Checkpoint has more classes: truncate to module's expected size
        adjusted_weights = checkpoint_weight[:num_classes_module, :]
        if checkpoint_bias is not None:
            adjusted_biases = checkpoint_bias[:num_classes_module]
    else:
        # Checkpoint has fewer classes: pad with module's initialized weights
        adjusted_weights = torch.cat(
            [
                checkpoint_weight.to(device),
                head_module.weight[num_classes_checkpoint:].detach().clone(),  # type: ignore[index]
            ],
            dim=0,
        )
        if checkpoint_bias is not None:
            adjusted_biases = torch.cat(
                [
                    checkpoint_bias.to(device),
                    head_module.bias[num_classes_checkpoint:].detach().clone(),  # type: ignore[index]
                ],
                dim=0,
            )

    state_dict[weight_key] = adjusted_weights
    if checkpoint_bias is not None:
        state_dict[bias_key] = adjusted_biases
    return True
