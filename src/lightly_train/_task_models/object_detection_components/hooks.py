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
"""Load-state-dict pre-hooks shared by RT-DETRv2 / D-FINE / ECSeg decoders."""

from __future__ import annotations

from typing import Any

from torch.nn import Module, ModuleList

from lightly_train._task_models import task_model_helpers


def denoising_class_embed_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Reinitialize denoising class embeddings when the number of classes changes."""
    embed_module = getattr(module, "denoising_class_embed", None)
    if embed_module is None:
        return
    task_model_helpers.reinit_class_heads_on_mismatch(
        state_dict=state_dict,
        prefix=prefix,
        heads={"denoising_class_embed": embed_module},
    )


def score_head_reuse_or_reinit_hook(
    module: Module,
    state_dict: dict[str, Any],
    prefix: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Reinitialize encoder and decoder score heads when the number of classes
    changes.
    """
    heads: dict[str, Module] = {}
    for name in ["enc_score_head", "dec_score_head"]:
        score_head = getattr(module, name, None)
        if isinstance(score_head, ModuleList):
            heads.update({f"{name}.{i}": head for i, head in enumerate(score_head)})
        elif score_head is not None:
            heads[name] = score_head
    # Deploy mode replaces unused heads with nn.Identity.
    heads = {name: head for name, head in heads.items() if hasattr(head, "weight")}
    task_model_helpers.reinit_class_heads_on_mismatch(
        state_dict=state_dict, prefix=prefix, heads=heads
    )
