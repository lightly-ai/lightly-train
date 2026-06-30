#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest
import torch
from torch.nn import Module

from lightly_train._task_models.ltdetr_object_detection.dinov3_vit_wrapper import (
    DINOSTAs,
)

from ...helpers import dummy_dinov3_vit_model

OLD_PREFIX = "dinov3."
NEW_PREFIX = "_model_wrapper._model."


def _remap_to_old_format(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert new-format keys to old-format keys for the backbone portion."""
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith(NEW_PREFIX):
            k = OLD_PREFIX + k[len(NEW_PREFIX) :]
        remapped[k] = v
    return remapped


@pytest.fixture
def wrapper() -> DINOSTAs:
    model_wrapper = dummy_dinov3_vit_model()
    return DINOSTAs(model_wrapper=model_wrapper)


@pytest.fixture
def fresh_wrapper() -> DINOSTAs:
    model_wrapper = dummy_dinov3_vit_model()
    return DINOSTAs(model_wrapper=model_wrapper)


class TestDINOSTAs:
    def test_load_state_dict__new_format_succeeds(
        self, wrapper: DINOSTAs, fresh_wrapper: DINOSTAs
    ) -> None:
        state_dict = wrapper.state_dict()
        fresh_wrapper.load_state_dict(state_dict)

    def test_load_state_dict__old_format_remaps_succeeds(
        self, wrapper: DINOSTAs, fresh_wrapper: DINOSTAs
    ) -> None:
        state_dict = wrapper.state_dict()
        old_state_dict = _remap_to_old_format(state_dict)
        fresh_wrapper.load_state_dict(old_state_dict)

    def test_load_state_dict__old_format_strict_false_succeeds(
        self, wrapper: DINOSTAs, fresh_wrapper: DINOSTAs
    ) -> None:
        state_dict = wrapper.state_dict()
        old_state_dict = _remap_to_old_format(state_dict)
        fresh_wrapper.load_state_dict(old_state_dict, strict=False)
        # Verify backbone weights are actually loaded, not silently skipped.
        for key, val in state_dict.items():
            assert torch.equal(fresh_wrapper.state_dict()[key], val)

    def test_load_state_dict__unrecognizable_format_raises(
        self, wrapper: DINOSTAs, fresh_wrapper: DINOSTAs
    ) -> None:
        fake_state_dict = {f"fake.{k}": v for k, v in wrapper.state_dict().items()}
        with pytest.raises(RuntimeError):
            fresh_wrapper.load_state_dict(fake_state_dict)

    def test_load_state_dict__old_format_via_parent_module_succeeds(
        self, wrapper: DINOSTAs, fresh_wrapper: DINOSTAs
    ) -> None:
        """Pre-hook must fire even when load_state_dict is called on a parent module."""

        class _Container(Module):
            def __init__(self, backbone: DINOSTAs) -> None:
                super().__init__()
                self.backbone = backbone

        container = _Container(wrapper)
        fresh_container = _Container(fresh_wrapper)

        # Build old-format full-model state dict: backbone.dinov3.* instead of
        # backbone._model_wrapper._model.*
        full_old = {}
        for k, v in container.state_dict().items():
            full_old[k.replace("backbone." + NEW_PREFIX, "backbone." + OLD_PREFIX)] = v

        # This should succeed and actually load the backbone weights.
        fresh_container.load_state_dict(full_old, strict=True)
        for key, val in container.state_dict().items():
            assert torch.equal(fresh_container.state_dict()[key], val)
