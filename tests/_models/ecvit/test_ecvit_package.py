#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest

from lightly_train._models.ecvit import ECVIT_PRESETS
from lightly_train._models.ecvit.ecvit_package import (
    EDGE_CRAFTER_PACKAGE,
    MODEL_NAME_TO_INFO,
)


def test_model_info_presets_match_ecvit_presets() -> None:
    preset_names = {
        model_info["preset_name"] for model_info in MODEL_NAME_TO_INFO.values()
    }

    assert preset_names == set(ECVIT_PRESETS)


@pytest.mark.parametrize(
    "num_input_channels, model_args, expected",
    [
        (4, None, 4),
        (3, {"in_chans": 4}, 4),
        (4, {"in_chans": 1}, 1),
    ],
)
def test_get_model__input_channels(
    num_input_channels: int, model_args: dict[str, int] | None, expected: int
) -> None:
    model = EDGE_CRAFTER_PACKAGE.get_model(
        model_name="_ecvittest-notpretrained",
        num_input_channels=num_input_channels,
        model_args=model_args,
        load_weights=False,
    )
    assert model.num_input_channels == expected
    assert model.backbone.patch_embed.in_chans == expected
