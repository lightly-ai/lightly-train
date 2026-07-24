#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._models.ecvit import ECVIT_PRESETS
from lightly_train._models.ecvit.ecvit_package import MODEL_NAME_TO_INFO


def test_model_info_presets_match_ecvit_presets() -> None:
    preset_names = {
        model_info["preset_name"] for model_info in MODEL_NAME_TO_INFO.values()
    }

    assert preset_names == set(ECVIT_PRESETS)
