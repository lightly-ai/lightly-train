#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

import pytest

from lightly_train._models import model_wrapper

from ..helpers import DummyCustomModel


@pytest.mark.parametrize(
    "exclude_module_attrs",
    [False, True],
)
def test_missing_model_wrapper_attrs(exclude_module_attrs: bool) -> None:
    assert not model_wrapper.missing_model_wrapper_attrs(
        DummyCustomModel(), exclude_module_attrs=exclude_module_attrs
    )


@pytest.mark.parametrize(
    "exclude_module_attrs, expected",
    [
        (True, ["feature_dim", "forward_pool"]),
        (
            False,
            [
                "T_destination",
                "feature_dim",
                "forward_pool",
                "load_state_dict",
                "parameters",
                "state_dict",
            ],
        ),
    ],
)
def test_missing_model_wrapper_attrs__missing(
    exclude_module_attrs: bool, expected: list[str]
) -> None:
    class InvalidCustomModelWrapper:
        def get_model(self) -> None:
            pass

        def forward_features(self) -> None:
            pass

    assert (
        model_wrapper.missing_model_wrapper_attrs(
            InvalidCustomModelWrapper(),
            exclude_module_attrs=exclude_module_attrs,
        )
        == expected
    )
