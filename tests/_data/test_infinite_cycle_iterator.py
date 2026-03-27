#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from lightly_train._data.infinite_cycle_iterator import InfiniteCycleIterator


def test_reset() -> None:
    iterator = InfiniteCycleIterator([1, 2])

    assert next(iterator) == 1

    iterator.reset()

    assert iterator.cycles == 0
    assert next(iterator) == 1
    assert next(iterator) == 2
    assert iterator.cycles == 0

    assert next(iterator) == 1
    assert iterator.cycles == 1
