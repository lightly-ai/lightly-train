#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

from typing_extensions import Self

_T = TypeVar("_T")


class InfiniteCycleIterator(Generic[_T]):
    def __init__(self, iterable: Iterable[_T]):
        self.iterable = iterable
        self.cycles = 0
        self._iter: Iterator[_T] | None = None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> _T:
        if self._iter is None:
            self._iter = iter(self.iterable)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.iterable)
            self.cycles += 1
            return next(self._iter)
