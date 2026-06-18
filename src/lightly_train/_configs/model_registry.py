#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Callable, Generic, Type, TypeVar

from lightly_train._configs.config import PydanticConfig

ConfigT = TypeVar("ConfigT", bound=PydanticConfig)


class ModelRegistry(Generic[ConfigT]):
    def __init__(self) -> None:
        self._registry: dict[str, Type[ConfigT]] = {}

    def register(self, *aliases: str) -> Callable[[Type[ConfigT]], Type[ConfigT]]:
        def decorator(cls: Type[ConfigT]) -> Type[ConfigT]:
            for alias in aliases:
                if alias in self._registry:
                    existing_cls = self._registry[alias].__name__
                    raise ValueError(
                        f"Conflict detected! The alias '{alias}' is already registered "
                        f"to the class '{existing_cls}'."
                    )
                self._registry[alias] = cls
            return cls

        return decorator

    def get(
        self,
        alias: str,
        default: Type[ConfigT] | None = None,
    ) -> Type[ConfigT]:
        if alias not in self._registry:
            if default is not None:
                return default
            raise KeyError(
                f"No model configuration registered under the alias '{alias}'."
            )
        return self._registry[alias]

    def list_aliases(self) -> dict[str, str]:
        return {alias: cls.__name__ for alias, cls in self._registry.items()}
