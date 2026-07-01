#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Type, TypeVar

from lightly_train._configs.config import PydanticConfig

ConfigT = TypeVar("ConfigT", bound=PydanticConfig)


@dataclass(frozen=True)
class DownloadableCheckpoint:
    url: str
    sha256: str


@dataclass(frozen=True)
class ModelAlias:
    name: str
    downloadable_checkpoint: DownloadableCheckpoint


AliasT = str | ModelAlias


class ModelRegistry(Generic[ConfigT]):
    def __init__(self) -> None:
        self._registry: dict[str, Type[ConfigT]] = {}
        self._alias_metadata: dict[str, ModelAlias] = {}

    def register(self, *aliases: AliasT) -> Callable[[Type[ConfigT]], Type[ConfigT]]:
        def decorator(cls: Type[ConfigT]) -> Type[ConfigT]:
            for alias in aliases:
                alias_name = alias.name if isinstance(alias, ModelAlias) else alias
                if alias_name in self._registry:
                    existing_cls = self._registry[alias_name].__name__
                    raise ValueError(
                        f"Conflict detected! The alias '{alias_name}' is already registered "
                        f"to the class '{existing_cls}'."
                    )
                self._registry[alias_name] = cls
                if isinstance(alias, ModelAlias):
                    self._alias_metadata[alias_name] = alias
            return cls

        return decorator

    def get(
        self,
        alias: str,
        default: Type[ConfigT] | None = None,
    ) -> Callable[[], ConfigT]:
        if alias not in self._registry:
            if default is not None:
                return default
            raise KeyError(
                f"No model configuration registered under the alias '{alias}'."
            )
        return self._registry[alias]

    def list_aliases(self) -> dict[str, str]:
        return {alias: cls.__name__ for alias, cls in self._registry.items()}

    def get_alias_metadata(self, alias: str) -> ModelAlias:
        if alias not in self._alias_metadata:
            raise KeyError(f"No metadata registered under the alias '{alias}'.")
        return self._alias_metadata[alias]

    def list_downloadable_model_url_and_hashes(self) -> dict[str, tuple[str, str]]:
        return {
            alias: (
                metadata.downloadable_checkpoint.url,
                metadata.downloadable_checkpoint.sha256,
            )
            for alias, metadata in self._alias_metadata.items()
        }
