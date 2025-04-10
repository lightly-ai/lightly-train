#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict


class PydanticConfig(BaseModel):
    """Base class for all configs."""

    # Settings
    model_config = ConfigDict(
        extra="forbid",  # Do not allow extra attributes.
        frozen=False,  # Allow updating attributes. This is needed for field resolving.
        strict=True,  # Strict validation.
        validate_assignment=True,  # Run validation on assignment.
        validate_default=True,  # Validate default fields.
        # By default, pydantic shows an error message if a field name starts with
        # 'model_' to prevent users from accidentally overriding internal model
        # attributes such as 'model_config'. We disable this because we have fields
        # named 'model' and 'model_args'. We modify the protected_namespaces to
        # explicitly exclude all predefined model attributes instead.
        protected_namespaces=tuple(
            name for name in dir(BaseModel) if name.startswith("model_")
        ),
    )

    def model_dump(
        self, serialize_as_any: bool = True, **kwargs: Any
    ) -> dict[str, Any]:
        """Serialize with duck typing

        See https://docs.pydantic.dev/latest/concepts/serialization/#overriding-the-serialize_as_any-default-false
        """
        return super().model_dump(serialize_as_any=serialize_as_any, **kwargs)

    def model_dump_json(self, serialize_as_any: bool = True, **kwargs: Any) -> str:
        """Serialize with duck typing

        See https://docs.pydantic.dev/latest/concepts/serialization/#overriding-the-serialize_as_any-default-false
        """
        return super().model_dump_json(serialize_as_any=serialize_as_any, **kwargs)

    def has_auto(self) -> bool:
        """Check if the config or any of its children has a field with value 'auto'.

        Args:
            config:
                The config to check.

        Returns:
            True if the config or any of its children has a field with value 'auto'.

        """
        return _has_auto(self)


def _has_auto(config: PydanticConfig | dict[str, Any]) -> bool:
    values: Iterable[Any]
    if isinstance(config, PydanticConfig):
        values = [getattr(config, field) for field in config.model_fields]
    else:
        values = config.values()

    for value in values:
        if value == "auto":
            return True
        elif isinstance(value, (PydanticConfig, dict)):
            if _has_auto(value):
                return True
    return False
