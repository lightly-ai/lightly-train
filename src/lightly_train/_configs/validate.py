#
# Copyright (c) Lightly AG and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from pydantic_core import ErrorDetails
from typing_extensions import Any, Literal, Type, TypeVar

from lightly_train._configs import omegaconf_utils
from lightly_train._configs.config import PydanticConfig
from lightly_train.errors import (
    ConfigValidationError,
    UnresolvedAutoError,
)

_PydanticConfig = TypeVar("_PydanticConfig", bound=PydanticConfig)


def pydantic_model_validate(
    model: Type[_PydanticConfig],
    obj: dict[str, Any],
) -> _PydanticConfig:
    try:
        return model.model_validate(obj)
    except ValidationError as ex:
        errors = ex.errors()
        messages = "\n".join([f"  {_pydantic_error_msg(err)}" for err in errors])
        raise ConfigValidationError(
            f"Found {len(errors)} errors in the config!\n{messages}"
        )


def pydantic_model_merge(
    model: _PydanticConfig,
    obj: dict[str, Any],
) -> _PydanticConfig:
    """Merges the current pydantic config with overrides from a dictionary.

    NOTE(Guarin, 10/24): This is a temporary workaround until merging configs is no
    longer necessary. This function should not be used for new code.
    """
    model_dict = model.model_dump()
    # Use OmegaConf to merge because it can handle nested dictionaries. Pydantic does
    # not support merging.
    merged_dict_cfg = OmegaConf.merge(model_dict, obj)
    assert isinstance(merged_dict_cfg, DictConfig)
    merged_dict = omegaconf_utils.config_to_dict(merged_dict_cfg)
    return pydantic_model_validate(model=model.__class__, obj=merged_dict)


_T = TypeVar("_T")


def no_auto(v: _T | Literal["auto"]) -> _T:
    """Raises an error if the value is 'auto', otherwise returns the value."""
    if v == "auto":
        raise UnresolvedAutoError(
            "Got unresolved 'auto' value. This is not a user error, please report this "
            "issue to the Lightly team."
        )
    return v


def assert_config_resolved(config: PydanticConfig) -> None:
    """Asserts that a config has no unresolved 'auto' values."""
    if config.has_auto():
        raise UnresolvedAutoError(
            f"Found unresolved 'auto' values in the config '{repr(config)}'. "
            "This is not a user error, please report this issue to the Lightly team."
        )


def _pydantic_error_msg(err: ErrorDetails) -> str:
    """Converts a Pydantic error to a human readable error message.

    Pydantic error messages can be very verbose. This function tries to simplify them
    as much as possible.

    Example 1:
        Pydantic Error Message:

            Input should be a valid integer [type=int_type, input_value='1', input_type=str]
                For further information visit https://errors.pydantic.dev/2.9/v/int_type

        Function Output:

            Invalid type for key 'a.b': Input should be a valid integer but got '1' with type 'str'


    Example 2:
        Pydantic Error Message:

            a.int
                Input should be a valid integer [type=int_type, input_value=0.1, input_type=float]
                    For further information visit https://errors.pydantic.dev/2.9/v/int_type
            a.str
                Input should be a valid string [type=string_type, input_value=0.1, input_type=float]
                    For further information visit https://errors.pydantic.dev/2.9/v/string_type
            a.literal['auto']
                Input should be 'auto' [type=literal_error, input_value=0.1, input_type=float]
                    For further information visit https://errors.pydantic.dev/2.9/v/literal_error
            a.lax-or-strict[lax=union[json-or-python[json=function-after[path_validator(), str],python=is-instance[Path]],function-after[path_validator(), str]],strict=json-or-python[json=function-after[path_validator(), str],python=is-instance[Path]]]
                Input should be an instance of Path [type=is_instance_of, input_value=0.1, input_type=float]
                    For further information visit https://errors.pydantic.dev/2.9/v/is_instance_of

        Function Output:

            Invalid type for key 'a.int': Input should be a valid integer but got 0.1 with type 'float'
            Invalid type for key 'a.str': Input should be a valid string but got 0.1 with type 'float'
            Invalid type for key 'a.literal['auto']': Input should be 'auto' but got 0.1 with type 'float'
            Invalid type for key 'a.Path': Input should be an instance of Path but got 0.1 with type 'float'

    See test_validate.py:test_pydantic_model_validate__error for more example messages.
    """
    # NOTE(Guarin, 09/24): Keep this function as simple as possible. There are many
    # possible error types and special cases. We should focus on the common ones and not
    # try to cover all edge cases.
    # Possible errors are listed here: https://docs.pydantic.dev/latest/errors/validation_errors/
    type_ = err["type"]
    loc = _pydantic_loc_to_dot_sep(err["loc"])
    input_ = err["input"]
    input_type = type(input_).__name__

    if type_ == "missing":
        return f"Missing key: '{loc}'"
    elif type_ == "extra_forbidden":
        return f"Unknown key: '{loc}'"
    elif type_.endswith("_type") or type_ in ("literal_error", "is_instance_of"):
        if type_ == "is_instance_of":
            # Remove last entry from loc as it usually contains complicated types.
            # E.g. when typing 'a: int | Path' one could get:
            # loc = ('a', 'lax-or-strict[lax=union[json-or-python[json=function-after...')
            loc = _pydantic_loc_to_dot_sep(err["loc"][:-1])
            # Add class name to loc to be consistent with other error messages.
            # E.g. when typing 'a: int | Path' one could get:
            # loc = 'a.Path'
            err_class = err.get("ctx", {}).get("class")
            if err_class:
                loc += f".{err_class}"
        return (
            f"Invalid type for key '{loc}': {err['msg']} but got {repr(input_)} "
            f"with type '{input_type}'"
        )
    else:
        return (
            f"Error for key '{loc}': {err['msg']} (error_type={type_}, "
            f"input={repr(input_)}"
        )


def _pydantic_loc_to_dot_sep(loc: tuple[str | int, ...]) -> str:
    """Converts a location tuple to a dot separated string."""
    # Implementation follows:
    # https://docs.pydantic.dev/latest/errors/errors/#customize-error-messages
    path = ""
    for i, x in enumerate(loc):
        if isinstance(x, str):
            if i > 0:
                path += "."
            path += x
        elif isinstance(x, int):
            path += f"[{x}]"
        else:
            raise TypeError("Unexpected type")
    return path
