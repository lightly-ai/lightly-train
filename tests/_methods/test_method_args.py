import pytest

from lightly_train._methods import method_helpers
from lightly_train._methods.method import Method

_METHODS = method_helpers._list_methods()

@pytest.mark.parametrize("method", _METHODS)
def test_method_args__training_length_defaults(method: Method) -> None:
    method_args_cls = method_helpers.get_method_cls(method).method_args_cls()

    assert not (method_args_cls.default_steps is None and method_args_cls.default_epochs is None)
    assert not (isinstance(method_args_cls.default_steps, int) and isinstance(method_args_cls.default_epochs, int))