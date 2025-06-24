# Python coding guidelines

## Imports

When importing:

- For classes, import classes directly.
- For functions, import the containing module and call the function using dot notation.
- Don't do imports that are not used in the file. If you accidentally import something that is not used, remove it.
- Avoid wildcard imports, e.g. `from module import *`.
- Sort imports alphabetically and also in the following order:
  - Standard library imports
  - Third-party imports
  - Local imports
    and group them accordingly with a blank line between each group.

## File Layout

Inside each file, the order of classes and functions should be:

- public classes
- public functions
- private classes
- private functions.

Classes should come before global functions, if they are present.

## Protocols and Abstract Base Classes

- DON'T inherit from Protocols.
- For Protocol names, use ...able or a general class name.
  ```python
  class JsonSerializable(Protocol):
      def serialize() -> Json:
          ...

  class Datasource(Protocol):
      def get_list_access() -> bool:
          ...
          def get_write_access() -> bool:
                  ...
  ```
- DON'T implement methods inside Protocols unless you have a strong reason to do so.
- DON'T mix ABCs with other non-ABC classes.
- DON'T mix `@abstractmethod` with `@staticmethod`.
- Use
  ```python
  @property
  @abstractmethod
  ```
  for abstract properties.

## TODOs

For TODOs, use a format `# TODO({Name}, {month}/{year}): {full_sentence_comment}`, e.g. `# TODO(Michal, 08/2023): Address in the next PR.`.

## Comments

For comments outside of docstrings, use full sentences and proper punctuation. E.g. `# This is a comment.` instead of `# this is a comment`.

## Assertions

DON’T use assertions for user errors.

DON’T assert that a variable follows its typehint.

## Positional vs. Keyword Arguments

Always use keyword arguments when calling functions, except for single-argument functions.

## Docstrings

If using docstrings, use the google style guide and triple quotes. Use `Args:` and `Returns:` sections. Don't repeat the type in the description.
Any example:

```python
def foo(bar: int) -> str:
    """Converts an integer to a string.

    Args:
        bar: The bar to convert.

    Returns:
        The converted bar.
    """
    return str(bar)
```

## Typing

Always use type hints, such that mypy passes.

Use Python > 3.10 syntax e.g. `list[int | None]` instead of `List[Optional[int]]`. Only when the latter is needed for Python < 3.10 compatibility, should you add `from __future__ import annotations` at the top of the file.

Use `Sequence` and `Mapping` instead of `list` and `dict` for immutable types. Import them from `collections.abc`.

Enforce abstract inputs and concrete outputs despite specified type in prompts. See this example:

```python
def add_suffix_to_list(lst: Sequence[str], suffix: str) -> list[str]:
    return [x + suffix for x in lst]
```

Be specific when ignoring type errors, e.g. `# type: ignore[no-untyped-call]` instead of `# type: ignore`.

Type all PyTorch tensors with `from torch import Tensor`. Note that things like `FloatTensor` and `LongTensor` should NOT be used.

## Testing

Always use pytest, never unittest.

When testing a class named `MyClass`, put all tests under a class named `TestMyClass`.

When testing a function or method, name it `test_{method_name_with_underscores}`.
E.g. the test for `_internal_function` is named `test__internal_function`.
E.g. the test for `MyClass.my_method` is named `TestMyClass.test_my_method`.

When testing a special case of a function or method append a `__{special_case}` to the test name.
E.g. the test for the function `compute_mean(arr: list[float])` for the empty array case
should be named `test_compute_mean__empty_array`.

# Markdown documentation style guidelines

Make sure Lines are wrapped at 88 characters.
