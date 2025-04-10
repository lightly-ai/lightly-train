import pytest
from pytest_examples import CodeExample, EvalExample, find_examples

"""
This is not a test, but provides the necessary commands to format and check
python code in code blocks in the markdown files of the docs.
It is called by `make format` and `make format-check`.

Code blocks containing 'skip_ruff' are skipped.
Additionally, as of `pytest-examples` v0.0.15 (2024-11-20), code blocks inside tabs
are not detected (see https://github.com/pydantic/pytest-examples/issues/51).
"""


@pytest.mark.parametrize("example", find_examples("docs"), ids=str)
def test_format_code_in_docs(example: CodeExample, eval_example: EvalExample) -> None:
    if "skip_ruff" in example.prefix:
        pytest.skip("Skip this example")
    eval_example.format_ruff(example)


@pytest.mark.parametrize("example", find_examples("docs"), ids=str)
def test_format_check_code_in_docs(
    example: CodeExample, eval_example: EvalExample
) -> None:
    if "skip_ruff" in example.prefix:
        pytest.skip("Skip this example")
    eval_example.lint_ruff(example)
